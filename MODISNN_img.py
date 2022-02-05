# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:10:22 2021

this file provide functions to run MODISNN over MODIS Level 2 imagery after `l2gen` processing.

@author: ZengC
"""

import numpy as np
from osgeo import gdal
import NN_prediction
import os, os.path
import pathlib

def Write_netCDF4(path_L2, Rrs_pred,out_bands=['681','708','753'], strOutput=None):
    """ write MODISNN result as a netcdf4 format """
    import netCDF4 as nc
    toexclude = ['Rrs', 'Masks']  #remove irrelevant bands
    
    if strOutput==None: 
        strOutput=str( pathlib.Path(path_L2).parent / f"MODISNN_{pathlib.Path(path_L2).name}" )
        
    with nc.Dataset(path_L2) as src, nc.Dataset(strOutput, "w",zlib=True) as dst:
        #src=src_root.groups['geophysical_data']
        # copy global attributes all at once via dictionary
        dst.setncatts(src.__dict__)
        dst.product_name='MODISNN_{}'.format(src.product_name)
        # copy dimensions
        for name, dimension in src.dimensions.items():
            dst.createDimension(
                name, (len(dimension) if not dimension.isunlimited() else None))
        
        # copy all file data except for the excluded
        for s in src.groups:
            src_grp=src.groups[s]
            dst_grp=dst.createGroup(src_grp.name)
            
            dst_grp.setncatts(src_grp.__dict__)
            for name, dimension in src_grp.dimensions.items():
                dst_grp.createDimension(
                    name, (len(dimension) if not dimension.isunlimited() else None))
                
            for name, variable in src_grp.variables.items():
                if not any(s in name for s in toexclude) :
                    _ = dst_grp.createVariable(name, variable.datatype, variable.dimensions)
                    # copy variable attributes all at once via dictionary
                    dst_grp.variables[name].setncatts(src_grp.variables[name].__dict__)
                    dst_grp[name][:] = src_grp.variables[name][:]
    
        ##append the MODISNN bands
        data=dst.groups['geophysical_data']
        for idx in range(len(out_bands)):
            NN = data.createVariable('NN'+out_bands[idx],'f4',('number_of_lines', 'pixels_per_line'))
            NN[:]= Rrs_pred[idx,:,:] #np.rot90(Rrs_pred[1,:,:],2)
            NN.long_name='MODISNN derived bands: {}nm'.format(out_bands[idx])
            if out_bands[idx]=='MCI':  NN.long_name='Max Chlorophyll Index (MCI) calculated from MODISNN derived bands'
            if out_bands[idx]=='Chl':  NN.long_name='Chl (ug/L) based on Binding et al,2011. [chl=1457*mci+2.895]'
        
def MODISNN_img(path_L2,lakeID='LNA'):
    """the main function for MODISNN to process image
       input: 
           path_L2:  the path to the file of the MODIS data after SEADAS `l2gen` processing
                    this given file needs to include `rhos_xxx` bands.
           for exmaple:   strDir=r"TestData/A2011305183500.L2"  
           lakeID: the lake ID to be used to chose a specific trained lake NN model, currenlty only choice are from:['LErie','LW','LoW','LNA']
           'LErie': for training model of Lake Erie
           'LW': Lake Winnipeg 
           'LoW':  Lake of the Woods
           'LNA': a general model using samples from the above 3 lakes, may apply to other NorthAmerican lakes, but lower accuracy
       output:
           a new file of 3 bands simulating the OLIC/MERIS sensor on [681,708,753] nm; storing in the same folder.
    """
    bandlist=[412,443,469,488,531,547,555,645,667,678,748,859,869,1240]  
    bandlist_saturated = [412,443,469,488,531, 555, 645, 859, 1240]  #for the MCI direct model, over saturated area
    
    #ds = gdal.Open(path_L2)  #open the full dataset
    ds= gdal.Open('HDF5:"{}"://geophysical_data/rhos_412'.format(path_L2)) #open a band subdataset
    DN_list=np.ndarray([len(bandlist) ,ds.RasterYSize,ds.RasterXSize], dtype=float)
    for idx in range(len(bandlist)):
        f='HDF5:"{path_L2}"://geophysical_data/rhos_{band}'.format(path_L2=path_L2,band=bandlist[idx])
        DN= gdal.Open(f).ReadAsArray()
        DN[(DN > 1)| (DN <-0.1)]=np.nan 
        DN_list[idx,:,:]=DN
    
    #STEP1: apply NN14B model:
    path_net=r'./NNmodels/{}_NN_params_MODIS_rhos_14B_to_MERISL2_3B.csv'.format(lakeID) 
    if not os.path.isfile(path_net):  
        print('---------error: cannot find MODISNN parameter csv file:{}, will exit--------------'.format(path_net))
        return -1
    nn_struct=NN_prediction.Load_NNparams(path_net)
    #reuse MATLAB NN model to predict: input KxN (K is sample #; N is input dimension; output: MxK, M is output dimension
    Rrs_pred=NN_prediction.NN_prediction(nn_struct,np.transpose(DN_list.reshape(len(bandlist),-1)))
    
###------for user-built NN models with pyrenn for new lakes:-------------
    # import pyrenn
    # net=pyrenn.loadNN(r'./NNmodels/{}_NN_params_MODIS_rhos_149B_to_MERISL2_3B.csv'.format(lakeID))
    # Rrs_pred = pyrenn.NNOut(np.transpose(DN_list.reshape(len(bandlist),-1)),net)
 ###------for user-built NN models for new lakes:-------------
   
    Rrs_pred=Rrs_pred.reshape(3,(DN.shape)[0],(DN.shape)[1])  #reformat the data into image shape
    Rrs_pred[(Rrs_pred > 1) | (Rrs_pred < -0.1)] =np.nan  # remove invalid values, tolerance very small negative Rrs

    ##STEP2: apply NN9B band model, to fill the saturated pixels when possible
    path_net=r'./NNmodels/{}_NN_params_MODIS_rhos_9B_to_MERISL2_3B.csv'.format(lakeID) 
    bandlist_saturated = [412,443,469,488,531, 555, 645, 859, 1240]
    nn_struct = NN_prediction.Load_NNparams(path_net)
    # reuse MATLAB NN model to predict: input KxN (K is sample #; N is input dimension; output: MxK, M is output dimension
    Rrs_pred_9B = NN_prediction.NN_prediction(nn_struct, np.transpose(DN_list[[0,1,2,3,4, 6, 7, 11, 13], :, :].reshape(len(bandlist_saturated), -1)))
    Rrs_pred_9B = Rrs_pred_9B.reshape(3, (DN.shape)[0], (DN.shape)[1])  # reformat the data into image shape
    Rrs_pred_9B[(Rrs_pred_9B > 1) | (Rrs_pred_9B < -0.1)] = np.nan  # remove invalid values
    flt = np.sum(np.isnan(DN_list).astype(int), 0)  #find the pixels with at least one nan for any band
    np.copyto(Rrs_pred, Rrs_pred_9B, where=flt>0)
    
    ##calculate MCI and CHL
    wateridx=lambda Rrs,Bands: Rrs[1,:,:]-Rrs[0,:,:]- (Bands[1]-Bands[0])/(Bands[2]-Bands[0])*(Rrs[2,:,:]-Rrs[0,:,:])
    out_data = np.zeros(shape=(5,(DN.shape)[0],(DN.shape)[1]))
    out_data[3,:,:]=wateridx(Rrs_pred,[680.8,708.329,753.37]) #MCI bands
    out_data[0:3,:,:]=Rrs_pred
    out_data[4,:,:]=1457*out_data[3,:,:]+2.895  #chl
    
    ##export the result as a netcdf4 file
    Write_netCDF4(path_L2, out_data,out_bands=['681','708','753','MCI','Chl'])

    del DN  # clean intermediate data    
    ds=None

##if run as a main program,[rather than import as a model]
if __name__ == "__main__":
    
    ##parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='MODISNN process image MODIS L2 files generated from l2gen')
    parser.add_argument('path_L2', metavar='path_L2', type=pathlib.Path, 
                        help='the path to the MODIS L2 data after SEADAS `l2gen` processing;\
                              this L2 file, in netCDF4 format, needs to include `rhos_xxx` bands\
                              example data provided: TestData/A2011253190500.L2F')
    parser.add_argument('-L', '--lakeID', metavar='', type=str,
                        #choices=['LW', 'LoW', 'LErie','LNA'],
                        default='lNA',
                        help='the choice lakeID for training model selection')
    args = parser.parse_args()
    
    MODISNN_img(str(args.path_L2.absolute()),lakeID=args.lakeID)
    print("===image processed by MODISNN; result write to the input image folder: MODISNN_{inputname} ===")