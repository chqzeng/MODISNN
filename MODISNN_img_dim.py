import numpy as np
from osgeo import gdal
from osgeo import gdal_array
import NN_prediction
import os, os.path

#import fileinput
#def GenHeader(strTemplate,strBand,strDir): #Generate a new header file for a given band
def GenHeader(strDir,strBand,width=None,height=None):
	#------------previously read from a template file, now change to load directly ----
	#with open(strTemplate,'r') as file:filedata=file.read()
    #filedata=filedata.replace('_BAND_',strBand)
	#with open(os.path.join(strDir,strTemplate.replace('_template',strBand)),'w') as file:
	#	file.write(filedata)
    #--------------------------------------------------
    filedata="""ENVI
description = Neural network NN predicted MERIS L2 reflectance at _BAND_ nm - Unit: 1 - Wavelength: {_BAND_}nm
samples = {_WIDTH_}
lines = {_HEIGHT_}
bands = 1
header offset = 0
file type = ENVI Standard
data type = 4
interleave = bsq
byte order = 1
band names =  NN{_BAND_} 
wavelength = {_BAND_}
data gain values = 1.0
data offset values = 0.0 """.format(_WIDTH_=width,_HEIGHT_=height,_BAND_=strBand)
    with open(os.path.join(strDir,'NN{}.hdr'.format(strBand)),'w') as file:
        file.write(filedata)

def WriteENVIoutput(strOutput,image_array):
    """ write a band to a file as ENVI format """
    driver = gdal.GetDriverByName('ENVI')
    rows, cols = image_array.shape
    bands=1
    outDs = driver.Create(strOutput, cols, rows, bands, gdal.GDT_Float32)  #GDT_Float32 GDT_Int16
    outBand = outDs.GetRasterBand(1)
    #outBand.SetNoDataValue(np.nan)
    outBand.WriteArray(image_array)
    
def bitorderinvert(x):
    import struct
    return struct.unpack('>f', struct.pack('=f', x))
    
def MODISNN_img_dim(strDir,lakeID='general'):
    """the main function for MODISNN to process image
       input: 
           strDir:  the path to the folder of the MODIS data after SEADAS `l2gen` processing
                    this given folder needs to include `rhos_xxx.img` and `rhos_xxx.hdr` files.
           for exmaple:   strDir=r"../MODIS_L2rhos_folder/A2011305183500.data/"  
           lakeID: the lake ID to be used to chose a specific trained lake NN model, currenlty only choice are from:['LErie','LW','LoW','general']
           'LErie': for training model of Lake Erie
           'LW': Lake Winnipeg 
           'LoW':  Lake of the Woods
           'general': a general model using samples from the above 3 lakes, may apply to other NorthAmerican lakes, but lower accuracy
       output:
           3 bands simulating the OLIC/MERIS sensor on [681,708,753] nm; storing in the same folder.
    """
    bandlist=[412,443,469,488,531,547,555,645,667,678,748,859,869,1240]  
    bandlist_saturated = [469, 555, 645, 859, 1240]  #for the MCI direct model, over saturated area
    path_img=os.path.join(strDir,'rhos_'+str(bandlist[0])+'.img')
    DN = gdal_array.LoadFile(path_img)
    DN_list=np.ndarray([len(bandlist) ,(DN.shape)[0],(DN.shape)[1]], dtype=float)
    
    #load each of the MODIS band
    for idx in range(len(bandlist)):
        path_img=os.path.join(strDir,'rhos_'+str(bandlist[idx])+'.img')
        DN= gdal_array.LoadFile(path_img)# Read raster data as numeric array from file
        DN[(DN > 1)| (DN <-0.1)]=np.nan  #remove invalid values
        DN_list[idx,:,:]=DN  #DN2ref(DN)

    #STEP1: apply NN14B model:
    path_net=r'./NNmodels/{}_NN_params_MODIS_rhos_14B_to_MERISL2_3B.csv'.format(lakeID) 
    if not os.path.isfile(path_net):  
        print('---------error: cannot find MODISNN parameter csv file:{}, will exit--------------'.format(path_net))
        return -1
    nn_struct=NN_prediction.Load_NNparams(path_net)
    #reuse MATLAB NN model to predict: input KxN (K is sample #; N is input dimension; output: MxK, M is output dimension
    Rrs_pred=NN_prediction.NN_prediction(nn_struct,np.transpose(DN_list.reshape(len(bandlist),-1)))
    Rrs_pred=Rrs_pred.reshape(3,(DN.shape)[0],(DN.shape)[1])  #reformat the data into image shape
    Rrs_pred[(Rrs_pred > 1) | (Rrs_pred < -0.1)] =-32767.0  # np.nan  # remove invalid values, tolerance very small negative Rrs

    ##STEP2: apply NN9B band model, to fill the saturated pixels when possible
    path_net=r'./NNmodels/{}_NN_params_MODIS_rhos_9B_to_MERISL2_3B.csv'.format(lakeID) 
    #print('=====debug: start 9B NN ===========')
    bandlist_saturated = [412,443,469,488,531, 555, 645, 859, 1240]
    nn_struct = NN_prediction.Load_NNparams(path_net)
    # reuse MATLAB NN model to predict: input KxN (K is sample #; N is input dimension; output: MxK, M is output dimension
    Rrs_pred_9B = NN_prediction.NN_prediction(nn_struct, np.transpose(DN_list[[0,1,2,3,4, 6, 7, 11, 13], :, :].reshape(len(bandlist_saturated), -1)))
    Rrs_pred_9B = Rrs_pred_9B.reshape(3, (DN.shape)[0], (DN.shape)[1])  # reformat the data into image shape
    Rrs_pred_9B[(Rrs_pred_9B > 1) | (Rrs_pred_9B < -0.1)] = -32767.0  # np.nan  # remove invalid values
    flt = np.sum(np.isnan(DN_list).astype(int), 0)  #find the pixels with at least one nan for any band
    np.copyto(Rrs_pred, Rrs_pred_9B, where=flt>0)

    del DN  # clean intermediate data

    ##prepare the reuslt file from the MODISNN output matrix
    bnd,col, row = Rrs_pred.shape
    Rrs_pred_rev = np.zeros(Rrs_pred.shape)
    for ibnd in range(bnd):# reverse the order
        for idx in range(col):
            for idy in range(row):
                Rrs_pred_rev[ibnd, idx, idy] = bitorderinvert(Rrs_pred[ibnd, idx, idy])[0]
    out_band_list=['681','708','753'] #,'MCI']  #correpsonding to actual MERSI bands: [681,708,753] #NN predicted MERIS band list, for further MCI
    #MERSI_bands=[681,708,753]

    #print('=====debug: write the NN result to file ===========')
    for idx in range(len(out_band_list)):  #write the NN predicted MERIS bands to replace some uncertainty layers
        strband=str(out_band_list[idx])
        path_img=os.path.join(strDir,'NN'+strband+'.img')
        if not os.path.isfile(path_img+'.bak'):  #keep a backup of the org unc band if not did so yet
            os.rename(path_img,path_img+'.bak')
            os.rename(path_img.replace('.img','.hdr'), path_img.replace('.img','.hdr') + '.bak')
        WriteENVIoutput(path_img,Rrs_pred_rev[idx,:,:])  #change the byte order to be consistent with img data in SNAP
        ##gdal_array.SaveArray(Rrs_pred[idx,:,:],path_img,'ENVI')  #save as 'img' format
        GenHeader(strDir,strband,height=Rrs_pred_rev.shape[1],width=Rrs_pred_rev.shape[2])  #str(MERSI_bands[idx])

##if run as a main program,[rather than import as a model]
if __name__ == "__main__":
    
    ##parse arguments
    import argparse
    import pathlib
    parser = argparse.ArgumentParser(description='MODISNN process image: ESA .dim files')
    parser.add_argument('strDir', metavar='strDir', type=pathlib.Path, nargs=1,
                        help='the path to the folder of the MODIS data after SEADAS `l2gen` processing;\
                              this folder needs to include `rhos_xxx.img` and `rhos_xxx.hdr` files.')
    parser.add_argument('-L', '--lakeID', metavar='lakeID', nargs=1, type=str,
                        #choices=['LW', 'LoW', 'LErie','general'],
                        default='general',
                        help='the choice lakeID for training model selection')
    args = parser.parse_args()
    
    #MODISNN_img_dim(r'./TestData/A2011253190500_NN.data/',lakeID='LW')
    #print('==test input arguments==: \n strDir: {} \n lakeID: {}'.format(str(args.strDir[0].absolute()),args.lakeID))
    #print("==directory exists? ",os.path.exists(str(args.strDir[0].absolute())))
    MODISNN_img_dim(str(args.strDir[0].absolute()),lakeID=args.lakeID[0])