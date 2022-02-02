# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 12:17:39 2021

@author: ZengC
"""

import os,pathlib
import numpy as np
import pandas as pd
import argparse

import NN_prediction

def WaterIndex(Rrs,Bands=[681, 709, 753]):
    """
    Rrs : input Rrs or Reflectance of the surface water.
    Bands : the 3 bands wavelength to define the line height (of the central band). The default is [681, 709, 753].
    -------
    Returns
    idx: The line height index such as MCI, or any other 3 band line height water index.
    """
    if np.any(np.isnan(Rrs)):  #data validation
        return np.nan
    else:
        idx=Rrs[1]-Rrs[0]- (Bands[1]-Bands[0])/(Bands[2]-Bands[0])*(Rrs[2]-Rrs[0])
    return idx

##setup the input arguments
parser = argparse.ArgumentParser(description='MODISNN process image: ESA .dim files')
parser.add_argument('path_modis_rhos_csv', metavar='path_modis_rhos', type=pathlib.Path, nargs=1,
                    help='the path to csv file of MODIS rhos bands, 14 MODIS bands [412nm to 1240nm] are required. \
                    \ninvalid band values as empty, lat/lon columns are optional; \
                    \nexample at: .\TestData\MODIS_rhos_samples.csv')
parser.add_argument('-L', '--lakeID', metavar='lakeID', nargs=1, type=str,
                    #choices=['LW', 'LoW', 'LErie','general'],
                    default=['general'],
                    help='the choice lakeID for training model selection')
args = parser.parse_args()
path_modis_rhos=args.path_modis_rhos[0]  #".\TestData\MODIS_rhos_samples.csv"
lakeID=args.lakeID[0]   #'LErie'
#print('path:',path_modis_rhos,'lakeID: ',lakeID)

## initialization and load data
bandlist=[412,443,469,488,531,547,555,645,667,678,748,859,869,1240]  ##NN14B
bandlist_saturated = [412,443,469,488,531, 555, 645, 859, 1240]      ##NN9B at saturated pixels
data=pd.read_csv(path_modis_rhos)
flt=data['rhos_547'].isnull() | data['rhos_667'].isnull() |data['rhos_678'].isnull() | data['rhos_748'].isnull() | data['rhos_869'].isnull()##separate the spectra with some missing bands

##NN14B prediction
path_net=r"NNmodels\{}_NN_params_MODIS_rhos_14B_to_MERISL2_3B.csv".format(lakeID)
if not os.path.exists(path_net): 
    print("---error: input training model {} cannot be found, program will exit ---".format(path_net))
    import sys
    sys.exit()
DN_list=data.loc[~flt, ['rhos_{0:03d}'.format(i) for i in bandlist]].to_numpy()
nn_struct=NN_prediction.Load_NNparams(os.path.realpath(path_net))
#reuse MATLAB NN model to predict: input KxN (K is sample #; N is input dimension; output: MxK, M is output dimension
Rrs_pred=NN_prediction.NN_prediction(nn_struct,DN_list.reshape(-1,len(bandlist)))
##create a temp dataframe
temp=pd.DataFrame(np.transpose(Rrs_pred),columns=['NN681', 'NN708', 'NN753'],index=data.index[~flt])
temp['NN#B']=14  ##NN14B result

##NN9B prediction
path_net=r"NNmodels\{}_NN_params_MODIS_rhos_9B_to_MERISL2_3B.csv".format(lakeID)
DN_list=data.loc[flt, ['rhos_{0:03d}'.format(i) for i in bandlist_saturated]].to_numpy()
nn_struct = NN_prediction.Load_NNparams(os.path.realpath(path_net))
# reuse MATLAB NN model to predict: input KxN (K is sample #; N is input dimension; output: MxK, M is output dimension
#Rrs_pred_9B = NN_prediction.NN_prediction(nn_struct, DN_list[:,[0,1,2,3,4, 6, 7, 11, 13]].reshape(-1,len(bandlist_saturated)))
Rrs_pred=NN_prediction.NN_prediction(nn_struct,DN_list.reshape(-1,len(bandlist_saturated)))
##create a temp dataframe
temp_saturated=pd.DataFrame(np.transpose(Rrs_pred),columns=['NN681', 'NN708', 'NN753'],index=data.index[flt])
temp_saturated['NN#B']=9  ##NN9B result

##combine and export the result
rst=pd.concat([temp, temp_saturated])
rst['MCI']=WaterIndex(np.transpose(rst).to_numpy(),Bands=[681.25,708.75,753.75])
rst['Chl']=rst['MCI']*1457+2.895

##export the new csv:
path_modis_nn=pathlib.Path(path_modis_rhos).parent / f"MODISNN_{pathlib.Path(path_modis_rhos).name}"
data.join(rst).to_csv(path_modis_nn,index=False)
print("===spectra processed by MODISNN; result write to: {} === ".format(path_modis_nn))