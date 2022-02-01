# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 12:17:39 2021

@author: ZengC
"""

import NN_prediction
import numpy as np
import pandas as pd
import os

##the line height water indexes 
def WaterIndex(Rrs,Bands=[681, 709, 753]):
    return Rrs[1]-Rrs[0]- (Bands[1]-Bands[0])/(Bands[2]-Bands[0])*(Rrs[2]-Rrs[0])

##given list of samples
DN_list=np.array([[0.027097443,	0.023360468,	0.030096319,	0.029640965,	0.045319773,	0.06489715,	0.067216955,	0.030723903,	0.030400945,	0.025962031,np.nan,		0.210847393,	np.nan,	0.089221708],
                  [0.01759918,0.017933078,0.021742476,0.021752099,0.030457929,0.039573926,0.041780472,0.024447482,0.02309229,0.02020702,0.029930705,0.01873157,0.017638113,0.003053678]])
#DN_list=np.array([[0.027097443,	0.023360468,	0.030096319,	0.029640965,	0.045319773,	0.06489715,	0.067216955,	0.030723903,	0.030400945,	0.025962031,np.nan,		0.210847393,	np.nan,	0.089221708]])

##load DN from file
work_dir=r'D:\Python_scripts\MODIS_NN_intensity'
data=pd.read_csv(os.path.join(work_dir,'MODIS_rhos_samples.txt'),skiprows=6,sep='\t')

if True: ##NN14B prediction
    path_net=r'C:\Users\ZengC\Python_scripts\MODIS_NN_intensity\NNmodels\LErie_NN_params_MODIS_rhos_14B_to_MERISL2_3B.csv'
    bandlist=[412,443,469,488,531,547,555,645,667,678,748,859,869,1240]
    DN_list=data.loc[~data['rhos_678'].isnull(), ['rhos_{0:03d}'.format(i) for i in bandlist]].to_numpy()
    nn_struct=NN_prediction.Load_NNparams(os.path.realpath(path_net))
    #reuse MATLAB NN model to predict: input KxN (K is sample #; N is input dimension; output: MxK, M is output dimension
    Rrs_pred=NN_prediction.NN_prediction(nn_struct,DN_list.reshape(-1,len(bandlist)))
    #Rrs_pred=Rrs_pred.reshape(3,(DN.shape)[0],(DN.shape)[1])
    #print(Rrs_pred)
    
    ##merge the MCI result bands to the raw data
    temp=pd.DataFrame(np.transpose(Rrs_pred),columns=['NN681', 'NN708', 'NN753'],index=data.index[~data['rhos_678'].isnull()])
    #data.join(temp)

#else:
if True:
    path_net=r'C:\Users\ZengC\Python_scripts\MODIS_NN_intensity\NNmodels\LErie_NN_params_MODIS_rhos_9B_to_MERISL2_3B.csv'
    bandlist_saturated = [412,443,469,488,531, 555, 645, 859, 1240]
    flt=~(data['rhos_531'].isnull()) & (data['rhos_678'].isnull())
    DN_list=data.loc[flt, ['rhos_{0:03d}'.format(i) for i in bandlist_saturated]].to_numpy()
    nn_struct = NN_prediction.Load_NNparams(os.path.realpath(path_net))
    # reuse MATLAB NN model to predict: input KxN (K is sample #; N is input dimension; output: MxK, M is output dimension
    
    #Rrs_pred_9B = NN_prediction.NN_prediction(nn_struct, DN_list[:,[0,1,2,3,4, 6, 7, 11, 13]].reshape(-1,len(bandlist_saturated)))
    Rrs_pred=NN_prediction.NN_prediction(nn_struct,DN_list.reshape(-1,len(bandlist_saturated)))
    
    #Rrs_pred_9B = Rrs_pred_9B.reshape(3, (DN.shape)[0], (DN.shape)[1]) 
    #print(Rrs_pred_9B)
    
    ##merge the MCI result bands to the raw data
    temp_saturated=pd.DataFrame(np.transpose(Rrs_pred),columns=['NN681', 'NN708', 'NN753'],index=data.index[flt])

temp_all=temp.append(temp_saturated)
temp_all['MCI']=WaterIndex(np.transpose(temp_all).to_numpy(),Bands=[681.25,708.75,753.75])
temp_all['Chl_NNMCI']=temp_all['MCI']*1457+2.895
rst=data[['Name', 'Latitude', 'Longitude','Date(yyyy-MM-dd)','Time(HH_mm_ss)', 'rhos_412', 'rhos_443', 'rhos_469', 'rhos_488',
       'rhos_531', 'rhos_547', 'rhos_555', 'rhos_645', 'rhos_667', 'rhos_678',
       'rhos_748', 'rhos_859', 'rhos_869', 'rhos_1240']].join(temp_all)   #temp.append(temp_saturated)
rst.to_csv(os.path.join(work_dir,"MDN_HabsGrab20190807_MODIS_rhos_NNMCI.csv"))

##MODIS standard L2 data derived chl using MDN
data_modisL2=pd.read_csv(os.path.join(work_dir,'HabsGrab20190807_MODIS_L2__Level 2_measurements.txt'),skiprows=6, sep='\t')
MODIS_L2_bands=[412,	443,	469,	488,	531,	547,	555,	645,	667,	678]
flt=~(data_modisL2['Rrs_488'].isnull() | data_modisL2['Rrs_645'].isnull())
MODIS_L2_Rrs=data_modisL2.loc[flt,['Rrs_{0:03d}'.format(i) for i in MODIS_L2_bands]].to_numpy() ##/np.pi,  DON'T need to /pi as it is already in sr^-1 unit
np.savetxt(os.path.join(work_dir,"HabsGrab20190807_MODIS_standardL2_Rrs.csv"),MODIS_L2_Rrs , delimiter=",")  

##load the insitu data to compare with MDN_chl
work_dir=r'C:\Users\ZengC\Exchange_files\LErie'
#data_insitu=pd.read_csv(os.path.join(work_dir,'2019 HABs grab data Chl_refined.csv'),sep='\t')

##export the merged dataset
##https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html#pandas.DataFrame.merge
data_merged=data_insitu.merge(rst[['Name','Chl_NNMCI']],how='left',on='Name')  #left_on=None, right_on=None, left_index=False, right_index=False, sort=False
data_merged.to_csv(os.path.join(work_dir,'MDN_HabsGrab20190807_MODIS_rhos_NNMCI_merged.csv'),sep=',') 
#[[0.02709744, 0.02336047, 0.03009632, 0.02964096, 0.04531977, 0.06721695, 0.0307239 , 0.21084739, 0.08922171]])