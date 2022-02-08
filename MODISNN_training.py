# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 10:13:46 2021
running a model training
@author: ZengC
"""

from matplotlib import pyplot as plt
import pickle
import pandas as pd
import numpy as np
import pyrenn
import KPI_stats

def sci_notation(number, sig_fig=2):
    """
    convert scientific numbers to "###x10^###" type
    """
    ret_string = "{0:.{1:d}e}".format(number, sig_fig)
    a,b = ret_string.split("e")
    b = int(b) #removed leading "+" and strips leading zeros too.
    return a + 'x10$^{' + str(b)+'}$'
def WaterIndex(Rrs,Bands=[681, 709, 753]):
    """
    cacluate line height water index
    """
    if np.any(np.isnan(Rrs)):  #data validation
        return np.nan
    else:
        idx=Rrs[1]-Rrs[0]- (Bands[1]-Bands[0])/(Bands[2]-Bands[0])*(Rrs[2]-Rrs[0])
    return idx

##configure the input arguments
import argparse
import pathlib
parser = argparse.ArgumentParser(description='MODISNN training')
parser.add_argument('TrainingFile', type=pathlib.Path, 
                    default='Training/LNA.pkl',
                    help='the path to the pickle file as the training dataset, \
                        example provided: Training/LNA.pkl')
parser.add_argument('-B', '--NNbands',metavar='', type=int,
                    default=14,choices=[14,9],
                    help='select # of NN model input bands,only support 14B and 9B, default:14')
parser.add_argument('-N','--nodes', metavar='',type=int, nargs='+',default=[64,64], help='hidden layer nodes#, default: 64 64')
parser.add_argument('-T', '--training', metavar='',type=int,
                    default=15,
                    help='the max number of iteration, default:15')
parser.add_argument('-M', '--model', action='store_true',
                    help='flag to save trained model, default is False')
parser.add_argument('-P', '--plot', action='store_true',
                    #choices=['True', 'False'],
                    #default='True',
                    help='flag to plot the training result, default is False')
args = parser.parse_args()  #['Training/LNA.pkl','-N',15,'-B',9,'-M'])
#print("=== debug: ",args.nodes)
with open(args.TrainingFile,'rb') as f:  
    LErie_OLCI, LoW_OLCI, LW_OLCI,LErie_MODIS,LoW_MODIS,LW_MODIS = pickle.load(f)
#combine datasets
OLCI=pd.concat([LErie_OLCI,LoW_OLCI,LW_OLCI], ignore_index=True)  #
MODIS=pd.concat([LErie_MODIS,LoW_MODIS,LW_MODIS], ignore_index=True)  #

##initiate the bands definition
MODISbands=['rhos_{}'.format(elem) for elem in [412,443,469,488,531,547,555,645,667,678,748,859,869,1240]]
MODIS9B=['rhos_{}'.format(elem) for elem in [412,443,469,488,531,555,645,859,1240]]
MERISbands=['M{0:02d}_rho_w_mean'.format(elem) for elem in [8,9,10]]  #MERIS MCI 3 bands: M\d{2}_rho_w_mean
OLCIbands=['Oa{0:02d}_reflectance_mean'.format(elem) for elem in [10,11,12]] #OLCI MCI 3 bands
OLCI_bands=['{}'.format(elem) for elem in [680.8,708.329,753.37]]

##prepare the dataset
flt_NN14B=np.all(MODIS.notna(),axis=1) & np.all(OLCI.notna(),axis=1) & np.all(OLCI>=0,axis=1)  ##all bands should be valid
flt_NN9B= np.all(MODIS.loc[:,MODIS9B].notna(),axis=1) & np.all(OLCI.notna(),axis=1) & np.all(OLCI>=0,axis=1) #only the NN9B bands
MODIS_NN14B=MODIS.loc[flt_NN14B,:].copy()  #a new dataset
OLCI_NN14B =OLCI.loc[flt_NN14B,:].copy()
MODIS_NN9B=MODIS.loc[flt_NN9B,MODIS9B].copy()  #a new dataset
OLCI_NN9B =OLCI.loc[flt_NN9B,:].copy()

##NN training data preparation
np.random.seed(2020)
if args.NNbands==14:  #switch to NN9B use "False"
    flt_train=np.random.rand(len(MODIS_NN14B))<=0.7  #70% for training+cross validation
    X=MODIS_NN14B.loc[flt_train,:]
    Y=OLCI_NN14B.loc[flt_train,:]
    X_test=MODIS_NN14B.loc[~flt_train,:]
    Y_test=OLCI_NN14B.loc[~flt_train,:]
    nodes=args.nodes
    nodes.insert(0,args.NNbands)  #input nodes
    nodes.append(3)    #output nodes
    net = pyrenn.CreateNN(nodes)  #[14,64,64,3]
else:
    flt_train=np.random.rand(len(MODIS_NN9B))<0.7  #70% for training+cross validation
    X=MODIS_NN9B.loc[flt_train,:]
    Y=OLCI_NN9B.loc[flt_train,:]
    X_test=MODIS_NN9B.loc[~flt_train,:]
    Y_test=OLCI_NN9B.loc[~flt_train,:]
    nodes=args.nodes
    nodes.insert(0,args.NNbands)  #input nodes
    nodes.append(3)
    net = pyrenn.CreateNN(nodes)#[14,64,64,3]

## ---- NN training   ---
str_param='LM:'+str(net['nn'])
net = pyrenn.train_LM(np.transpose(X.to_numpy().astype(float)),\
                         np.transpose(Y.to_numpy().astype(float)),\
                         net,verbose=True,k_max=args.training,E_stop=1e-5)
if args.model:  #save model
    file_model=args.TrainingFile.parent / "{lakeID}_NN_params_MODIS_rhos_{NNbands}B_to_MERISL2_3B.csv" \
        .format(lakeID=args.TrainingFile.name.replace(".pkl",""), NNbands=args.NNbands)
    pyrenn.saveNN(net,file_model)
    print('===MODISNN training completed; a trained model written to: {} ==='.format(file_model))

## convert model ouput (OLCI/MERIS bands) --> MCI --> chl    
Y_pred = pyrenn.NNOut(np.transpose(X_test.to_numpy().astype(float)),net)
MCI_obs=WaterIndex(np.transpose(Y_test.to_numpy().astype(float)),  Bands=[680.8,708.329,753.37])
MCI_pre=WaterIndex(Y_pred,Bands=[680.8,708.329,753.37])
Chl_obs=1457*MCI_obs+2.895
Chl_pre=1457*MCI_pre+2.895
stats=KPI_stats.KPI_stats(Chl_obs,Chl_pre,metrics=('rmse', 'bias', 'mdape', 'rsquare'))

## ---  view the NN performance   --- 
if args.plot:
    plt.figure()  
    plt.title(str_param,fontweight='bold',fontsize=12,color= 'k')
    plt.plot(Chl_obs,Chl_pre,'b.')
    plt.text(0.7,0.05, " R$^2$={:.3f} \n RMSE={} \n BIAS={} \n MAPE={:.3f}".format
             (stats['rsquare'],sci_notation(stats['rmse']), sci_notation(stats['bias']),stats['mdape']),
             fontsize=12,weight='bold',transform=plt.gca().transAxes)
    #plt.text(0.05, 0.9,'NN: {}'.format(str(net['nn'])),fontsize=12,weight='bold',transform=plt.gca().transAxes)
    plt.xlabel('MERIS/OLCI Chl [\u03BCg/L]'),plt.ylabel('MODISNN Chl [\u03BCg/L]')
    plt.grid()
    plt.savefig(args.TrainingFile.parent / 'training.svg')
    plt.show()
    print('===MODISNN plot completed; a result svg written to the input folder: training.svg ===')