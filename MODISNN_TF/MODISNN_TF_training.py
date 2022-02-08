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
import KPI_stats
import tensorflow as tf

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
                    default=5,
                    help='the number of repeated training for avg coefficients, default:5')
parser.add_argument('-M', '--model', action='store_true',
                    help='flag to save trained model, default is False')
parser.add_argument('-V', '--verbose', action='store_true',
                    help='flag to verbose the training process, default is False')                    
parser.add_argument('-P', '--plot', action='store_true',
                    #choices=['True', 'False'],
                    #default='True',
                    help='flag to plot the training result, default is False')
args = parser.parse_args()  #['Training/LNA.pkl','-N',15,'-B',9,'-M'])

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
#np.random.seed(2020)
if args.NNbands==14:  
    flt_train=np.random.rand(len(MODIS_NN14B))<=0.7  #70% for training+cross validation
    X=MODIS_NN14B.loc[flt_train,:]
    Y=OLCI_NN14B.loc[flt_train,:]
    X_test=MODIS_NN14B.loc[~flt_train,:]
    Y_test=OLCI_NN14B.loc[~flt_train,:]
else:  #switch to NN9B 
    flt_train=np.random.rand(len(MODIS_NN9B))<0.7  #70% for training+cross validation
    X=MODIS_NN9B.loc[flt_train,:]
    Y=OLCI_NN9B.loc[flt_train,:]
    X_test=MODIS_NN9B.loc[~flt_train,:]
    Y_test=OLCI_NN9B.loc[~flt_train,:]

## ---- NN training   ---
# Create a `Sequential` model and add Dense layers 
model_list=list()
drop_rate=0.3
reg_factor=1e-8
regularizer=tf.keras.regularizers.L1(reg_factor)  
init=tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(X.shape[1],), name="MODISrhos"))
for idx in range(len(args.nodes)):
    model.add(tf.keras.layers.Dense(args.nodes[idx], activation='relu', name="dense{}".format(idx)))
#model.add(tf.keras.layers.Dense(64, activation='relu', name="dense1"))
#model.add(tf.keras.layers.Dense(64,kernel_regularizer=regularizer,kernel_initializer = init, activation='relu', name="dense1"))
#model.add(tf.keras.layers.Dropout(drop_rate))
#model.add(tf.keras.layers.Dense(128, activation='relu', name="dense2"))  #kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform'
#model.add(tf.keras.layers.Dense(64, activation='relu', name="dense3"))
#model.add(tf.keras.layers.Dense(64,kernel_regularizer=regularizer,kernel_initializer = init, activation='relu', name="dense3"))
#model.add(tf.keras.layers.Dropout(drop_rate))
model.add(tf.keras.layers.Dense(3,  name="3MCIbands")) #activation='linear',
#model.load_weights('MODISNN.h5')

##  -----optimizer ----
str_param="MODISNN Training TF2"   #'MSE-adam-batch64' #'batch64' #'MSE-adam'
#opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
#model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
#model.compile(loss='mae',optimizer=tf.keras.optimizers.Adam(0.001))

model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])  #optimizer='adam'
Wreset = model.get_weights()     ##store the param for reset
Ntraining=args.training  #10  #the # of repeated training 
print('=== repeat training (100%): ', end="")
for idx in range(0,Ntraining):
    flt_train=np.random.rand(len(X))<=0.7  #?% for training 
    history = model.fit(X.loc[flt_train,:],Y.loc[flt_train,:],batch_size=100,epochs=100,verbose=args.verbose)
    model_list.append({"idx":idx,"weights":model.get_weights(),"history":history} ) #store the model history
    #model.save('__test__.model')  #save the full keras model;  load later as: model=keras.models.load_model("__test__.model")
    model.set_weights(Wreset)  
    print('...{}'.format(int(idx*100/(Ntraining-1))), end="")
print('\n')

if args.model:  #save model
    file_model=args.TrainingFile.parent / "{lakeID}_NNTF_params_MODIS_rhos_{NNbands}_{node}_{output}.h5" \
        .format(lakeID=args.TrainingFile.name.replace(".pkl",""), NNbands=args.NNbands,node='_'.join((str(s) for s in args.nodes)), output=3)
    Wavg=list()
    for idx in range(len(model_list[0]['weights'])):
        params=[m['weights'][idx] for m in model_list]
        Wavg.append(np.mean(params,axis=0))
    model.set_weights(Wavg)
    model.save(file_model)
    print('===MODISNN TF training completed; a trained model written to: {} ==='.format(file_model))
    
## run the model validation
Y_pred_list=np.ndarray([Ntraining,Y_test.shape[0],Y_test.shape[1]], dtype=float)
for idx in range(0,Ntraining):
    model.set_weights(model_list[idx]['weights'])
    Y_pred_list[idx,:,:]=model.predict(X_test) 
    
## using avg spectra, convert model ouput (OLCI/MERIS bands) --> MCI --> chl    
Y_pred=np.median(Y_pred_list,axis=0)    
MCI_obs=WaterIndex(np.transpose(Y_test.to_numpy().astype(float)),  Bands=[680.8,708.329,753.37])
Chl_obs=1457*MCI_obs+2.895
MCI_pre=WaterIndex(np.transpose(Y_pred),Bands=[680.8,708.329,753.37])
Chl_pre=1457*MCI_pre+2.895
stats=KPI_stats.KPI_stats(Chl_obs,Chl_pre,metrics=('rmse', 'bias', 'mdape', 'rsquare'))

## ---  view the NN performance   --- 
if args.plot:
    ss=''  
    for ly in model.layers: ss+='%s,' % ly.input.shape[1]
    ss+='3' #output 3 bands
    plt.figure()  
    plt.title(str_param,fontweight='bold',fontsize=12,color= 'k')
    plt.plot(Chl_obs,Chl_pre,'b.')
    plt.text(0.7,0.05, " R$^2$={:.3f} \n RMSE={} \n BIAS={} \n MAPE={:.3f}".format
             (stats['rsquare'],sci_notation(stats['rmse']), sci_notation(stats['bias']),stats['mdape']),
             fontsize=12,weight='bold',transform=plt.gca().transAxes)
    plt.text(0.05, 0.9,'NN: {}'.format(ss),fontsize=12,weight='bold',transform=plt.gca().transAxes)
    plt.xlabel('observed Chl [\u03BCg/L]'),plt.ylabel('predicted Chl [\u03BCg/L]')
    plt.grid()
    plt.savefig(args.TrainingFile.parent / 'training_tf.svg')
    plt.show()
    print('===MODISNN plot completed; a result svg written to the input folder: training_tf.svg ===')