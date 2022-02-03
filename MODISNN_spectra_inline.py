# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 13:28:17 2022

@author: ZengC
"""
import os
import numpy as np
import NN_prediction

##give one MODIS rhos spectra:bandlist=[412,443,469,488,531,547,555,645,667,678,748,859,869,1240]nm
DN_list=np.array([[0.0176,0.01796,0.021746,0.021756,0.03046,0.0395,0.0417,0.02444,0.023,0.0202,0.02993,0.0187,0.01763,0.00305]])

##load a trained NN model
lakeID='LErie'
path_net=r".\NNmodels\{}_NN_params_MODIS_rhos_14B_to_MERISL2_3B.csv".format(lakeID)
nn_struct=NN_prediction.Load_NNparams(os.path.realpath(path_net))

##run the model and calcuate MCI derived chl
Ref_pred=NN_prediction.NN_prediction(nn_struct,DN_list)
MCI=lambda Rrs,Bands: Rrs[1]-Rrs[0]- (Bands[1]-Bands[0])/(Bands[2]-Bands[0])*(Rrs[2]-Rrs[0])
mci=MCI(Ref_pred,[681.25,708.75,753.75])
mci_chl=1457*mci+2.895
print('==The chl derived from this spectra is: {} \u03BCg/L) =='.format(mci_chl))