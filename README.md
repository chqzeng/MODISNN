# MODISNN
## About
From MODIS bands, use nueral network(NN) to predict MERIS/OLCI bands for chlorophyll-a concnetration modelling 

This repository is the implementation of the following publication:

- [Zeng,C. and Binding,C. Consistent Multi-Mission Measures of Inland Water Algal Bloom Spatial Extent Using MERIS, MODIS and OLCI. 2021. Remote Sensing 13(17):3349. DOI: 10.3390/rs13173349.](https://www.mdpi.com/2072-4292/13/17/3349)


## Setup
There are a few geospatial python packages are relied to run this program. It is recommended that you use `Anaconda` to build an exclusive python enviornment for MODISNN. here are some suggested steps if you are note familiar with Anaconda:
- install Anaconda, [Installation Guide](https://docs.anaconda.com/anaconda/install/)
- create a conda environment: `conda create -n modisnn python`
- enter the new enviroment: `conda activate modisnn`
- install the required packages: `conda install -c conda-forge matplotlib numpy pandas gdal netcdf4`

---

## Usage
The package can be cloned into a directory with:
`git clone https://github.com/chqzeng/MODISNN`
or you can manually download this repository and unzip to a local directory

within the `modisnn` conda environment or other similar setup, run the MODISNN in one of the following two approaches:
### 1. input spectra as a table
An example with input `csv` file [`example1_spectra_csv.py`]:
```
python example1_spectra_csv.py .\TestData\MODIS_rhos_samples.csv -L LErie
```

An inline example [`example2_spectra_inline.py`]:
```
import os
import numpy as np
import pandas as pd
import NN_prediction

##give one MODIS rhos spectra:bandlist=[412,443,469,488,531,547,555,645,667,678,748,859,869,1240]nm
DN_list=np.array([[0.0176,0.01796,0.021746,0.021756,0.03046,0.0395,0.0417,0.02444,0.023,0.0202,0.02993,0.0187,0.01763,0.00305]])

##load a treained NN model
lakeID='LErie'
path_net=r".\NNmodels\{}_NN_params_MODIS_rhos_14B_to_MERISL2_3B.csv".format(lakeID)
nn_struct=NN_prediction.Load_NNparams(os.path.realpath(path_net))

##run the model and calcuate MCI derived chl
Ref_pred=NN_prediction.NN_prediction(nn_struct,DN_list)
MCI=lambda Rrs,Bands: Rrs[1]-Rrs[0]- (Bands[1]-Bands[0])/(Bands[2]-Bands[0])*(Rrs[2]-Rrs[0])
mci=MCI(Ref_pred,[681.25,708.75,753.75])
mci_chl=1457*mci+2.895
print('==The chl derived from this spectra is: {} \u03BCg/L) =='.format(mci_chl))
```

### 2. input spectra as an image

---
## Extra: [Optional]

### How to run the script with ESA `.dim` files?
ESA SNAP uses `.dim` file, which is different in file organization than popular `geotiff`, ERDAS `.img` etc. The `.dim` file organizes the file structure, dimensions, projections, etc for all the bands, while the bands stored in a separate folder with same name but ending with `.data`

An example script `MODISNN_img_dim.py` provided to process MODINN for ESA SNAP `.dim` file format specifically. 

run with commandline:
```
##test with an image given in "./TestData" for Lake Winnipeg
python MODISNN_img_dim.py "./TestData/A2011253190500_NN.data" --lakeID "LW"
```
or use within a python script:
```
import MODISNN_img_dim as moidsnn
modisnn.MODISNN_img_dim("./TestData/A2011253190500_NN.data",lakeID="LW") 
```


### How to prepare MODIS rhos imagery?
- download MODIS L1 imagery from NASA oceancolor datacatalog
- run the L1 to L2 processing using   [l2gen](https://seadas.gsfc.nasa.gov/help-8.1.0/processors/ProcessL2gen.html), 
  suggesting you use the `SEADAS` software: SeaDAS-OCSSW
- An example command of `l2gen`:
  `l2gen   ...`

### How to train my own NN models?
in case you would like to apply MODISNN to a new lake and would like to improve the accruacy from the default `general` model I provide, you can collect some training dataset and train a new model for your lake of interest. specifically, you need to prepare MODIS +  MERIS/OLCI data for that lake when BOTH the sensors are available. 
- download MODIS L1 data and run `l2gen` to process toward `_rhos` as in the step above
- download OLCI/MERIS L2 data from ESA website
- prepare a list of grid points in your lake of interest; ideally at >5km for any of the two grid points to make sure samples are independant.
- run pixel extraction tools (e.g., PixEx tool in the ESA SNAP software) to extract spectra from both MODIS and ESA(MERIS/OLCI) sensors.
- join & filter the spectra from the two sensors.  using the script I provided `myscript.py`.
- train the MODISNN model, using the script I provided `myscript2.py`
- after training, the model will be exported as a `.csv` file. add 14band and 9band NN models into the `./NNModel` folder of this repository for further use by assign `lakeID`

