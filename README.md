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

use [`MODISNN_spectra_csv.py`]:
```
>>>python MODISNN_spectra_csv.py './TestData/MODIS_rhos_samples.csv' -L 'LErie'

>>>python MODISNN_spectra_csv.py -h
usage: MODISNN_spectra_csv.py [-h] [-L] path_modis_rhos
MODISNN applying to spectra from a csv table
positional arguments:
  path_modis_rhos  the path to csv file of MODIS rhos bands, 14 MODIS bands [412nm to 1240nm] are required. invalid band values as empty, lat/lon columns are optional; start
                   with provided example: ./TestData/MODIS_rhos_samples.csv
options:
  -h, --help       show this help message and exit
  -L , --lakeID    the choice lakeID for training model selection, default: 'LNA' for Lakes of North America 
```

### 2. input spectra inline
use [`MODISNN_spectra_inline.py`] , replace the spectra and pick a training model:
```
python MODISNN_spectra_inline.py
```

### 3. input spectra as an image

---
## Extra: [Optional]

### How to run the script with ESA `.dim` files?
ESA SNAP uses `.dim` file, which is different in file organization than popular `geotiff`, ERDAS `.img` etc. The `.dim` file organizes the file structure, dimensions, projections, etc for all the bands, while the bands stored in a separate folder with same name but ending with `.data`

An example script `MODISNN_img_dim.py` provided to process MODINN for ESA SNAP `.dim` file format specifically. 

run with commandline:
```
##test with an image given in "./TestData" for Lake Winnipeg
>>>python MODISNN_img_dim.py "./TestData/A2011253190500_NN.data" --lakeID "LW"

>>>python .\MODISNN_img_dim.py -h
usage: MODISNN_img_dim.py [-h] [-L lakeID] strDir
MODISNN process image: ESA .dim files
positional arguments:
  strDir                the path to the folder of the MODIS data after SEADAS `l2gen` processing; this folder needs to include `rhos_xxx.img` and `rhos_xxx.hdr` files.
options:
  -h, --help            show this help message and exit
  -L lakeID, --lakeID lakeID
                        the choice lakeID for training model selection
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
- join & filter the spectra from the two sensors.  using the script `***.py`.
- train the MODISNN model, using the script [`MODISNN_training.py`]
```
>>>python .\MODISNN_training.py Training/LNA.pkl -B 14 -T 15 -M -P -N 64 64
Iteration:  0           Error:  47352.16670898312       scale factor:  3.0
Iteration:  1           Error:  65.41165369868943       scale factor:  0.3
Iteration:  2           Error:  6.093200905484815       scale factor:  0.3
Iteration:  3           Error:  3.300948175016174       scale factor:  0.3
Iteration:  4           Error:  2.9215735852598375      scale factor:  0.3
Iteration:  5           Error:  2.836860392057827       scale factor:  0.3
Iteration:  6           Error:  2.8051699981198466      scale factor:  0.3
Iteration:  7           Error:  2.787749742485841       scale factor:  0.3
Iteration:  8           Error:  2.7755542476818205      scale factor:  0.3
Iteration:  9           Error:  2.7658275984845857      scale factor:  0.3
Iteration:  10          Error:  2.757547235728398       scale factor:  0.3
Iteration:  11          Error:  2.7502610527212106      scale factor:  0.3
Iteration:  12          Error:  2.74373364254917        scale factor:  0.3
Iteration:  13          Error:  2.7378239137088087      scale factor:  0.3
Iteration:  14          Error:  2.7324374348362612      scale factor:  0.3
Iteration:  15          Error:  2.73076121140133        scale factor:  0.03
Maximum number of iterations reached
===MODISNN training completed; a trained model written to: Training\LNA_NN_params_MODIS_rhos_14B_to_MERISL2_3B.csv ===
===MODISNN plot completed; a result svg written to the input folder: training.svg ===

>>>python .\MODISNN_training.py -h
usage: MODISNN_training.py [-h] [-P] [-N] [-B] TrainingFile

MODISNN training

positional arguments:
  TrainingFile       the path to the pickle file as the training dataset, example provided: Training/All_Lakes_Training_NN.pkl

options:
  -h, --help         show this help message and exit
  -P, --plot         flag to plot the training result, default is False
  -N , --Ntraining   the max number of iteration, default:15
  -B , --NNbands     select # of NN model input bands,only support 14B and 9B, default:14
```
- after training, the model will be exported as a `.csv` file. add 14B AND 9B NN models into the `./NNModels` folder of this repository for further use by assigning the `lakeID`
---
the example training result using the given `LNA.pkl`:
![MODISNN_training.svg](Training/MODISNN_training.svg "MODISNN_training")
