# MODISNN
## About
From MODIS bands, use nueral network(NN) to predict MERIS/OLCI bands for chlorophyll-a concnetration modelling 

This repository is the implementation of the following publication:
[Zeng,C. and Binding,C. Consistent Multi-Mission Measures of Inland Water Algal Bloom Spatial Extent Using MERIS, MODIS and OLCI. 2021. Remote Sensing 13(17):3349. DOI: 10.3390/rs13173349.](https://www.mdpi.com/2072-4292/13/17/3349)


## Setup
there are a few geospatial python packages are relied to run this program. It is recommended that you use `Anaconda` to build an exclusive python enviornment for MODISNN. here are some suggested steps if you are note familiar with Anaconda:
- install Anaconda, [Installation Guide](https://docs.anaconda.com/anaconda/install/)
- create an environment within anaconda: `create 
## Usage
The package can be cloned into a directory with:
`git clone https://github.com/chqzeng/MODISNN`
or you can manually download this repository and unzip to a local directory

and then run the MODISNN in one of the following two approaches:
### 1. input spectra as a table

### 2. input spectra as an image


## Extra: [Optional]

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

### how to run the script with ESA SNAP?
ESA SNAP use `.dim` files, which are different in file organization than popular `geotiff`, ERDAS `.img` etc. the `.dim` organizes the file structure, dimensions, projections, etc for all the bands, while the bands stored in a separate folder with same name but ending with `.data`
specifically for MODIS imagery, it applies a conversion when loading the imagery: `Rrs=0.05+2*10-6*DN`.

I provide an example script `` to process MODINN for ESA SNAP `.dim` file format specifically. use as is.
`MODISrhos_img_NN_Rrs_MERIS.py`

###

