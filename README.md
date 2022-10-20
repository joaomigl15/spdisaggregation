# Table of Contents
- [Code Usage](https://github.com/joaomigl15/spdisaggregation/blob/main/README.md#code-usage)
- [Disaggregated Results](https://github.com/joaomigl15/spdisaggregation/blob/main/README.md#disaggregated-results)
- [Datasets](https://github.com/joaomigl15/spdisaggregation/blob/main/README.md#datasets)
- [Contact](https://github.com/joaomigl15/spdisaggregation/blob/main/README.md#contact)
- [Citation](https://github.com/joaomigl15/spdisaggregation/blob/main/README.md#citation)
- [Acknowledgements](https://github.com/joaomigl15/spdisaggregation/blob/main/README.md#acknowledgements)

# Spatial Disaggregation



# Code Usage
The disaggregation method relies on a regression model to combine the ancillary data and produce disaggregation estimates. The entire procedure was implemented in the Python language, using the frameworks [scikit-learn](http://scikit-learn.org) and [Tensorflow](http://www.tensorflow.org).

## Spatial Disaggregation
To run the disaggregation algorithm, for the amount of withdrawals in ATMs:
```
./disaggregate_withdrawals.sh
```

## Dasymetric Mapping
To run dasymetric mapping with basis on population distribution, for the amount of withdrawals in ATMs:
```
./dasymapping_withdrawals.sh
```

## Pycnophylactic Interpolation
To run the pycnophylactic interpolation, for the amount of withdrawals in ATMs:
```
./pycno_withdrawals.sh
```

## Evaluation of Results
To evaluate results from .tif files available in '/results/Withdrawals/2Evaluate' at the level of intermediary regions, and using the pre-instaled QGis application, run:
```
cd code/evaluation
qgis -nologo --code zstats.py
screen -L -Logfile ev.txt -S evaluate
python3 evaluateResults.py
exit
python3 processEvaluation.py
```


# Disaggregated Results
The .tif files containing the disaggregated results reported in our study are available in:
- Baseline corresponding to smooth weighted interpolation - /results/Withdrawals/Baselines/smoothtd_200m.tif
- Self-training with random forests - /results/Withdrawals/Self-Training/Random Forests
- Self-training with the CNN - /results/Withdrawals/Self-Training/UNet
- Co-training with random forests and CNN - /results/Withdrawals/Co-Training/Target Smooth Td

Extra:
- Baseline corresponding to mass-preserving areal weighting - /results/Withdrawals/Baselines/massp_200m.tif
- Baseline corresponding to pycnophylactic interpolation - /results/Withdrawals/Baselines/pycno_200m.tif
- Baseline corresponding to weighted interpolation - /results/Withdrawals/Baselines/td_200m.tif
- Co-training with two CNNs differing in their initializations - /results/Withdrawals/Co-Training 1UNet 2UNet
- Ensembling of random forests and CNN - /results/Withdrawals/Co-Training Baselines/50Unet 50RF
- Co-training with random forests and CNN, with target as random forests 10It - /results/Withdrawals/Co-Training/Target RF 10It
- Co-training with random forests and CNN, with target as random forests 30It - /results/Withdrawals/Co-Training/Target RF 30It
- Co-training with random forests and CNN, with MSE as loss - /results/Withdrawals/Co-Training w Robust Losses/RMSE
- Co-training with random forests and CNN, with robust loss - /results/Withdrawals/Co-Training w Robust Losses/Adaptive

# Datasets


## Ancillary Data
Ancillary layers are required at the target resolution to guide the procedure. Each dataset is normalized to the target resolution, through averaging/summing cells in the cases where the original raster had a higher resolution, or through bicubic interpolation in the cases where the original raster had a lower resolution. The links for the ancillary data that we used in our study are provided next, although any dataset can be leveraged. The normalized datasets at a resolution of 200 meters are available in '/rasters/Withdrawals'.
- [GHSL Population grid - ghspg](https://ghsl.jrc.ec.europa.eu/)
- [GHSL Terrain development layer - bua](https://ghsl.jrc.ec.europa.eu/)
- [Copernicus Land cover layer - lc](https://land.copernicus.eu/pan-european)
- [Copernicus Human settlements layer - hs](https://land.copernicus.eu/pan-european)
- [VIIRS Nighttime lights layer - nl](http://gis.ngdc.noaa.gov/arcgis/rest/services/NPP_VIIRS_DNB)

## Shapefiles
The shapefiles containing the polygons for source and evaluation zones are available in:
- Source zones (NUTS III regions) - /shapefiles/Withdrawals/NUTSIII.shp
- Evaluation zones (municipalities) - /shapefiles/Withdrawals/MUNICIP.shp


## Aggregated Statistics
The csv files containing the aggregated statistics for source and evaluation zones are available in:
- Source zones (NUTS III regions) - /statistics/Withdrawals/NUTSIII.csv
- Evaluation zones (municipalities) - /statistics/Withdrawals/MUNICIP.csv


# Contact
João Monteiro - joao.miguel.monteiro@tecnico.ulisboa.pt

# Citation
If you use this algorithm in your research or applications, please cite:
Monteiro, J., Martins, B., Costa, M., & Pires, J. M. (2021). Geospatial Data Disaggregation through Self-Trained Encoder–Decoder Convolutional Models. ISPRS International Journal of Geo-Information, 10(9), 619.

# Acknowledgements
We gratefully acknowledge the support of NVIDIA Corporation, with the donation of the two Titan Xp GPUs used in our experiments. This work was also partially supported by Thales Portugal, through the Ph.D. scholarship of João Monteiro, by the EU H2020 research and innovation program, through the MOOD project corresponding to the grant agreement No. 874850, and by the Fundação para a Ciência e Tecnologia (FCT), under the MIMU project with reference PTDC/CCI-CIF/32607/2017 and also under the INESC-ID multi-annual funding (UIDB/50021/2020).
