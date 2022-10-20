# Table of Contents
- [Code Usage](https://github.com/joaomigl15/spdisaggregation/blob/main/README.md#code-usage)
- [Disaggregated Results](https://github.com/joaomigl15/spdisaggregation/blob/main/README.md#disaggregated-results)
- [Datasets](https://github.com/joaomigl15/spdisaggregation/blob/main/README.md#datasets)
- [Contact](https://github.com/joaomigl15/spdisaggregation/blob/main/README.md#contact)
- [Citation](https://github.com/joaomigl15/spdisaggregation/blob/main/README.md#citation)
- [Acknowledgements](https://github.com/joaomigl15/spdisaggregation/blob/main/README.md#acknowledgements)

# Spatial Disaggregation



# Code Usage

## Spatial Disaggregation
To run the disaggregation algorithm:
```
runDisaggregation.py
```

## Dasymetric Mapping
To run dasymetric mapping with basis on population distribution:
```
runDisaggregation.py
```

## Pycnophylactic Interpolation
To run the pycnophylactic interpolation:
```
runDisaggregation.py
```


# Disaggregated Results

# Datasets


## Ancillary Data
Ancillary layers are required at the target resolution to guide the procedure. Each dataset is normalized to the target resolution, through averaging/summing cells in the cases where the original raster had a higher resolution, or through bicubic interpolation in the cases where the original raster had a lower resolution. The links for the ancillary data that we used in our study are provided next, although any dataset can be leveraged. The normalized datasets at a resolution of 200 meters are available in '/rasters/Withdrawals'
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
