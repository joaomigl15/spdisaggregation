import osgeoutils as osgu, pycno
import os


indicators = [['Withdrawals', 'NUTSIII', 'NUTSIII']]
popraster = 'ghspg_2015_200m.tif'


for indicator in indicators:
    print('--- Running pycnophylactic interpolation for the indicator', indicator[0])

    # Read rasters and shapefiles
    ds, rastergeo = osgu.readRaster(os.path.join('Rasters', indicator[0], popraster))
    nrowsds = ds.shape[1]
    ncolsds = ds.shape[0]

    fshapea = os.path.join('Shapefiles', indicator[0], (indicator[2] + '.shp'))
    fshape = osgu.copyShape(fshapea, 'pycno')
    fcsv = os.path.join('Statistics', indicator[0], (indicator[2] + '.csv'))

    # Merge shapefile and .csv file
    osgu.addAttr2Shapefile(fshape, fcsv, [indicator[2].upper()], encoding='utf-8')

    # Create datasets and run dasymetric mapping
    tempfileid = indicator[0]
    idsdataset = osgu.ogr2raster(fshape, attr='ID', template=[rastergeo, nrowsds, ncolsds])[0]
    polygonvaluesdataset, rastergeo = osgu.ogr2raster(fshape, attr='VALUE', template=[rastergeo, nrowsds, ncolsds])
    pycnodataset, rastergeo = pycno.runPycno(idsdataset, polygonvaluesdataset, rastergeo, tempfileid=tempfileid)

    # Write raster and remove temp files
    osgu.writeRaster(pycnodataset[:, :, 0], rastergeo, indicator[0] + '_pycnointerpolation.tif')
    osgu.removeShape(fshape)
