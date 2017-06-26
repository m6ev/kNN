"""
/***************************************************************************
 kNN
                                 A QGIS plugin
 Pixel based k nearest neighbours supervised clasification.
                             -------------------
        begin                : 2017-02-13
        copyright            : (C) 2017 by Ventspils University College
        email                : venta@venta.lv
        git sha              : $Format:%H$
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/"""

import os
import sys
import glob
import gdal
import ogr
import numpy
from osgeo import osr, ogr
class additionalFunctions:

	def array2raster(self,newRasterfn,rasterOrigin,pixelWidth,pixelHeight,array,bands,type,epsg):
		cols = array.shape[1]
		rows = array.shape[0]
		# print array.shape
		originX = rasterOrigin[0]
		originY = rasterOrigin[1]
		driver = gdal.GetDriverByName('GTiff')
		outRaster = driver.Create(newRasterfn, cols, rows, bands, type)
		outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
		if bands==1:
			print "Writing array to tiff"
			outband = outRaster.GetRasterBand(1)
			outband.WriteArray(array)
		else:
			for i in range(0, bands):
				temparr=array[:,:,i]
				outband=outRaster.GetRasterBand(i + 1)
				outband.WriteArray(temparr)
		outRasterSRS = osr.SpatialReference()
		outRasterSRS.ImportFromEPSG(epsg)
		outRaster.SetProjection(outRasterSRS.ExportToWkt())
		outband.FlushCache()
