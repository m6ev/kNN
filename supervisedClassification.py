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

import gdal
from osgeo import ogr, osr
import numpy
from additionalFunctions import additionalFunctions
import os
import math
import sys
import multiprocessing
import scipy.stats
import numpy.matlib
import time
import copy
from PyQt4.QtGui import QMessageBox
from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from time import sleep

class supervisedClassification:
	selectFS_msg = 0
	a = 0
	af=additionalFunctions()

	def knn(self,imageFN,trainingFN,whichBands,descriptorNames,k,variableNames,outputPath,modeList,iface,FeatWght,Wght_En,Std_En,distSel,geoDist,geoDist_En,mask,mask_en):
		numpy.set_printoptions(suppress=True)
		prefix = os.path.basename(imageFN)
		self.selectFS_msg = 0

		if type(iface) != int:
			###########Progress bar###########
			#Clear the message bar
			iface.messageBar().clearWidgets()
			#Set a new message bar
			progressMessageBar = iface.messageBar().createMessage("Progress")
			#Object type QProgressBar
			progress = QProgressBar()
			#Maximum is set to 100, making it easy to work with percentage of completion
			progress.setMaximum(100)
			#Alignment
			progress.setAlignment(Qt.AlignLeft|Qt.AlignVCenter)
			#Adds widget
			progressMessageBar.layout().addWidget(progress)
			#Pass the progress bar to the message Bar
			iface.messageBar().pushWidget(progressMessageBar,iface.messageBar().INFO)
		else:
			pass

		###########Main image###########
		###########Open .TIF###########
		imd1=gdal.Open(imageFN)
		############Get EPSG###########
		geot=imd1.GetGeoTransform()
		Projection = osr.SpatialReference()
		Projection.ImportFromWkt(imd1.GetProjectionRef())
		EPSG=(Projection.GetAttrValue("AUTHORITY", 1))
		###########Get .TIF parameters###########
		rows=(imd1.RasterYSize)						#Get Y dimension
		columns=(imd1.RasterXSize)					#Get X dimension
		bands = len(whichBands)						#Get number of bands needed
		allbands=numpy.zeros((rows,columns,bands))	#Init allbands array

		###########Create array with specified bands (image)###########
		for i in range(0,bands):
			temp=imd1.GetRasterBand(whichBands[i])
			allbands[:,:,i]=temp.ReadAsArray()

		if mask_en == 1:
			allmask=numpy.zeros((rows,columns))
			imgMask = gdal.Open(mask)
			temp = imgMask.GetRasterBand(1)
			allmask = temp.ReadAsArray()
		else:
			pass

		###########Create array with x and y. Used to get coordinates from sample data###########
		XY = ['x','y']
		###########Get sample and classesNumeric and sample_XY data###########
		sampleData, classesNumeric, sample_XY, avg, stdDev = self.readSample(trainingFN,descriptorNames,variableNames,Std_En,XY,geoDist_En,iface)
		###########Init resulting matrix###########
		rows = (allbands.shape[0])								#Dismensions of classifiable image
		columns = (allbands.shape[1])
		rez = numpy.zeros((rows,columns,len(variableNames)))	#3rd dimension lenght

		########################With progress bar########################
		if type(iface) != int:
			if geoDist_En == False:
				############Calculate result matrix###########
				for i in range(0,rows):
					time.sleep(0.005)
					val = (i/float(rows))*100
					progress.setValue(val)
					QtGui.QApplication.processEvents()
					for j in range(0,columns):
					###Mask check###
						if mask_en == 1 and allmask[i,j] == 1:
							#Vector of pixel along Z axis (pixel to be Classified)
							Zband=(allbands[i,j,:])

							if Std_En == 1:
								Zband = Zband - avg[0,:]
								Zband = numpy.divide(Zband,stdDev[0,:],dtype=float)
							else:
								pass

							sampleDataNew = numpy.zeros((sampleData.shape[0],sampleData.shape[1]))
							classesNumericNew = numpy.zeros((classesNumeric.shape[0],classesNumeric.shape[1]))
							sampleDataNew[:] = numpy.NAN
							classesNumericNew[:] =numpy.NAN
							#Calculated pixel
							temp = self.kNNProcedure(Zband,sampleData,classesNumeric,sampleDataNew,classesNumericNew,k,modeList,FeatWght,Wght_En,distSel,iface)
							#Matrix containing resuts for all pixels
							rez[i,j,:]=temp
						elif mask_en == 1 and allmask[i,j] == 0:
							pass
						else:
							#Vector of pixel along Z axis (pixel to be Classified)
							Zband=(allbands[i,j,:])

							if Std_En == 1:
								Zband = Zband - avg[0,:]
								Zband = numpy.divide(Zband,stdDev[0,:],dtype=float)
							else:
								pass

							sampleDataNew = numpy.zeros((sampleData.shape[0],sampleData.shape[1]))
							classesNumericNew = numpy.zeros((classesNumeric.shape[0],classesNumeric.shape[1]))
							sampleDataNew[:] = numpy.NAN
							classesNumericNew[:] =numpy.NAN
							#Calculated pixel
							temp = self.kNNProcedure(Zband,sampleData,classesNumeric,sampleDataNew,classesNumericNew,k,modeList,FeatWght,Wght_En,distSel,iface)
							#Matrix containing resuts for all pixels
							rez[i,j,:]=temp

			else:
				for i in range(0,rows):
					time.sleep(0.005)
					val = (i/float(rows))*100
					progress.setValue(val)
					QtGui.QApplication.processEvents()
					for j in range(0,columns):
						if mask_en == 1 and allmask[i,j] == 1:
							#Coordinates of current pixel
							pix_coord = self.imc2proj(geot,[i,j])
							#Vector of pixel along Z axis (pixel to be Classified)
							Zband=(allbands[i,j,:])

							if Std_En == 1:
								Zband = Zband - avg[0,:]
								Zband = numpy.divide(Zband,stdDev[0,:],dtype=float)
							else:
								pass
							#Get new set of training data based on geoDist
							sampleDataNew, classesNumericNew = self.selectFromSample(sampleData,classesNumeric,sample_XY,pix_coord,geoDist)
							#Calculated pixel
							temp = self.kNNProcedure(Zband,sampleData,classesNumeric,sampleDataNew,classesNumericNew,k,modeList,FeatWght,Wght_En,distSel,iface)
							#Matrix containing resuts for all pixels
							rez[i,j,:]=temp
						elif mask_en == 1 and allmask[i,j] == 0:
							pass
						else:
							#Coordinates of current pixel
							pix_coord = self.imc2proj(geot,[i,j])
							#Vector of pixel along Z axis (pixel to be Classified)
							Zband=(allbands[i,j,:])

							if Std_En == 1:
								Zband = Zband - avg[0,:]
								Zband = numpy.divide(Zband,stdDev[0,:],dtype=float)
							else:
								pass
							#Get new set of training data based on geoDist
							sampleDataNew, classesNumericNew = self.selectFromSample(sampleData,classesNumeric,sample_XY,pix_coord,geoDist)
							#Calculated pixel
							temp = self.kNNProcedure(Zband,sampleData,classesNumeric,sampleDataNew,classesNumericNew,k,modeList,FeatWght,Wght_En,distSel,iface)
							#Matrix containing resuts for all pixels
							rez[i,j,:]=temp
		else:
		########################Without progress bar########################
			if geoDist_En == False:
				############Calculate result matrix###########
				for i in range(0,rows):
					for j in range(0,columns):
						#Vector of pixel along Z axis (pixel to be Classified)
						if mask_en == 1 and allmask[i,j] == 1:
							Zband=(allbands[i,j,:])

							if Std_En == 1:
								Zband = Zband - avg[0,:]
								Zband = numpy.divide(Zband,stdDev[0,:],dtype=float)
							else:
								pass

							sampleDataNew = numpy.zeros((sampleData.shape[0],sampleData.shape[1]))
							classesNumericNew = numpy.zeros((classesNumeric.shape[0],classesNumeric.shape[1]))
							sampleDataNew[:] = numpy.NAN
							classesNumericNew[:] =numpy.NAN
							#Calculated pixel
							temp = self.kNNProcedure(Zband,sampleData,classesNumeric,sampleDataNew,classesNumericNew,k,modeList,FeatWght,Wght_En,distSel,iface)
							#Matrix containing resuts for all pixels
							rez[i,j,:]=temp
						elif mask_en == 1 and allmask[i,j] == 0:
							pass
						else:
							Zband=(allbands[i,j,:])

							if Std_En == 1:
								Zband = Zband - avg[0,:]
								Zband = numpy.divide(Zband,stdDev[0,:],dtype=float)
							else:
								pass

							sampleDataNew = numpy.zeros((sampleData.shape[0],sampleData.shape[1]))
							classesNumericNew = numpy.zeros((classesNumeric.shape[0],classesNumeric.shape[1]))
							sampleDataNew[:] = numpy.NAN
							classesNumericNew[:] =numpy.NAN
							#Calculated pixel
							temp = self.kNNProcedure(Zband,sampleData,classesNumeric,sampleDataNew,classesNumericNew,k,modeList,FeatWght,Wght_En,distSel,iface)
							#Matrix containing resuts for all pixels
							rez[i,j,:]=temp
			else:
				for i in range(0,rows):
					for j in range(0,columns):
						if mask_en == 1 and allmask[i,j] == 1:
							#Coordinates of current pixel
							pix_coord = self.imc2proj(geot,[i,j])
							#Vector of pixel along Z axis (pixel to be Classified)
							Zband=(allbands[i,j,:])

							if Std_En == 1:
								Zband = Zband - avg[0,:]
								Zband = numpy.divide(Zband,stdDev[0,:],dtype=float)
							else:
								pass
							#Get new set of training data based on geoDist
							sampleDataNew, classesNumericNew = self.selectFromSample(sampleData,classesNumeric,sample_XY,pix_coord,geoDist)
							#Calculated pixel
							temp = self.kNNProcedure(Zband,sampleData,classesNumeric,sampleDataNew,classesNumericNew,k,modeList,FeatWght,Wght_En,distSel,iface)
							#Matrix containing resuts for all pixels
							rez[i,j,:]=temp
						elif mask_en == 1 and allmask[i,j] == 0:
							pass
						else:
							#Coordinates of current pixel
							pix_coord = self.imc2proj(geot,[i,j])
							#Vector of pixel along Z axis (pixel to be Classified)
							Zband=(allbands[i,j,:])

							if Std_En == 1:
								Zband = Zband - avg[0,:]
								Zband = numpy.divide(Zband,stdDev[0,:],dtype=float)
							else:
								pass
							#Get new set of training data based on geoDist
							sampleDataNew, classesNumericNew = self.selectFromSample(sampleData,classesNumeric,sample_XY,pix_coord,geoDist)
							#Calculated pixel
							temp = self.kNNProcedure(Zband,sampleData,classesNumeric,sampleDataNew,classesNumericNew,k,modeList,FeatWght,Wght_En,distSel,iface)
							#Matrix containing resuts for all pixels
							rez[i,j,:]=temp

		############Write to result to .TIF############
		rasterOrigin = [geot[0],geot[3]]
		#For single parameter
		if(rez.shape[2] == 1):
			newfull = outputPath + str('\SC_') + prefix[:-4] + variableNames[0] + "_" + str(k) + modeList[0] + ".tif"
			if type(iface) != int:
				QMessageBox.information(None,"Done",str("Classification finshed "))
			else:
				print("Classification finshed")
			self.af.array2raster(newfull,rasterOrigin,geot[1],geot[5],rez[:,:,0],1,gdal.GDT_Float32,int(EPSG))
		#For multiple parameters
		else:
			for i in range(0,rez.shape[2]):
				add = variableNames[i]
				newfull = outputPath + str('\SC_') + prefix[:-4] + add + "_" + str(k) + modeList[i] + ".tif"
				self.af.array2raster(newfull,rasterOrigin,geot[1],geot[5],rez[:,:,i],1,gdal.GDT_Float32,int(EPSG))
			if type(iface) != int:
				QMessageBox.information(None,"Done",str("Classification finshed "))
			else:
				print("Classification finshed")
		self.knnLOL(sampleData,classesNumeric,variableNames,imageFN,outputPath,k,modeList,FeatWght,Wght_En,distSel,iface)

		############Release gdal############
		imd1 = None
		############Clear widgets############

		if type(iface) != int:
			iface.messageBar().clearWidgets()
		else:
			pass

		return rez
	def imc2proj(self,gt,pointc):
		#Return pixel Coordinates
		npoint=[0,0]
		npoint[0]=gt[0]+pointc[1]*gt[1]+pointc[0]*gt[2]
		npoint[1]=gt[3]+pointc[1]*gt[4]+pointc[0]*gt[5]

		if self.a == 0:
			self.a = 1
			print(npoint)
		else:
			pass

		return npoint
	def selectFromSample(self,sampleData,classesNumeric,sample_XY,tbc_XY,geoDist):
		###Select from training data based on Geodistance###
		###Pixels Geo Coordinates (to be classified)###
		Xc = tbc_XY[0]
		Yc = tbc_XY[1]
		###Boundries for valid sample data###
		Xs = Xc - geoDist
		Xe = Xc + geoDist
		Ys = Yc + geoDist
		Ye = Yc - geoDist
		###Init new arrays for worst case scenario if all points ar within the bounds###
		###This case is equal as if no GeoDistance would be specified###
		sampleDataN = numpy.zeros((sampleData.shape[0],sampleData.shape[1]))
		sampleDataN [:] = numpy.NAN
		classesNumericN = numpy.zeros((classesNumeric.shape[0],classesNumeric.shape[1]))
		classesNumericN [:] = numpy.NAN
		############Get valid sampleData and classesNumeric############
		for i in range(0,sampleData.shape[0]):
			if (sample_XY[i,0] > Xs) and (sample_XY[i,0] < Xe) and (sample_XY[i,1] < Ys) and (sample_XY[i,1] > Ye):
				sampleDataN[i,:] = sampleData[i,:]
				classesNumericN[i,:] = classesNumeric[i,:]
		###Removes all rows with nan values and get new sampleData and classesNumericN for current Pixel###
		###Differs for each pixel###
		sampleDataN = sampleDataN[~numpy.isnan(sampleDataN).any(axis=1)]
		classesNumericN = classesNumericN[~numpy.isnan(classesNumericN).any(axis=1)]
		###Return new sample data###
		return sampleDataN, classesNumericN
	def knnLOL(self,sampleData,classesNumeric,variableNames,imageFN,outputPath,k,modeList,FeatWght,Wght_En,distSel,iface):

		prefix = os.path.basename(imageFN)
		rows = sampleData.shape[0]
		rez_LOL = numpy.zeros((rows,classesNumeric.shape[1]))

		###Single PTBC rez matrix (single input)###
		for i in range(0,rows):
			tempPIX = sampleData[i,:]
			tempSample = numpy.delete(sampleData,i,0)
			tempNumeric = numpy.delete(classesNumeric,i,0)
			###Create NAN arrays to avoid error and use passed tempSample and tempNumeric###
			sampleDataNew = numpy.zeros((sampleData.shape[0],sampleData.shape[1]))
			classesNumericNew = numpy.zeros((classesNumeric.shape[0],classesNumeric.shape[1]))
			sampleDataNew[:] = numpy.NAN
			classesNumericNew[:] =numpy.NAN

			rez_LOL[i,:] = self.kNNProcedure(tempPIX,tempSample,tempNumeric,sampleDataNew,classesNumericNew,k,modeList,FeatWght,Wght_En,distSel,iface)

		accPath = str(outputPath) + str("\SC_") + prefix[:-4] + str("Acc") + str(".txt")
		f = open(accPath, 'w')

		for x in range (0,len(variableNames)):
			if modeList[x] == 'c' or modeList[x] == 'C':
				u=numpy.unique(classesNumeric[:,x])				# Vector of unique elements)
				classN=len(u)									# Number of elements
				confusionMatrix=numpy.zeros((classN,classN)) 	# Zero matrix

				a = (range(0,classN))						# 0, 1, 2, ....classN
				corrAns= copy.deepcopy(classesNumeric[:,x])	# DeepCopy
				knnAns= copy.deepcopy(rez_LOL[:,x])			# DeepCopy

				###Correct index###
				for e in range(0,classN):
					corrAns[classesNumeric[:,x]==u[e]]=a[e]
					knnAns[rez_LOL[:,x]==u[e]]=a[e]

				numpy.set_printoptions(suppress=True)
				###ConfusionMatrix fill###
				for n in range(0,rows):
					correctAnswer = corrAns[n]
					kNNResponse = knnAns[n]
					if math.isnan(correctAnswer):
						pass
					elif math.isnan(kNNResponse):
						pass
					else:
						confusionMatrix[int(kNNResponse),int(correctAnswer)]=confusionMatrix[int(kNNResponse),int(correctAnswer)]+1

				###########Parameter calculation	OA,UA,PA###########
				dia = confusionMatrix.diagonal()
				OA = dia.sum()/confusionMatrix.sum()
				PA = numpy.zeros((classN))
				UA = numpy.zeros((classN))

				for i in range (0,classN):
					P = confusionMatrix[i,i]/confusionMatrix[:,i].sum()
					U = confusionMatrix[i,i]/confusionMatrix[i,:].sum()
					if math.isnan(P):
						pass
					else:
						PA[i]=confusionMatrix[i,i]/confusionMatrix[:,i].sum()
					if math.isnan(U):
						pass
					else:
						UA[i]=confusionMatrix[i,i]/confusionMatrix[i,:].sum()

				###########KHAT calc###########
				cM_elements = confusionMatrix.sum()
				dia_n = len(dia)
				obs_acc = cM_elements*dia.sum()
				chance_aggr = 0

				for i in range(0,classN):
					chance_aggr = chance_aggr + confusionMatrix[i,:].sum()*confusionMatrix[:,i].sum()

				KHAT = (obs_acc - chance_aggr)/(cM_elements * cM_elements - chance_aggr)

				###########Write Accuracy calculation to .txt file###########
				f.write(str("Accuracy for :  " ) + " ' " + str(variableNames[x]) + " ' \n" + "MODE : C\n")
				f.write(str("PA = ") + str(PA) + '\n' )
				f.write(str("UA = ") + str(UA) + '\n' )
				f.write(str("OA = ") + str(OA) + '\n' )
				f.write(str("KHAT = ") + str(KHAT) + '\n\n' )
				f.write(str("CONF = ") + str(confusionMatrix) + '\n\n' )
			else:
				###########Only for continious clasification(N mode)###########
				###########RMSE calc###########
				RMSE_matrix = (rez_LOL[:,x] - classesNumeric[:,x])
				RMSE_matrix = numpy.multiply(RMSE_matrix,RMSE_matrix)
				RMSE = math.sqrt( RMSE_matrix.sum()/rez_LOL.shape[0])
				RMSEN = float(RMSE/(max(classesNumeric[:,x])-min(classesNumeric[:,x])))
				###########Write Accuracy calculation to .txt file###########
				f.write(str("Accuracy for :  " ) + " ' " + str(variableNames[x]) + " ' \n" + "MODE : N\n")
				f.write(str("RMSE = ") + str(RMSE) + '\n' )
				f.write(str("RMSEN = ") + str(RMSEN) + '\n\n' )
				f.write(str("max = ") + str((max(classesNumeric[:,x]))) + '\n\n' )
				f.write(str("min = ") + str((min(classesNumeric[:,x]))) + '\n\n' )
		###Close .TXT file###
		f.close()
	def kNNProcedure(self, toBeClassified, sampleData, classesNumeric,sampleDataNew,classesNumericNew,k,modeList,FeatWght,Wght_En,distSel,iface):
		#Check and assign sampleDataU
		if numpy.isnan(numpy.sum(sampleDataNew)):		#If array sum is NAN use full range
			sampleDataU = sampleData
			classesNumericU = classesNumeric
		else:
			if sampleDataNew.shape[0] >= k:				#If data in array is sufficient use New data
				sampleDataU = sampleDataNew
				classesNumericU = classesNumericNew
			else:
				if self.selectFS_msg == 0:				#Full scale error message
					sampleDataU = sampleData
					classesNumericU = classesNumeric
					if type(iface) != int:
						QMessageBox.information(None,"Error",str("No enought sample points in specified GeoDistance. Using full range."))
					else:
						print("No enought sample points in specified GeoDistance. Using full range.")
					self.selectFS_msg = 1
				else:
					sampleDataU = sampleData
					classesNumericU = classesNumeric


		rows = sampleDataU.shape[0]								  	#Rows in sampleData		#SHP field count
		columns = sampleDataU.shape[1]
		tbC = numpy.matlib.repmat(toBeClassified,rows,1)		  	#Prime for matrix operations #


		###Convert to float###
		for i in range(0,len(FeatWght)):
			FeatWght[i] = float(FeatWght[i])

		###Weighting disabled###
		if Wght_En == 0:
			if distSel == 2:
				distance = numpy.sum(numpy.absolute(sampleDataU-tbC),axis=1) 			#Distance calculation with numpy
			elif distSel == 1:
				distance = numpy.sqrt(numpy.sum(numpy.power(sampleDataU-tbC,2),axis=1))

		###Weighting enabled###
		elif Wght_En == 1:
			if distSel == 2:	#Manhattan
				aR=numpy.matlib.repmat(FeatWght,rows,1)
				distance = numpy.sum(numpy.multiply(aR,(numpy.absolute(sampleDataU-tbC))),axis=1)
			elif distSel == 1:	#Euclidian
				aR=numpy.matlib.repmat(FeatWght,rows,1)
				distance = numpy.sqrt(numpy.sum(numpy.power(numpy.multiply(aR,(sampleDataU-tbC)),2),axis=1))

		index = (numpy.argsort(distance))		#Soreted index for distance
		varC = classesNumericU.shape[1]			#Number of classifiable elements
		obj = numpy.zeros((k,varC))				#Init for k nearest
		response = numpy.zeros((1,varC))		#Return value declaration

		###Get k of nearest classesNumeric values###
		for i in range(0,k):
			obj[i,:] = classesNumericU[index[i],:]
		for i in range(0,varC):
			if str(modeList[i])=='c' or str(modeList[i])=='C' :
				###MODE C return most common class nummeric###
				mostcommon = (scipy.stats.mode(obj[:,i]))
				response[0,i] = mostcommon[0]
			elif str(modeList[i])=='n' or str(modeList[i])=='N':
				###Distance array for k nearest###
				kdistances = numpy.zeros(k)
				for z in range(0,k):								#Get k shortest distance values
					kdistances [z] = distance[index[z]]+0.000001	# +0.000....1 to avoid 0 situation
				temp1 = numpy.sum(1/kdistances)						#Sum array elements
				w = numpy.zeros((1,k))								#Init array of weight coef.
				for z in range (0,k):								#Build weight array
					w[0,z] = (1/kdistances[z])/temp1
				mp=numpy.multiply(w,obj[:,i])
				s=numpy.sum(mp)
				response[0,i] = s
			else:
				None
		return response
	def readSample(self,trainingFN,descriptorNames,variableNames,Std_En,XY,geoDist_En,iface):
		columns = (len(descriptorNames))				#Number of selected descriptorNames
		columnsVar = len(variableNames)
		daShapefile = trainingFN						#File location adress
		driver = ogr.GetDriverByName('ESRI Shapefile')	#SHP file driver
		dataSource = driver.Open(daShapefile, 0) 		#0 - read-only
														#1 - writeable.

		###########Check if .SHP is Opened corectley###########
		if dataSource is None:
			# None
			###Optional message via print or QMessageBox###
			if type(iface) != int:
				QMessageBox.information(None,"readSample",str("Could not open .SHP"))
			else:
				print("Could not open .SHP")
				#print 'Could not open %s' % (daShapefile)	#Print file stats (Opened or None)
		else:
			###Optional message via print or QMessageBox###
			#print 'Opened %s' % (daShapefile)
			layer = dataSource.GetLayer()							#Whole shape file contents
			featureCount = layer.GetFeatureCount()					#featureCount == rows in .SHP file

		###########Init numpy array for results###########
		sampleData = numpy.zeros((layer.GetFeatureCount(),columns))
		classesNumeric = numpy.zeros((layer.GetFeatureCount(),columnsVar))
		sampleXY = numpy.zeros((layer.GetFeatureCount(),2))

		###########Get classesNumeric and sampleData from layer###########
		if geoDist_En == True:	#If distance enabled read values x,y from table
			i=0
			for featureCount in layer:
				for z in range(0,columnsVar):
					classesNumeric[i][z]=featureCount.GetField(str(variableNames[z]))		#Data about parameter holding class data
				for j in range(0,columns):
					sampleData[i][j] = featureCount.GetField(str(descriptorNames[j]))		#Data about selected descriptors
				for k in range(0,2):
					sampleXY[i][k] = featureCount.GetField(str(XY[k]))
				i = i + 1
		else:	#If distance disabled read only descriptors and classesNumeric
			i=0
			for featureCount in layer:
				for z in range(0,columnsVar):
					classesNumeric[i][z]=featureCount.GetField(str(variableNames[z]))		#Data about parameter holding class data
				for j in range(0,columns):
					sampleData[i][j] = featureCount.GetField(str(descriptorNames[j]))		#Data about selected descriptors
				i = i + 1

		###########Exec if Standartization is enabled###########
		if Std_En == 1:
			avg = numpy.zeros((1,columns))			#Init as numpy array
			stdDev = numpy.zeros((1,columns))		#Init as numpy array
			###########Calculate avg and stdDev for each column###########
			for e in range(0,columns):
				avg[0,e] = numpy.mean(sampleData[:,e])				#Average value of column
				stdDev [0,e] = numpy.std(sampleData[:,e])			#stdDev value of column
			###########Use repmat to avoid for loop###########
			avg = numpy.matlib.repmat(avg,sampleData.shape[0],1)
			#QMessageBox.information(None,"STD",str(avg))
			stdDev = numpy.matlib.repmat(stdDev,sampleData.shape[0],1)

			###########Calculate Standartized sampleData###########
			sampleData = sampleData - avg
			sampleData = numpy.divide(sampleData,stdDev,dtype=float)
		###If not enabled do nothing###
		else:
			avg = 0
			stdDev = 0
		###Release data source###
		dataSource.Destroy()
		return sampleData, classesNumeric, sampleXY, avg, stdDev
