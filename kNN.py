# -*- coding: utf-8 -*-
"""
/***************************************************************************
 kNN
                                 A QGIS plugin
 Pixel based k nearest neighbours supervised clasification.
                              -------------------
        begin                : 2017-02-13
        git sha              : $Format:%H$
        copyright            : (C) 2017 by Ventspils University College
        email                : venta@venta.lv
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import QSettings, QTranslator, qVersion, QCoreApplication
from PyQt4.QtGui import QAction, QIcon, QFileDialog, QMessageBox
# Initialize Qt resources from file resources.py
import resources
# Import the code for the dialog
from qgis.core import QgsMapLayer
import gdal
from kNN_dialog import kNNDialog
import os.path
from osgeo import ogr, osr
import numpy
from supervisedClassification import supervisedClassification

class kNN:
    """QGIS Plugin Implementation."""
    sc=supervisedClassification()
    layer_listRasterSource = []     #Array of raster layers addresses
    layers = []                     #Array of raster layers
    layer_listOther = []            #Array of non raster layers
    layer_listOtherSource = []      #Array of non raster layers addresses
    pathSHP = ''                    #Path to SHP file (kNN)
    pathTIF = ''                    #Path to TIF file (kNN)
    pathMASK = ''                   #Path to MASK file
    BandsTIF = []                   #Array of selected bands (kNN)
    SelectedDescriptors = []        #Array of SelectedDescriptors
    SelectedDescriptorsNames = []   #Array of SelectedDescriptorsNames (kNN)
    k = 1                           #Init k value (kNN)
    SelectedToBeClass = []          #Item from SHP to be classified
    SelectedToBeClassNames = []     #Item from SHP to be classified names (kNN)
    outputPath = ''                 #Path where to save result (kNN)
    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """


        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'kNN_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)

            if qVersion() > '4.3.3':
                QCoreApplication.installTranslator(self.translator)


        # Declare instance attributes
        self.dlg=kNNDialog()
        self.actions = []
        self.menu = self.tr(u'&kNN')
        # TODO: We are going to let the user set this up in a future iteration
        self.toolbar = self.iface.addToolBar(u'kNN')
        self.toolbar.setObjectName(u'kNN')

        ###########Clears line###########
        self.dlg.save_lineEdit.clear()
        ###########Launches event when button clicked###########
        self.dlg.save_browse.clicked.connect(self.select_output_file)
        ###########Launches event when item in dropdown menu is changed###########
        self.dlg.multiband_dropdown.currentIndexChanged.connect(self.updatebands)
        # self.dlg.mask_dropdown.currentIndexChanged.connect(self.upadtemask)
        self.dlg.mask_dropdown.currentIndexChanged.connect(self.updateMASK)


        self.dlg.SHP_dropdown.currentIndexChanged.connect(self.updateSHP)
        self.dlg.list_select_descriptors.itemClicked.connect(self.updateDescriptors)
        self.dlg.list_select_tobe_classified.itemClicked.connect(self.updateToBeClassified)
        self.dlg.list_select_bands.itemClicked.connect(self.updateSelectedBands)
        ###########Disable save path editing###########
        self.dlg.save_lineEdit.setEnabled(False)
        ###########Set default tab (Basic not advanced)###########
        self.dlg.tabWidget.setCurrentIndex(0)
        ###Set default values for check boxes###
        self.dlg.FeatureWghtLine.setEnabled(False)
        self.dlg.Geo_dist_spinBox.setEnabled(False)
        self.dlg.mask_checkbox.setEnabled(True)
        ###Call update Feature Wght and GeoDistance on state change###
        self.dlg.FeatureWghtLine_checkBox.stateChanged.connect(self.updateFeatureWghtLine)
        self.dlg.Geo_Dist_Checkbox.stateChanged.connect(self.updateGeoDistance)

    def updateGeoDistance(self):
        ###Update geo distance checkbox###
        if (self.dlg.Geo_Dist_Checkbox.isChecked()):
            self.dlg.Geo_dist_spinBox.setEnabled(True)
        else:
            self.dlg.Geo_dist_spinBox.setEnabled(False)
    def updateFeatureWghtLine(self):
        ###Update FeatureWght checkbox###
        if (self.dlg.FeatureWghtLine_checkBox.isChecked()):
            self.dlg.FeatureWghtLine.setEnabled(True)
        else:
            self.dlg.FeatureWghtLine.setEnabled(False)
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('kNN', message)
    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        # Create the dialog (after translation) and keep reference

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.toolbar.addAction(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action
    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""
        ###########Executes when plugin is launched###########
        icon_path = ':/plugins/kNN/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'kNN clasifier'),
            callback=self.run,
            parent=self.iface.mainWindow())
    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&kNN'),
                action)
            self.iface.removeToolBarIcon(action)
        #Remove the toolbar
        del self.toolbar
    def select_output_file(self):
        filenamedir = QFileDialog.getExistingDirectory(self.dlg,"Open Directory","/home")
        self.dlg.save_lineEdit.setText(filenamedir)
        self.outputPath = filenamedir
    def updatebands(self):
        ###########Updates select bands field on input image field event###########
        ###Currently selected image###
        imnameInd=self.dlg.multiband_dropdown.currentIndex()
        ###Open file###
        imlay=gdal.Open(self.layer_listRasterSource[imnameInd])
        ###Returns rasterCount###
        rasterCount=imlay.RasterCount

        bandList=range(1,rasterCount+1)
        self.dlg.list_select_bands.clear()

        if (len(bandList) == 0):
            self.dlg.list_select_bands.addItems(str('1'))
        else:
            for i in range(0,len(bandList)):
                self.dlg.list_select_bands.addItem(str(bandList[i]))

        self.pathTIF =self.layer_listRasterSource[(self.dlg.multiband_dropdown.currentIndex())]
        self.BandsTIF = self.dlg.list_select_bands.selectedItems()
    
    def updateSHP(self):
        ###########Updates .SHP file contents###########
        ###Reference to selected SHP file from dropdown###
        selectedSHP = self.dlg.SHP_dropdown.currentIndex()
        ###Adress of selected shape file###
        daShapefile = self.layer_listOtherSource[selectedSHP]
        ###SHP file driver###
        driver = ogr.GetDriverByName('ESRI Shapefile')
        ###0 - read-only mode###
        dataSource = driver.Open(daShapefile, 0)

        ###########SHP field names###########
        daLayer = dataSource.GetLayer(0)
        layerDefinition = daLayer.GetLayerDefn()
        layer_listSHP = []

        for i in range(layerDefinition.GetFieldCount()):
            #Append field names to array
            layer_listSHP.append((layerDefinition.GetFieldDefn(i).GetName()))

        ###########Add contents to descriptor window###########
        ###Enable multiselect###
        self.dlg.list_select_descriptors.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection )
        ###Clear current list contents###
        self.dlg.list_select_descriptors.clear()
        ###Add contents to the list###
        self.dlg.list_select_descriptors.addItems(layer_listSHP)

        ###########Add contents to parameter window###########
        ###Enable multiselect###
        self.dlg.list_select_tobe_classified.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        ###Clear current list contents###
        self.dlg.list_select_tobe_classified.clear()
        ###Add contents to the list###
        self.dlg.list_select_tobe_classified.addItems(layer_listSHP)


        self.pathSHP = self.layer_listOtherSource[(self.dlg.SHP_dropdown.currentIndex())]
        #Selected descriptors
        self.list_select_descriptors =  self.dlg.list_select_descriptors.selectedItems()
    def updateMASK(self):
        imnameInd = self.dlg.mask_dropdown.currentIndex()
        self.pathMASK = self.layer_listRasterSource[(self.dlg.mask_dropdown.currentIndex())]
    def updateDescriptors(self):
        ###Selected descriptors###
        self.SelectedDescriptors = self.dlg.list_select_descriptors.selectedItems()
        self.SelectedDescriptorsNames = []
        ###Append items to the list###
        for i in range (0,len(self.SelectedDescriptors)):
            self.SelectedDescriptorsNames.append(self.SelectedDescriptors[i].text())
    def updateToBeClassified(self):
        ###Selected descriptors###
        self.SelectedToBeClass = self.dlg.list_select_tobe_classified.selectedItems()
        self.SelectedToBeClassNames = []

        ###Append items to list###
        for i in range (0,len(self.SelectedToBeClass)):
            self.SelectedToBeClassNames.append(self.SelectedToBeClass[i].text())
    def updateValueK(self):
        ###Get k value from field###
        self.k = self.dlg.k_value.value()
    def updateSelectedBands(self):
        ###Append image bands to the list###
        temp = []
        temp = self.dlg.list_select_bands.selectedItems()
        self.BandsTIF = []

        for i in range (0,len(temp)):
            self.BandsTIF.append(int(temp[i].text()))
    def run(self):
        """Run method that performs all the real work"""
        self.dlg.checkBox_Load_when_finished.setChecked(1)
        layers = self.iface.legendInterface().layers()
        layer_list = []

        ###Reinit globals###
        self.SelectedDescriptors = []
        self.SelectedToBeClass = []
        ###Show the dialog box###
        self.dlg.show()
        ###Add multiselect for bands list###
        self.dlg.list_select_bands.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)

        ###Define empty
        layer_listRaster = []

        ###Add layers in dropdown (raster type only, no .shp)
        self.layer_listRasterSource=[]
        for layer in layers:
            #Iterate over layers, get names and path
            if layer.type()==QgsMapLayer.RasterLayer:
                layer_listRaster.append(layer.name())				#Name
                self.layer_listRasterSource.append(layer.source())	#Path
        ###Clear multiband dropdown###
        self.dlg.multiband_dropdown.clear()
        self.dlg.mask_dropdown.clear()
        ###Add list to dropdown menu (multiband menu)###
        self.dlg.multiband_dropdown.addItems(layer_listRaster)
        self.dlg.mask_dropdown.addItems(layer_listRaster)

        ###Clear .SHP dropdown###
        self.dlg.SHP_dropdown.clear()
        ###Init empty###
        self.layer_listOther = []
        self.layer_listOtherSource = []

        for layer in layers:
            if layer.type() == QgsMapLayer.VectorLayer:
                self.layer_listOther.append(layer.name())
                self.layer_listOtherSource.append(layer.source())

        ###Add items to SHP  dropdown###
        self.dlg.SHP_dropdown.addItems(self.layer_listOther)
        ######################Run the dialog event loop######################

        result = self.dlg.exec_()
        ###########Executes when OK button is pressed###########
        if result :
            ###########Read k valu and clasification mode line###########
            self.k = self.dlg.k_value.value()
            categ = self.dlg.lineCat.text()
            prefix = os.path.basename(self.pathTIF)
            ###########Distance selection###########
            ###Default value is 0 but is switched to 1 if untouched###
            distSel = 0
            if self.dlg.ManDistRadioBtn.isChecked():
                distSel = 2
            else:
                distSel = 1
            ###########Set geo distance value and operation flag###########
            geoDist_En = False
            if self.dlg.Geo_Dist_Checkbox.isChecked():
                geoDist = self.dlg.Geo_dist_spinBox.value()*1000
                geoDist_En = True
            else:
                geoDist = 0
                geoDist_En = False

            ###########Standartization check###########
            ###Standartization disabled(0) by default###
            Std_En = 0
            WghtError = 0
            Wght_En = 0
            Mask_En = 0


            #Mask_En check
            if self.dlg.mask_checkbox.isChecked():
                Mask_En = 1
            else:
                Mask_En = 0
            ###Check if Standartization is enabled in GUI###
            if self.dlg.Stand_En_Checkbox.isChecked():
                Std_En = 1
            else:
                Std_En = 0
            ###########Weighted coef.###########
                Wght_En = 0     #Weighting disabled(0) by default
                WghtError = 0   #Error clear
            ###########If is checked do following###########
            if self.dlg.FeatureWghtLine_checkBox.isChecked():
                ###Get values from field FeatureWghtLine###
                FeatWght = self.dlg.FeatureWghtLine.text()
                ###Remove ',' symbols###
                FeatWght = FeatWght.split(",")
                ###Check if number of coef. == number of selected bands###
                if len(FeatWght) == len(self.BandsTIF):
                    Wght_En = 1
                    WghtError = 0
                else:
                    Wght_En = 0
                    WghtError = 1
            else:
                ###If box is not checked FeatWght empty and Wght_En == 0 (line 375)###
                FeatWght = ""

            ###Count symbols (',' not counted)###
            cnt = 0
            ###Signals that enetered mode is invalid###
            invalid_mode = 0

            for i in range(0,(len(categ))):
                if categ[i] != ',':
                    cnt = cnt + 1
                    if (str(categ[i]) == 'n' or str(categ[i]) == 'N' or str(categ[i]) == 'c' or str(categ[i]) == 'C'):
                        pass
                    else:
                        ###If entered chracter is Invalid###
                        invalid_mode = 1
                else:
                    pass
            ###########Get rid of ',' and get indexable array###########
            ###Holds field values###
            cat = []
            ###Removes ',' symbols###
            cat = categ.split(",")

            ###########Number of things to classify###########
            numoffiles = len(self.SelectedToBeClass)
            ###Disables processing of kNN at start###
            allowProcessing = 0
            ###Create empty error message(list)###
            error_msg = []

            ######################GUI input control######################
            if ((len(self.BandsTIF) >= 1 and len(self.SelectedDescriptors)) >=1 and len(self.SelectedToBeClass) >= 1 ):
                pass
            else:
                error_msg.append("Select at least one parameter in each list !\n\n")

            if (len(self.BandsTIF) == len(self.SelectedDescriptors)):
                pass
            else:
                error_msg.append("Number of selected bands must match number of selected descriptors !\n\n")

            if ((numoffiles == len(cat)) and (len(cat) >= 1)):
                pass
            else:
                error_msg.append("Number of parameters to be classified must match number of entered clasification modes !\n\n")

            if (len(self.outputPath) != 0):
                pass
            else:
                error_msg.append("Select save directory !\n\n")

            if len(cat) == cnt:
                pass
            else:
                error_msg.append("Error entering modes !\n\n")

            if invalid_mode == 1:
                error_msg.append("Invalid mode entered !\n\n")
                cat = []
                categ = []
            else:
                pass

            ###Wght error###
            if WghtError == 1:
                error_msg.append("Error entering weight coeficients !\n\n")
            else:
                pass

            error_txt = ''

            for e in range(0,len(error_msg)):               #Convert list to one error message
                error_txt = error_txt + error_msg[e]

            if len(error_msg) == 0 :
                allowProcessing = 1
                error_msg=[]
            else:
                QMessageBox.information(None,"Error",str(error_txt))
                self.run()


            ###########Check processing rule###########
            if allowProcessing==1:
                ###########Disables processing to prevent hazardous self.run()
                allowProcessing = 0
                self.sc.knn(self.pathTIF,self.pathSHP,self.BandsTIF,self.SelectedDescriptorsNames,self.k,self.SelectedToBeClassNames,self.outputPath,cat,self.iface,FeatWght,Wght_En,Std_En,distSel,geoDist,geoDist_En,self.pathMASK,Mask_En)

                if self.dlg.checkBox_Load_when_finished.isChecked():        #Add layers when plugin is done
                    for i in range (0,numoffiles):
                        temp = self.outputPath + str('\SC_') + prefix[:-4] + self.SelectedToBeClassNames[i] + "_" + str(self.k) + cat[i] + ".tif"
                        self.iface.addRasterLayer(temp, (('SC_' + prefix[:-4] + self.SelectedToBeClassNames[i] + "_" + str(self.k) + cat[i]) ))
                else:
                    pass
            else:
                pass
