# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'kNN_dialog_base.ui'
#
# Created: Tue Feb 14 12:16:50 2017
#      by: PyQt4 UI code generator 4.10.2
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_kNNDialogBase(object):
    def setupUi(self, kNNDialogBase):
        kNNDialogBase.setObjectName(_fromUtf8("kNNDialogBase"))
        kNNDialogBase.resize(670, 788)
        self.layoutWidget = QtGui.QWidget(kNNDialogBase)
        self.layoutWidget.setGeometry(QtCore.QRect(30, 20, 631, 431))
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.gridLayout_5 = QtGui.QGridLayout(self.layoutWidget)
        self.gridLayout_5.setMargin(0)
        self.gridLayout_5.setObjectName(_fromUtf8("gridLayout_5"))
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.label_2 = QtGui.QLabel(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 2)
        self.label = QtGui.QLabel(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 0, 0, 1, 2)
        self.label_3 = QtGui.QLabel(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 4, 0, 1, 2)
        self.multiband_dropdown = QtGui.QComboBox(self.layoutWidget)
        self.multiband_dropdown.setObjectName(_fromUtf8("multiband_dropdown"))
        self.gridLayout.addWidget(self.multiband_dropdown, 1, 0, 1, 1)
        self.SHP_dropdown = QtGui.QComboBox(self.layoutWidget)
        self.SHP_dropdown.setObjectName(_fromUtf8("SHP_dropdown"))
        self.gridLayout.addWidget(self.SHP_dropdown, 3, 0, 1, 1)
        self.save_lineEdit = QtGui.QLineEdit(self.layoutWidget)
        self.save_lineEdit.setObjectName(_fromUtf8("save_lineEdit"))
        self.gridLayout.addWidget(self.save_lineEdit, 5, 0, 1, 1)
        self.save_browse = QtGui.QToolButton(self.layoutWidget)
        self.save_browse.setObjectName(_fromUtf8("save_browse"))
        self.gridLayout.addWidget(self.save_browse, 5, 1, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.gridLayout_3 = QtGui.QGridLayout()
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.k_value = QtGui.QSpinBox(self.layoutWidget)
        self.k_value.setMinimum(1)
        self.k_value.setMaximum(10000)
        self.k_value.setObjectName(_fromUtf8("k_value"))
        self.gridLayout_3.addWidget(self.k_value, 0, 1, 1, 1)
        self.gridLayout_2 = QtGui.QGridLayout()
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.list_select_bands = QtGui.QListWidget(self.layoutWidget)
        self.list_select_bands.setObjectName(_fromUtf8("list_select_bands"))
        self.gridLayout_2.addWidget(self.list_select_bands, 2, 0, 1, 1)
        self.list_select_tobe_classified = QtGui.QListWidget(self.layoutWidget)
        self.list_select_tobe_classified.setObjectName(_fromUtf8("list_select_tobe_classified"))
        self.gridLayout_2.addWidget(self.list_select_tobe_classified, 2, 2, 1, 1)
        self.list_select_descriptors = QtGui.QListWidget(self.layoutWidget)
        self.list_select_descriptors.setObjectName(_fromUtf8("list_select_descriptors"))
        self.gridLayout_2.addWidget(self.list_select_descriptors, 2, 1, 1, 1)
        self.select_bands_label = QtGui.QLabel(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.select_bands_label.sizePolicy().hasHeightForWidth())
        self.select_bands_label.setSizePolicy(sizePolicy)
        self.select_bands_label.setObjectName(_fromUtf8("select_bands_label"))
        self.gridLayout_2.addWidget(self.select_bands_label, 1, 0, 1, 1)
        self.list_select_descriptors_label = QtGui.QLabel(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.list_select_descriptors_label.sizePolicy().hasHeightForWidth())
        self.list_select_descriptors_label.setSizePolicy(sizePolicy)
        self.list_select_descriptors_label.setObjectName(_fromUtf8("list_select_descriptors_label"))
        self.gridLayout_2.addWidget(self.list_select_descriptors_label, 1, 1, 1, 1)
        self.list_select_tobe_classified_label = QtGui.QLabel(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.list_select_tobe_classified_label.sizePolicy().hasHeightForWidth())
        self.list_select_tobe_classified_label.setSizePolicy(sizePolicy)
        self.list_select_tobe_classified_label.setObjectName(_fromUtf8("list_select_tobe_classified_label"))
        self.gridLayout_2.addWidget(self.list_select_tobe_classified_label, 1, 2, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_2, 1, 0, 1, 2)
        self.k_value_label = QtGui.QLabel(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.k_value_label.sizePolicy().hasHeightForWidth())
        self.k_value_label.setSizePolicy(sizePolicy)
        self.k_value_label.setObjectName(_fromUtf8("k_value_label"))
        self.gridLayout_3.addWidget(self.k_value_label, 0, 0, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout_3, 1, 0, 1, 1)
        self.checkBox_Load_when_finished = QtGui.QCheckBox(self.layoutWidget)
        self.checkBox_Load_when_finished.setObjectName(_fromUtf8("checkBox_Load_when_finished"))
        self.gridLayout_5.addWidget(self.checkBox_Load_when_finished, 2, 0, 1, 1)
        self.gridLayout_4 = QtGui.QGridLayout()
        self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))
        self.progressBar = QtGui.QProgressBar(self.layoutWidget)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName(_fromUtf8("progressBar"))
        self.gridLayout_4.addWidget(self.progressBar, 0, 0, 1, 1)
        self.button_box_ok_cancel = QtGui.QDialogButtonBox(self.layoutWidget)
        self.button_box_ok_cancel.setOrientation(QtCore.Qt.Horizontal)
        self.button_box_ok_cancel.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.button_box_ok_cancel.setObjectName(_fromUtf8("button_box_ok_cancel"))
        self.gridLayout_4.addWidget(self.button_box_ok_cancel, 0, 1, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout_4, 3, 0, 1, 1)

        self.retranslateUi(kNNDialogBase)
        QtCore.QObject.connect(self.button_box_ok_cancel, QtCore.SIGNAL(_fromUtf8("accepted()")), kNNDialogBase.accept)
        QtCore.QObject.connect(self.button_box_ok_cancel, QtCore.SIGNAL(_fromUtf8("rejected()")), kNNDialogBase.reject)
        QtCore.QMetaObject.connectSlotsByName(kNNDialogBase)

    def retranslateUi(self, kNNDialogBase):
        kNNDialogBase.setWindowTitle(_translate("kNNDialogBase", "kNN", None))
        self.label_2.setText(_translate("kNNDialogBase", "Training data(.SHP):", None))
        self.label.setText(_translate("kNNDialogBase", "Multiband input image (.TIF ):", None))
        self.label_3.setText(_translate("kNNDialogBase", "Save output:", None))
        self.save_browse.setText(_translate("kNNDialogBase", "...", None))
        self.select_bands_label.setText(_translate("kNNDialogBase", "Select bands:", None))
        self.list_select_descriptors_label.setText(_translate("kNNDialogBase", "Select descriptors:", None))
        self.list_select_tobe_classified_label.setText(_translate("kNNDialogBase", "Select parameter to be classified:", None))
        self.k_value_label.setText(_translate("kNNDialogBase", "Number of nearest neighbours (k):", None))
        self.checkBox_Load_when_finished.setText(_translate("kNNDialogBase", "Load into canvas when finished", None))

