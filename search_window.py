# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'search_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1100, 800)
        self.search_label = QtWidgets.QLabel(Form)
        self.search_label.setGeometry(QtCore.QRect(20, 50, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(40)
        self.search_label.setFont(font)
        self.search_label.setObjectName("search_label")
        self.filter_edit = QtWidgets.QLineEdit(Form)
        self.filter_edit.setGeometry(QtCore.QRect(160, 40, 550, 50))
        font = QtGui.QFont()
        font.setPointSize(40)
        self.filter_edit.setFont(font)
        self.filter_edit.setText("")
        self.filter_edit.setObjectName("filter_edit")
        self.advanced_button = QtWidgets.QPushButton(Form)
        self.advanced_button.setGeometry(QtCore.QRect(730, 40, 350, 50))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(90, 170, 164))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(79, 74, 94))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(90, 170, 164))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(90, 170, 164))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(90, 170, 164))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(79, 74, 94))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(90, 170, 164))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(90, 170, 164))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(90, 170, 164))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(148, 148, 148))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(90, 170, 164))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(90, 170, 164))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.advanced_button.setPalette(palette)
        font = QtGui.QFont()
        font.setPointSize(40)
        self.advanced_button.setFont(font)
        self.advanced_button.setStyleSheet("background-color: #5AAAA4; border-radius: 10px;\n"
"")
        self.advanced_button.setObjectName("advanced_button")
        self.scrollArea = QtWidgets.QScrollArea(Form)
        self.scrollArea.setGeometry(QtCore.QRect(20, 100, 1060, 680))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1058, 678))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.table_widget = QtWidgets.QTableWidget(self.scrollAreaWidgetContents)
        self.table_widget.setGeometry(QtCore.QRect(0, 0, 1060, 680))
        self.table_widget.setObjectName("table_widget")
        self.table_widget.setColumnCount(16)
        self.table_widget.setRowCount(10)
        self.table_widget.setHorizontalHeaderLabels(['ID', 'Title', 'Model', 'Activation','L1', 'L2', 'Dropout', 'Momentum', 'Learning Rate', 
                                                    'Epochs', 'Note', 'Training Photo', 'Heatmap', 'Train Loss', 'Train Accuracy', 'Date'])
        
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.search_label.setText(_translate("Form", "Search:"))
        self.filter_edit.setPlaceholderText(_translate("Form", "Filter"))
        self.advanced_button.setText(_translate("Form", "Advanced Search"))
