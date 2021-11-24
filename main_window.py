

from PyQt5 import QtCore, QtGui, QtWidgets
import model_window as mw
import search_analysis as sa
import settings_window as sw
import SimpleImagePreprocessor as sip
import data_window as dw
import AlexNet as AlexNet
import LeNet as LeNet
import LeNetReg as LeNetReg
import MiniGoogLeNet as MiniGoogLeNet
import MiniVGGNet as MiniVGGNet
import ResNet as ResNet
import VGGNet16 as VGGNet16
import VGGNet19 as VGGNet19
import hyper_classes as hc


import tensorflow as tf
from tensorflow.python.client import device_lib
import cv2
import os
import numpy as np
from imutils import paths
from random import randint

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import add

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.regularizers import l1_l2

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import InputLayer


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        
        self.buttons_background = '#5AAAA4'
        self.buttons_text = '#575366'
        self.run_background = '#575366'
        self.run_text = '#5AAAA4'
        self.items = []
        
        self.models = [AlexNet.AlexNet(), LeNet.LeNet(), LeNetReg.LeNetReg(), MiniVGGNet.MiniVGGNet(), ResNet.ResNet(), VGGNet16.VGGNet16(), VGGNet19.VGGNet19()]
        
        
        
        
        
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1100, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.data_button = QtWidgets.QPushButton(self.centralwidget)
        self.data_button.setGeometry(QtCore.QRect(20, 20, 400, 250))
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
        self.data_button.setPalette(palette)
        font = QtGui.QFont()
        font.setPointSize(50)
        self.data_button.setFont(font)
        self.data_button.setStyleSheet(f"background-color: {self.buttons_background};\n"
"border-radius: 10px;")
        self.data_button.setObjectName("data_button")
        self.model_button = QtWidgets.QPushButton(self.centralwidget)
        self.model_button.setGeometry(QtCore.QRect(680, 20, 400, 250))
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
        self.model_button.setPalette(palette)
        font = QtGui.QFont()
        font.setPointSize(50)
        self.model_button.setFont(font)
        self.model_button.setStyleSheet(f"background-color: {self.buttons_background};\n"
"border-radius: 10px;")
        self.model_button.setObjectName("model_button")
        self.settings_button = QtWidgets.QPushButton(self.centralwidget)
        self.settings_button.setGeometry(QtCore.QRect(20, 490, 525, 250))
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
        self.settings_button.setPalette(palette)
        font = QtGui.QFont()
        font.setPointSize(50)
        self.settings_button.setFont(font)
        self.settings_button.setStyleSheet(f"background-color: {self.buttons_background};\n"
"border-radius: 10px;")
        self.settings_button.setObjectName("settings_button")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(20, 280, 1060, 200))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(87, 83, 102))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(85, 166, 158))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(87, 83, 102))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(87, 83, 102))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(87, 83, 102))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(85, 166, 158))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(87, 83, 102))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(87, 83, 102))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(87, 83, 102))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(148, 148, 148))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(87, 83, 102))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(87, 83, 102))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.pushButton_5.setPalette(palette)
        font = QtGui.QFont()
        font.setPointSize(50)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setStyleSheet(f"background-color: {self.run_background};\n"
"border-radius: 10px;")
        self.pushButton_5.setObjectName("pushButton_5")
        self.history_button = QtWidgets.QPushButton(self.centralwidget)
        self.history_button.setGeometry(QtCore.QRect(555, 490, 525, 250))
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
        self.history_button.setPalette(palette)
        font = QtGui.QFont()
        font.setPointSize(50)
        self.history_button.setFont(font)
        self.history_button.setStyleSheet(f"background-color: {self.buttons_background};\n"
"border-radius: 10px;")
        self.history_button.setObjectName("history_button")
        self.logo = QtWidgets.QLabel(self.centralwidget)
        self.logo.setGeometry(QtCore.QRect(430, 20, 240, 250))
        self.logo.setText("")
        self.logo.setObjectName("logo")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1100, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.model_button.clicked.connect(self.open_model_window)
        self.data_button.clicked.connect(self.open_data_window)
        self.settings_button.clicked.connect(self.open_settings_window)
        self.pushButton_5.clicked.connect(self.run_model)
        self.history_button.clicked.connect(self.open_history_window)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.data_button.setText(_translate("MainWindow", "Data"))
        self.model_button.setText(_translate("MainWindow", "Model"))
        self.settings_button.setText(_translate("MainWindow", "Settings"))
        self.pushButton_5.setText(_translate("MainWindow", "RUN"))
        self.history_button.setText(_translate("MainWindow", "History"))
        
    def preprocess(self):

        imagePaths = list(paths.list_images(self.path))

        # chose the size for the image
        preprocessor = sip.SimpleImagePreprocessor(int(self.sizex), int(self.sizey))

        # initialize the data loader
        dataLoader = sip.SimpleImageDatasetLoader(1, preprocessors = [preprocessor.resize])

        # load the data into lists
        self.trainX, self.trainY = dataLoader.load(imagePaths, verbose = 100)
        
        
        what = []

        for i, label in enumerate(self.trainY):
            if label == 'rock':
                what.append(0)
            if label == 'paper':
                what.append(1)
            if label == 'scissors':
                what.append(2)

        self.trainY = what
        self.trainX = self.trainX.astype('float32')/255.0
        self.trainY = to_categorical(self.trainY, int(self.num_classes))
        
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(self.trainX, self.trainY, test_size = .2)
        
    def run_model(self):
        if self.rgb:
            self.depth = 3
        else:
            self.depth = 1
            
        model = LeNetReg.LeNetReg.build(int(self.sizex), int(self.sizey), self.depth, int(self.num_classes), lam1 = int(self.hyper), lam2 = 0, dropout = self.dropout, activation_input = self.activation_function)
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

        H = model.fit(self.trainX, self.trainY, validation_split = 0.20, batch_size = 128, epochs = int(self.epochs), verbose = 1)
        
    def open_model_window(self):
        model_window = QtWidgets.QDialog()
        ui = mw.Ui_model_window()
        ui.setupUi(model_window, self.models, self.buttons_background)
        model_window.exec_()
        
        self.dropout, self.epochs, self.hyper, self.hypername, self.activation_function, self.model_name = ui.send_everything()
        print(self.dropout, self.epochs, self.hyper, self.hypername, self.activation_function, self.model_name)
        
    def open_data_window(self):
        Form = QtWidgets.QDialog()
        ui = dw.Ui_Form()
        ui.setupUi(Form, self.buttons_background)
        Form.show()
        Form.exec_()
        
        self.path, self.sizex, self.sizey, self.dataaug, self.num_classes, self.rgb = ui.send_path_data()
        print(self.path, self.sizex, self.sizey, self.dataaug)
        self.preprocess()
        
    def open_settings_window(self):
        Form = QtWidgets.QDialog()
        ui = sw.Ui_Form()
        ui.setupUi(Form)
        Form.show()
        Form.exec_()
        self.buttons_background, self.buttons_text, self.run_background, self.run_text = ui.send_colors()
        self.history_button.setStyleSheet(f"background-color: {self.buttons_background};\n""border-radius: 10px;")
        self.pushButton_5.setStyleSheet(f"background-color: {self.run_background};\n""border-radius: 10px;")
        self.settings_button.setStyleSheet(f"background-color: {self.buttons_background};\n""border-radius: 10px;")
        self.model_button.setStyleSheet(f"background-color: {self.buttons_background};\n""border-radius: 10px;")
        self.data_button.setStyleSheet(f"background-color: {self.buttons_background};\n""border-radius: 10px;")
        
    def open_history_window(self):
  
        Form = QtWidgets.QDialog()
        ui = sa.Ui_Form()
        ui.setupUi(Form)
        Form.show()
        Form.exec_()
            
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
 