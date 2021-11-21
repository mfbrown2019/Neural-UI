# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'model.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_model_window(object):
    
    
    def setupUi(self, model_window, models):
        
        
        self.models = models
        
        
        
        model_window.setObjectName("model_window")
        model_window.resize(800, 400)
        self.hyperparameters = QtWidgets.QComboBox(model_window)
        self.hyperparameters.setGeometry(QtCore.QRect(20, 20, 300, 30))
        self.hyperparameters.setMinimumSize(QtCore.QSize(0, 0))
        self.hyperparameters.setObjectName("hyperparameters")
        self.model_dropdown = QtWidgets.QComboBox(model_window)
        self.model_dropdown.setGeometry(QtCore.QRect(480, 20, 300, 30))
        self.model_dropdown.setObjectName("model_dropdown")
        self.activation_dropdown = QtWidgets.QComboBox(model_window)
        self.activation_dropdown.setGeometry(QtCore.QRect(480, 200, 300, 30))
        self.activation_dropdown.setObjectName("activation_dropdown")
        self.parameter_slider = QtWidgets.QSlider(model_window)
        self.parameter_slider.setGeometry(QtCore.QRect(20, 200, 300, 30))
        self.parameter_slider.setOrientation(QtCore.Qt.Horizontal)
        self.parameter_slider.setObjectName("parameter_slider")
        self.hyper_edit = QtWidgets.QLineEdit(model_window)
        self.hyper_edit.setGeometry(QtCore.QRect(20, 250, 300, 30))
        font = QtGui.QFont()
        font.setPointSize(24)
        self.hyper_edit.setFont(font)
        self.hyper_edit.setObjectName("hyper_edit")
        self.submit_button = QtWidgets.QPushButton(model_window)
        self.submit_button.setGeometry(QtCore.QRect(580, 330, 100, 50))
        self.submit_button.setObjectName("submit_button")
        self.close_button = QtWidgets.QPushButton(model_window)
        self.close_button.setGeometry(QtCore.QRect(680, 330, 100, 50))
        self.close_button.setObjectName("close_button")

        self.retranslateUi(model_window)
        QtCore.QMetaObject.connectSlotsByName(model_window)
        for x in self.models:
            print(x.name)
        
    def get_params(self):
        return self.hyper_edit.text()
    
    def retranslateUi(self, model_window):
        _translate = QtCore.QCoreApplication.translate
        model_window.setWindowTitle(_translate("model_window", "Form"))
        self.hyper_edit.setText(_translate("model_window", "63463"))
        self.submit_button.setText(_translate("model_window", "Submit"))
        self.close_button.setText(_translate("model_window", "Close"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    model_window = QtWidgets.QWidget()
    ui = Ui_model_window()
    ui.setupUi(model_window)
    model_window.show()
    sys.exit(app.exec_())
