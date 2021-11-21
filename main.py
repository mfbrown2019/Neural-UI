from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QHeaderView

import sys

import model_window as mw

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.x = 800
        self.y = 600
        self.cat_list = []
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(self.x, self.y)
        print(self.x, self.y)
        MainWindow.setWindowTitle("Expense Report")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.add_rev_button = QtWidgets.QPushButton(self.centralwidget)
        self.add_rev_button.setGeometry(QtCore.QRect(225, 30, 150, 50))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.add_rev_button.setFont(font)
        self.add_rev_button.setStyleSheet("background-color: #cccccc;\n"
										"border-radius: 8px;")
        self.add_rev_button.setObjectName("add_rev_button")
        self.add_cat_button = QtWidgets.QPushButton(self.centralwidget)
        self.add_cat_button.setGeometry(QtCore.QRect(25, 30, 150, 50))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.add_cat_button.setFont(font)
        self.add_cat_button.setStyleSheet("background-color: #cccccc;\n"
										"border-radius: 8px;")
        self.add_cat_button.setObjectName("add_cat_button")
        self.add_exp_button = QtWidgets.QPushButton(self.centralwidget)
        self.add_exp_button.setGeometry(QtCore.QRect(425, 30, 150, 50))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.add_exp_button.setFont(font)
        self.add_exp_button.setStyleSheet("background-color: #cccccc;\n"
										"border-radius: 8px;")
        self.add_exp_button.setObjectName("add_exp_button")
        self.trans_money_button = QtWidgets.QPushButton(self.centralwidget)
        self.trans_money_button.setGeometry(QtCore.QRect(625, 30, 150, 50))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.trans_money_button.setFont(font)
        self.trans_money_button.setStyleSheet("background-color: #cccccc;\n"
											"border-radius: 8px;")
        self.trans_money_button.setObjectName("trans_money_button")
        self.category_combo_box = QtWidgets.QComboBox(self.centralwidget)
        self.category_combo_box.setGeometry(QtCore.QRect(25, 110, 351, 50))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.category_combo_box.setFont(font)
        self.category_combo_box.setEditable(True)
        self.category_combo_box.setObjectName("category_combo_box")
        self.category_table = QtWidgets.QTableWidget(self.centralwidget)
        self.category_table.setGeometry(QtCore.QRect(25, 190, 350, 190))
        self.category_table.setObjectName("category_table")
        self.category_table.setColumnCount(2)
        self.category_table.setRowCount(0)
        self.category_table.setHorizontalHeaderLabels(['Item', 'Price'])
        self.category_table.horizontalHeader().setStretchLastSection(True)
        self.category_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.total_line_edit = QtWidgets.QLabel(self.centralwidget)
        self.total_line_edit.setGeometry(QtCore.QRect(260, 410, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.total_line_edit.setFont(font)
        self.total_line_edit.setObjectName("total_line_edit")
        self.total_line_edit.setStyleSheet("background-color: white;\n"
											"border-radius: 5px;\n"
                                            "border-style: solid;\n"
                                            "border-width: 1px;")
        for i in self.cat_list:
            if self.category_combo_box.currentText() == i.name:
                self.total_line_edit.setText(i.amount)
        self.total_label = QtWidgets.QLabel(self.centralwidget)
        self.total_label.setGeometry(QtCore.QRect(30, 410, 141, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.total_label.setFont(font)
        self.total_label.setObjectName("total_label")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(0, 450, 800, 16))
        self.line.setStyleSheet("color: black;")
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.exit_button = QtWidgets.QPushButton(self.centralwidget)
        self.exit_button.setGeometry(QtCore.QRect(625, 480, 150, 50))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.exit_button.setFont(font)
        self.exit_button.setStyleSheet("background-color: #cccccc;\n"
									"border-radius: 8px;")
        self.exit_button.setObjectName("exit_button")
        self.pie_chart_label = QtWidgets.QLabel(self.centralwidget)
        self.pie_chart_label.setGeometry(QtCore.QRect(400, 110, 381, 331))
        self.pie_chart_label.setText("")
        self.pie_chart_label.setObjectName("pie_chart_label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        self.add_rev_button.setText(_translate("MainWindow", "Add Revenue"))
        self.add_cat_button.setText(_translate("MainWindow", "Add Category"))
        self.add_exp_button.setText(_translate("MainWindow", "Add Expense"))
        self.trans_money_button.setText(_translate("MainWindow", "Transfer Money"))
        self.category_combo_box.setCurrentText(_translate("MainWindow", ""))
        self.total_line_edit.setText(_translate("MainWindow", "0.00"))
        self.total_label.setText(_translate("MainWindow", "TOTAL"))
        self.exit_button.setText(_translate("MainWindow", "Exit"))
        
        
        

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
