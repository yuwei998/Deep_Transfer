# -*- coding: utf-8 -*-

# MainWindow implementation generated from reading ui file 'first.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(514, 359)
        self.label = QtWidgets.QLabel(MainWindow)
        self.label.setGeometry(QtCore.QRect(30, 20, 181, 171))
        self.label.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"font: 18pt \"Agency FB\";\n"
"border-width: 1px;\n"
"border-style: solid;\n"
"border-color: rgb(0, 0, 0)")
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(MainWindow)
        self.pushButton.setGeometry(QtCore.QRect(70, 220, 91, 41))
        self.pushButton.setObjectName("pushButton")
        self.label_2 = QtWidgets.QLabel(MainWindow)
        self.label_2.setGeometry(QtCore.QRect(300, 20, 201, 171))
        self.label_2.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"font: 24pt \"Agency FB\";")
        self.label_2.setText("")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.pushButton_2 = QtWidgets.QPushButton(MainWindow)
        self.pushButton_2.setGeometry(QtCore.QRect(360, 220, 91, 41))
        self.pushButton_2.setObjectName("pushButton_2")

        self.retranslateUi(MainWindow)
        self.pushButton.clicked.connect(MainWindow.openimage)
        self.pushButton_2.clicked.connect(MainWindow.test)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "请输入一张图片"))
        self.pushButton.setText(_translate("MainWindow", "open image"))
        self.pushButton_2.setText(_translate("MainWindow", "Test"))

