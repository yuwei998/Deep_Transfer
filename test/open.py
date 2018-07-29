# -*- coding: utf-8 -*-
from PyQt5 import QtWidgets, QtGui
import sys
from first import Ui_MainWindow   # 导入生成first.py里生成的类
from PyQt5.QtWidgets import QFileDialog
from recognize import inference
import tensorflow as tf
class mywindow(QtWidgets.QWidget,Ui_MainWindow):
    def __init__(self):
        super(mywindow,self).__init__()
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        #self.setupUi(self)
        self.env_setup()
        #定义槽函数
    def openimage(self):
   # 打开文件路径
   #设置文件扩展名过滤,注意用双分号间隔
        imgName,imgType= QFileDialog.getOpenFileName(self,
                                    "打开图片",
                                    " *.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")

        print(imgName)
        self.image_path= imgName 
        #利用qlabel显示图片
        png = QtGui.QPixmap(imgName).scaled(self.ui.label.width(), self.ui.label.height())
        self.ui.label.setPixmap(png)
    def test(self):
        #print(self.image_path)
        result=inference(self.image_path)
        print(result)
        self.ui.label_2.setText(result)
    def env_setup(self):
        with tf.gfile.FastGFile("../output_graph_new.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

app = QtWidgets.QApplication(sys.argv)
window = mywindow()
window.show()
sys.exit(app.exec_())