#! /usr/bin/env python
#-*-coding:utf-8-*-

__author__ = 'huligong'


'''
---线性回归预测算法---

回归问题场景

房价预测(简单)：

依据房屋面积，预测房子的价格

示例数据集如下：
   id  square_feet  price
0   1          150   6450
1   2          200   7450
2   3          250   8450
3   4          300   9450
4   5          350  11450
5   6          400  15450
6   7          600  18450

其中id为编号，square_feet为平方英尺，即房子大小，price为房子价格

假设房子价格与房屋面积存在一种线性关系，假设方程式为 f(x) = a + bx
f(x)是关于房屋面积的价格值,即我们要预测的值,a是一个常数,b是回归系数


本例参考：http://python.jobbole.com/81215/ 在Python中使用线性回归预测数据

'''

import sys
import os

import numpy as np
from sklearn import datasets, linear_model

import pandas as pd
import numpy as np

#Python27解决中文乱码问题
reload(sys)
sys.setdefaultencoding('utf8')


def input_data():
	#本地样本数据目录
	sample_data_dir =  os.path.join(os.path.dirname(os.getcwd()),'sample_data')	
	data_file = os.path.join(sample_data_dir,'house_price','simple_house_price.csv')	
	data = pd.read_csv(data_file)
	print data
	X_parameter = []
	Y_parameter = []
	for single_square_feet ,single_price_value in zip(data['square_feet'],data['price']):
	   X_parameter.append([float(single_square_feet)])
	   Y_parameter.append(float(single_price_value))
	return X_parameter,Y_parameter

# 训练回归模型
def train_model(X_parameters,Y_parameters):
	regr = linear_model.LinearRegression()
	# 把X_parameter和Y_parameter拟合为线性回归模型
	regr.fit(X_parameters, Y_parameters)
	return regr


# 显示数据拟合的直线
def show_linear_line(regr,X_parameters,Y_parameters):
	import matplotlib.pyplot as plt
	from pylab import mpl

	fig, ax = plt.subplots()
	mpl.rcParams['font.sans-serif'] = ['FangSong'] #指定默认字体,解决中文乱码问题
	mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
	ax.scatter(X_parameters,Y_parameters,color='blue')
	ax.plot(X_parameters,regr.predict(X_parameters),color='red',linewidth=4)
	plt.xticks(())
	plt.yticks(())
	ax.set_xlabel('房屋面积')
	ax.set_ylabel('房价')
	plt.show()

#预测
def execute_predict():
	X,Y = input_data()
	predict_value = 700 # 房屋面积,就是f(x)=a+bx线性函数中x的值
	regr = train_model(X,Y)
	pred = regr.predict(predict_value)

	print "Intercept value " , regr.intercept_ #（截距值）就是f(x)=a+bx线性函数中a的值
	print "coefficient" , regr.coef_ #（系数）就是f(x)=a+bx线性函数中b的值
	print "Predicted value: ",pred #预测的房子价格
	
	show_linear_line(regr,X,Y)


def main():
    execute_predict()

if __name__ == "__main__" : 
    main()