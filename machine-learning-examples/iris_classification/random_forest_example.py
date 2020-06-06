#! /usr/bin/env python
#-*-coding:utf-8-*-

__author__ = 'huligong'


'''
---随机森林分类算法---

分类问题场景

鸢尾花分类：

鸢尾花数据集sklearn自带的众多经典的数据集之一

鸢尾花数据集只有150个样本，每个样本只有4个特征

数据包含三种鸢尾花的四个特征，
分别是花萼长度(cm)、花萼宽度(cm)、花瓣长度(cm)、花瓣宽度(cm)，
这些形态特征在过去被用来识别物种.

本例即是根据鸢尾花的特征进行分类识别具体是哪一种鸢尾花。

三种鸢尾花分别是:
山鸢尾花(Iris Setosa)
变色鸢尾花(Iris Versicolor)
维吉尼亚鸢尾花(Iris Virginica)

iris.csv格式样例：
     Sepal Length  Sepal Width  Petal Length  Petal Width    Species
0             5.1          3.5           1.4          0.2     setosa
1             4.9          3.0           1.4          0.2     setosa
2             4.7          3.2           1.3          0.2     setosa
3             4.6          3.1           1.5          0.2     setosa
...

'''


import sys
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn import model_selection
from sklearn.datasets import load_iris

import pandas as pd
import numpy as np

#iris = load_iris() #加载鸢尾花数据集
#print(iris.data.shape) #(150, 4) # 150个样本，每个样本4个特征


#本地样本数据 分成训练集、测试集（占0.24）
def input_data():
	#本地样本数据目录
	sample_data_dir =  os.path.join(os.path.dirname(os.getcwd()),'sample_data')	
	data_file = os.path.join(sample_data_dir,'iris','iris.csv')
	iris = pd.read_csv(data_file)
	print(iris)
	iris_target = iris["Species"] #目标变量
	iris_data = iris.iloc[:,1:4] #自变量
	train_data,test_data,train_target,test_target = model_selection.train_test_split(iris_data,
		iris_target,test_size=0.24,random_state=0)
	return train_data,test_data,train_target,test_target

#构造训练集，120/150
def input_data_trains():	
	# 训练集
	train_data = np.concatenate((iris.data[0:40, :], iris.data[50:90, :], iris.data[100:140, :]), axis = 0)
	# 训练集样本类别
	train_target = np.concatenate((iris.target[0:40], iris.target[50:90], iris.target[100:140]), axis = 0)
	return train_data,train_target

#构造测试集，30/150
def input_data_tests():
	# 测试集
	test_data = np.concatenate((iris.data[40:50, :], iris.data[90:100, :], iris.data[140:150, :]), axis = 0)
	#测试集样本类别
	test_target = np.concatenate((iris.target[40:50], iris.target[90:100], iris.target[140:150]), axis = 0)
	return test_data,test_target

#训练分类器
def train_clf(train_data, train_tags):
    from sklearn.ensemble import RandomForestClassifier	
    clf = RandomForestClassifier(n_jobs=2)
    clf.fit(train_data, np.asarray(train_tags))
    return clf


#评价分类器
def eval_state_model(actual,pred):
    m_accuracy_score = accuracy_score(actual, pred)
    m_precision = precision_score(actual,pred, average="macro") #准确率
    m_recall = recall_score(actual,pred, average="macro") #召回率
    m_f1_score = f1_score(actual, pred, average="macro")

    print('predict info:')
    print('precision:{0:.4f}'.format(m_precision))
    print('recall:{0:0.4f}'.format(m_recall))
    print('f1-score:{0:.4f}'.format(m_f1_score)) 
    print('accuracy_score:',round(m_accuracy_score,4))


#可视化决策树
#http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
def export_graphviz(clf):	
	from sklearn.tree import export_graphviz
	from sklearn import tree
	#from StringIO import StringIO
	#out = StringIO()
	#out = tree.export_graphviz(clf, out_file=out)
	out = tree.export_graphviz(clf,out_file='iris_tree.dot')
	
	#$ sudo apt-get install graphviz 
	#$ dot -Tpng iris_tree.dot -o iris_tree.png  # 生成png图片
	#$ dot -Tpdf iris_tree.dot -o iris_tree.pdf  # 生成pdf

#分类
def execute_clf():
	train_data,test_data,train_target,test_target = input_data()

	#train_data,train_target = input_data_trains()
	#test_data,test_target = input_data_tests()

	clf = train_clf(train_data, train_target)

	pred = clf.predict(test_data) #预测

	eval_state_model(test_target,pred) #评价预测效果

	print('test data count:%d'%(test_data.shape[0])) #测试样本数量
	print('pred success count:%d'%(sum(test_target==pred))) #预测成功数量

	#export_graphviz(clf)

def main():
    execute_clf()

if __name__ == "__main__" : 
    main()