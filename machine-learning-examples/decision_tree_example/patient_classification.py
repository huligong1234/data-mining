#! /usr/bin/env python
#-*-coding:utf-8-*-

__author__ = 'huligong'

'''
---决策树分类算法---

分类问题场景

病人分类：
某个医院早上收了六个门诊病人，如下表。
　　症状　　职业　　　疾病
　　打喷嚏　护士　　　感冒 
　　打喷嚏　农夫　　　过敏 
　　头痛　　建筑工人　脑震荡 
　　头痛　　建筑工人　感冒 
　　打喷嚏　教师　　　感冒 
　　头痛　　教师　　　脑震荡
现在又来了第七个病人，是一个打喷嚏的建筑工人。请问他患上感冒的概率有多大？

问题来源：http://www.ruanyifeng.com/blog/2013/12/naive_bayes_classifier.html


症状和职业即特征(features)，疾病即标签(lables)


特征预处理：

症状和职业都是非连续特征。非连续特征能够比较方便地进行因子化，或者它本身就是二元特征。
方法如下：

特征“症状”：{打喷嚏，头痛} 因子化后的结果为：
特征“是否打喷嚏”：{是，否}
特征“是否头痛”：{是，否}

特征“职业”：{护士，农夫，建筑工人，教师} 因子化后的结果为：
特征“是否护士”：{是，否}
特征“是否农夫”：{是，否}
特征“是否建筑工人”：{是，否}
特征“是否教师”：{是，否}


'''

import numpy as np

	
#0：感冒；1：过敏；2：脑震荡
target_names = ["cold", "irritability", "concussion"] 


#训练分类器
def train_clf(train_data, train_tags):
	from sklearn.tree import DecisionTreeClassifier	
	clf = DecisionTreeClassifier()
	clf.fit(train_data, np.asarray(train_tags))
	return clf

def main():

	'''
	[1,0,1,0,0,0]表示的就是第1条病例的特征(症状和职业):
	[是打喷嚏,否感冒,是护士,否农夫,否建筑工人,否教师]
	'''
	x_features = np.array([
		[1,0,1,0,0,0], #第1条病例特征
		[1,0,0,1,0,0], #第2条病例特征
		[0,1,0,0,1,0], #第3条病例特征
		[0,1,0,0,1,0], #第4条病例特征
		[1,0,0,0,0,1], #第5条病例特征
		[0,1,0,0,0,1]  #第6条病例特征
		])

	#labels = ["感冒", "过敏", "脑震荡", "感冒", "感冒", "脑震荡"]
	y_labels = np.array([0,1,2,0,0,2]) 

	clf = train_clf(x_features,y_labels)

	#打喷嚏的建筑工人 [是打喷嚏,否感冒,否护士,否农夫,是建筑工人,否教师]
	x_test = [[1,0,0,0,1,0]]

	y_pred =  clf.predict(x_test) #预测最有可能是哪种疾病

	print('predict result:')
	print(target_names[y_pred[0]])
	pred_proba = clf.predict_proba(x_test) #预测各种疾病的概率
	print(zip(target_names, pred_proba[0]))

if __name__ == "__main__" : 
    main()