#! /usr/bin/env python
#-*-coding:utf-8-*-

__author__ = 'huligong'

'''
---朴素贝叶斯分类算法---

分类问题场景

新闻分类：

现有已分类好的的训练集和测试集train_file.txt和test_file.txt，格式如下：
health	夏末秋初红眼病高发 白领群体患病居高不下
joke	损人真是件爆笑又过瘾滴事
digi	北京公交WiFi网络实测" 免费午餐不好拿"
joke	这是何必呢
constellation	揭秘12星座喜欢人的方式大不同
movie	丸山隆平确定出演《圆桌》 扮演芦田爱菜班主任
star	调查：朱莉为防癌切除乳腺你如何看待？
science	大蒜，你为什么那么受欢迎？
photo	只有美国人不爱无反相机？
photo	300幅摄影作品诠释人文澄迈(图)
pet	的哥养蝈蝈陪驾来解闷
photo	瑞典摄影师获第56届“荷赛”年度奖
...

本例目标：
通过训练集数据训练分类器模型，测试分类的效果

'''

import sys
import os
import string
import re
import codecs
import jieba
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.feature_extraction.text import HashingVectorizer

#Python27解决中文乱码问题
reload(sys)
sys.setdefaultencoding('utf8')

#样本数据目录
sample_data_dir =  os.path.join(os.path.dirname(os.getcwd()),'sample_data')

#使用结巴分词器进行分词
comma_tokenizer = lambda x: jieba.cut(x)
hashing_vectorizer = HashingVectorizer(tokenizer=comma_tokenizer, n_features=30000, non_negative=True)


#返回文本和标签
def input_data(train_file):
    train_words = []
    train_tags = []
    with open(train_file, 'r') as f1:
        for line in f1:
            tks = line.split('\t', 1)
            train_tags.append(tks[0])
            train_words.append(tks[1])
    return train_words, train_tags

def vectorize_train(train_words):
    train_data = hashing_vectorizer.fit_transform(train_words)
    return train_data

def vectorize_test(train_words):
    test_data = hashing_vectorizer.transform(test_words)
    return test_data   

#训练分类器
def train_clf(train_data, train_tags):
    from sklearn.naive_bayes import MultinomialNB	
    clf = MultinomialNB(alpha=0.01)
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

#新闻分类
def text_clf():
    train_file = os.path.join(sample_data_dir,'news','train_file.txt')
    test_file = os.path.join(sample_data_dir,'news','test_file.txt')
    
    train_words, train_tags = input_data(train_file)

    train_data = vectorize_train(train_words)
    clf = train_clf(train_data, train_tags)

    test_words, test_tags = input_data(train_file)
    test_data = vectorize_test(test_words)

    pred = clf.predict(test_data) #预测

    eval_state_model(test_tags,pred) #评价预测效果
    
    print('test data count:%d'%(test_data.shape[0])) #测试样本数量
    print('pred success count:%d'%(sum(test_tags==pred))) #预测成功数量

def main():
    text_clf()

if __name__ == "__main__" : 
    main()