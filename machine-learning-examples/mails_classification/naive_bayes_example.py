#! /usr/bin/env python
#-*-coding:utf-8-*-

__author__ = 'huligong'

'''
---朴素贝叶斯分类算法---

分类问题场景

邮件分类(垃圾邮件识别)：

现有已分类好垃圾邮件(spam)和正常邮件(ham)的的训练集和测试集：
./sample_data/mails/train/ham/0001.1999-12-10.farmer.ham.txt
./sample_data/mails/train/ham/0001.1999-12-10.kaminski.ham.txt
...

./sample_data/mails/train/spam/0002.2001-05-25.SA_and_HP.spam.txt
./sample_data/mails/train/spam/0004.2004-08-01.BG.spam.txt
...

./sample_data/mails/test/ham/0001.2001-02-07.kitchen.ham.txt
./sample_data/mails/test/ham/0002.2001-02-07.kitchen.ham.txt
...

./sample_data/mails/test/spam/0011.2001-06-28.SA_and_HP.spam.txt
./sample_data/mails/test/spam/0016.2001-07-05.SA_and_HP.spam.txt
...


本例目标：
通过训练集数据训练分类器模型，测试邮件分类的效果

'''

import sys
import os
import string
import re
import codecs
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer


#样本数据目录
sample_data_dir =  os.path.join(os.path.dirname(os.getcwd()),'sample_data')

tfidf_vectorizer = TfidfVectorizer(binary = False, decode_error = 'ignore', stop_words = 'english')

#返回文本和标签
def input_train_data(ham_train_dir,spam_train_dir):
    train_words = []
    train_tags = []

    print('load ham train data...')
    ham_train_filenames = os.listdir(ham_train_dir) 
    for fname in ham_train_filenames:
        fpath = os.path.join(ham_train_dir,fname)
        if os.path.isfile(fpath):
            with open(fpath, 'r') as f1:
                fcontent = f1.read()
                train_words.append(fcontent)
                train_tags.append('ham')

    print('load spam train data...')
    spam_train_filenames = os.listdir(spam_train_dir) 
    for fname in spam_train_filenames:
        fpath = os.path.join(spam_train_dir,fname)
        if os.path.isfile(fpath):
            with open(fpath, 'r',encoding='iso-8859-1') as f1:
                fcontent = f1.read()
                train_words.append(fcontent)
                train_tags.append('spam')
    return train_words, train_tags

def input_test_data(ham_test_dir,spam_test_dir):
    ham_test_words = []
    ham_test_tags = []

    test_spam_words = []
    test_spam_tags = []

    print('load ham test data...')
    ham_test_filenames = os.listdir(ham_test_dir) 
    for fname in ham_test_filenames:
        fpath = os.path.join(ham_test_dir,fname)
        if os.path.isfile(fpath):
            with open(fpath, 'r') as f1:
                fcontent = f1.read()
                ham_test_words.append(fcontent)
                ham_test_tags.append('ham')


    print('load spam test data...')
    spam_test_filenames = os.listdir(spam_test_dir) 
    for fname in spam_test_filenames:
        fpath = os.path.join(spam_test_dir,fname)
        if os.path.isfile(fpath):
            with open(fpath, 'r',encoding="iso-8859-1") as f1:
                fcontent = f1.read()
                test_spam_words.append(fcontent)
                test_spam_tags.append('spam')    
    return ham_test_words,ham_test_tags,test_spam_words,test_spam_tags

def vectorize_train(train_words):
    train_data = tfidf_vectorizer.fit_transform(train_words)
    return train_data

def vectorize_test(test_words):
    test_data = tfidf_vectorizer.transform(test_words)
    return test_data


#训练分类器
def train_clf(train_data, train_tags):
    from sklearn.naive_bayes import MultinomialNB	
    clf = MultinomialNB(alpha=0.01)
    clf.fit(train_data, np.asarray(train_tags))
    return clf

#评价分类器
def eval_state_model(actual,pred):
    pass

#邮件分类
def mails_clf():
    ham_train_dir = os.path.join(sample_data_dir,'mails','train','ham')
    spam_train_dir = os.path.join(sample_data_dir,'mails','train','spam')

    ham_test_dir = os.path.join(sample_data_dir,'mails','test','ham')
    spam_test_dir = os.path.join(sample_data_dir,'mails','test','spam')

    train_words,train_tags = input_train_data(ham_train_dir,spam_train_dir)

    ham_test_words,ham_test_tags,spam_test_words,spam_test_tags = input_test_data(ham_test_dir,spam_test_dir)

    print('start train data ...')
    train_data = vectorize_train(train_words)
    clf = train_clf(train_data, train_tags)

    ham_test_data = vectorize_test(ham_test_words)      
    spam_test_data = vectorize_test(spam_test_words)

    print('start ham predict data...')
    ham_pred = clf.predict(ham_test_data) #预测 正常邮件识别
    print('ham test data count:%d'%(ham_test_data.shape[0])) #测试样本数量
    print('ham pred success count:%d'%(sum(ham_test_tags==ham_pred))) #预测成功数量 

    print('start spam predict data...')
    spam_pred = clf.predict(spam_test_data) #预测 垃圾邮件识别
    
    print('spam test data count:%d'%(spam_test_data.shape[0])) #测试样本数量
    print('spam pred success count:%d'%(sum(spam_test_tags==spam_pred))) #预测成功数量                


def main():
    mails_clf()

if __name__ == "__main__" : 
    main()