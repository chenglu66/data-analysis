# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 16:07:34 2017

@author: Lenovo-Y430p
"""
from sklearn.feature_selection import SelectKBest
from minepy import MINE
from  sklearn.datasets import load_iris
iris=load_iris()
 #由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5
def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return (m.mic(), 0.5)

#选择K个最好的特征，返回特征选择后的数据
SelectKBest(lambda X, Y: array(map(lambda x:mic(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)
