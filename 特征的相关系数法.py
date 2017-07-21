# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 15:49:15 2017

@author: Lenovo-Y430p
"""
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from  sklearn.datasets import load_iris
from numpy import array
iris=load_iris()
#选择K个最好的特征，返回选择特征后的数据
#第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
#参数k为选择的特征个数
print(SelectKBest(lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target))



