# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 16:03:18 2017

@author: Lenovo-Y430p
"""
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from  sklearn.datasets import load_iris
iris=load_iris()
#选择K个最好的特征，返回选择特征后的数据
'''
卡方检验就是统计样本的实际观测值与理论推断值之间的偏离程度，
实际观测值与理论推断值之间的偏离程度就决定卡方值的大小，
卡方值越大，越不符合；卡方值越小，偏差越小，越趋于符合，
若两个值完全相等时，卡方值就为0，表明理论值完全符合。
检验样本是否符合正太分布
'''
#文本分析，和筛选异常用户
print(SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target))

