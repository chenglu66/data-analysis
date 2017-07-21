# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:58:19 2017

@author: Lenovo-Y430p
"""
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from  sklearn.datasets import load_iris
iris=load_iris()
#GBDT作为基模型的特征选择
print(SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data, iris.target))
