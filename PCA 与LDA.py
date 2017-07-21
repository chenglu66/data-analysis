# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 16:13:57 2017

@author: Lenovo-Y430p
"""
from sklearn.decomposition import PCA
from  sklearn.datasets import load_iris
iris=load_iris()
#主成分分析法，返回降维后的数据
#参数n_components为主成分数目
print(PCA(n_components=2).fit_transform(iris.data))
from sklearn.lda import LDA
#线性判别分析法，返回降维后的数据
#参数n_components为降维后的维数
print(LDA(n_components=2).fit_transform(iris.data, iris.target))
