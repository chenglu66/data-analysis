# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 15:27:50 2017

@author: Lenovo-Y430p
"""
from numpy import vstack, array, nan
from sklearn.preprocessing import Imputer
from  sklearn.datasets import load_iris
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
#多项式转换
#参数degree为度，默认值为2
iris=load_iris()
print(PolynomialFeatures().fit_transform(iris.data))
from numpy import log1p
from sklearn.preprocessing import FunctionTransformer
#自定义转换函数为对数函数的数据变换
#第一个参数是单变元函数
print(FunctionTransformer(log1p).fit_transform(iris.data))

