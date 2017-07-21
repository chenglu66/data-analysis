# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 14:41:37 2017

@author: Lenovo-Y430p
"""
from numpy import vstack, array, nan
from sklearn.preprocessing import Imputer
#缺失值计算，返回值为计算缺失值后的数据
#参数missing_value为缺失值的表示形式，默认为NaN
#参数strategy为缺失值填充方式，默认为mean（均值）
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X=[[1, 2], [nan, 3], [7, 6]]
Y=[[nan, 2], [6, nan], [7, 6]]
print(imp.fit(X))
print(imp.transform(X))
print(imp.transform(Y))