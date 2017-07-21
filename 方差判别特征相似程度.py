# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 15:35:51 2017

@author: Lenovo-Y430p
"""
from  sklearn.datasets import load_iris
import numpy as np
from sklearn.feature_selection import VarianceThreshold
iris=load_iris()
#方差选择法，返回值为特征选择后的数据
#参数threshold为方差的阈值
print(VarianceThreshold(threshold=3).fit_transform(iris.data))
def variance():
    x=np.mean(iris.data)
    temp=np.multiply((iris.data-x),(iris.data-x))
    temp1=np.sqrt(np.sum(temp,axis=0))
    print(temp1)
def main():
    variance()
if __name__=='__main__':
    main()
    
    
    
    
