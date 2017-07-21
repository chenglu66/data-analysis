# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 14:22:12 2017

@author: Lenovo-Y430p
"""
from  sklearn.datasets import load_iris
from sklearn.preprocessing import Binarizer
import numpy as np
iris=load_iris()
#二值化，阈值设置为3，返回值为二值化后的数据
#print(Binarizer(threshold=10).fit_transform(iris.data))
def Binarizer1(threshold):
    m=np.shape(iris.data)[0]
    n=np.shape(iris.data)[1]
    for i in range(m):
        for j in range(n):
            if iris.data[i,j]>=threshold:
                iris.data[i,j]=1
            else:
                iris.data[i,j]=0
    return iris.data
def main():
    k=Binarizer1(3)
    print(k)
if __name__=='__main__':
    main()
    
