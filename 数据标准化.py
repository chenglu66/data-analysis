# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:59:15 2017

@author: Lenovo-Y430p
"""

from  sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import preprocessing 
import numpy as np  
iris=load_iris()
#不能保存
def bianhuan():
    X_scaled = preprocessing.scale(iris.data) 
    print(X_scaled)
    print(X_scaled.mean(axis=0))#均值   
    print(X_scaled.std(axis=0)) #方差
#能保存
def baocuan():
    scaler = StandardScaler().fit(iris.data)
    print(scaler)
    #StandardScaler(copy=True, with_mean=True, with_std=True)
    print(scaler.mean_) 
    print(scaler.std_)
    #测试将该scaler用于输入数据，变换之后得到的结果同上
    print(scaler.transform(iris.data))
    
if __name__=='__main__':
    x=np.mean(iris.data,axis=0)#axis=0是按列，axis=1是按行
    #print(x)
    c=np.multiply((iris.data-x),(iris.data-x))
    b=c/len(iris.data)
    t=sum(b,axis=0)
    t=np.sqrt(t)
    #print(t)
    #baocuan()
    print((iris.data-x)/t)#具有广播规则
#标准化源码公式：x=(x-mean(x))/std(x) 相当于标准正态分布



    