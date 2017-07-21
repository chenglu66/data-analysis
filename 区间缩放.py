# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 11:03:39 2017

@author: Lenovo-Y430p
"""
from  sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import numpy as np
#区间放缩法
iris=load_iris()
#def qujian():
    #print(MinMaxScaler().fit_transform(iris.data))

#代码实现
def minmax():
    m=np.shape(iris.data)[1]
    temp1=[];temp2=[]
    for i in range(m):
        t=np.max(iris.data[:,i])
        t1=np.min(iris.data[:,i])
        temp1.append(float('%.2f'%t))#可以在转换成float的时候，就指定精度
        temp2.append(float('%.2f'%t1))
    return np.mat(temp1),np.mat(temp2)

def main():
    #qujian()
    maxvalue,minvalue=minmax()
    offset=maxvalue-minvalue
    temp=(iris.data-minvalue)/offset
    print(temp)
    
if __name__=='__main__':
    main()