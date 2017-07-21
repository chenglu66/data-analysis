# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 11:33:41 2017

@author: Lenovo-Y430p
"""
#归一化处理
from  sklearn.datasets import load_iris
from sklearn.preprocessing import Normalizer
import numpy as np
#归一化，返回值为归一化后的数据
iris=load_iris()
print(Normalizer().fit_transform(iris.data))
#归一化就是把数据化成单位向量，这是按行进行处理，这样在用核函数计算相似性时具有统一的标准,使用的是L2归一化公式
temp1=[]
def guiyi():
    for i in range(np.shape(iris.data)[0]):
        temp=np.sqrt(sum(np.multiply(iris.data[i,:],iris.data[i,:])))
        print(temp)
        iris.data[i,:]/=temp
    print(iris.data)
        
def main():
    guiyi()
    
    
if __name__=='__main__':
    main()
    

      
        
    





