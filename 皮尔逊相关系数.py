# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 18:08:48 2017

@author: Lenovo-Y430p
"""
#皮尔逊相关系数(来处理线性关系)
#1.1 Pearson相关系数 （Pearson Correlation) [-1,1]
#理解特征和响应变量之间关系的方法，该方法衡量的是变量之间的线性相关性，可以理解为两个向量之间的夹角
#之所以能度量相似程度就是因为时距离均值的大小，同正同负表征了变化的方向和大小。但因为离散程度不同
#需要加入对标准差的除法
'''
统计学上规定的P值意义见下表
  P值 碰巧的概率 对无效假设 统计意义 
P＞0.05 碰巧出现的可能性大于5% 不能否定无效假设 两组差别无显著意义 
P＜0.05 碰巧出现的可能性小于5% 可以否定无效假设 两组差别有显著意义 
P ＜0.01 碰巧出现的可能性小于1% 可以否定无效假设 两者差别有非常显著意义 
'''
import numpy as np
from scipy.stats import pearsonr  #从scipy中引入pearsonr
from  sklearn.datasets import load_iris
from numpy import *
iris=load_iris()
m=shape(iris.data)[1]
for i in range(m):
    print ("i", pearsonr(iris.data[:,i], iris.target))
#print ("Higher noise", pearsonr(x, x + np.random.normal(0, 10, size)))
#明显缺陷:作为特征排序机制，他只对线性关系敏感.即便两个变量具有一一对应的关系，Pearson相关性也可能会接近0
a = np.random.uniform(-1, 1, 100000)   #uniform(low,high,size) 随机数
print (pearsonr(a, a**2))
#返回两个值评分和p值，p值越大越坏
def pearson(x,y):
    xmean=mean(x,axis=0)
    ymean=mean(y,axis=0)
    covxy=dot((x-xmean).T,y-ymean)
    xvar=dot((x-xmean).T,x-xmean)
    yvar=dot((y-ymean).T,y-ymean)
    p=covxy/sqrt(xvar*yvar)
    print(p)
def main():
    iris=load_iris()
    m=shape(iris.data)[1]
    for i in range(m):
        pearson(iris.data[:,i], iris.target)
if __name__=='__main__':
    main()
    