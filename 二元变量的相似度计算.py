# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 20:48:21 2017

@author: Lenovo-Y430p
"""
#-*-coding:utf-8 -*-
# 计算jaccard系数
'''
当数据集为二元变量时，我们只有两种状态：0或者1。
这个时候以上的计算相似度的方法就无法派上用场，
于是我们引出Jaccard系数，
这是一个能够表示两个数据集都是二元变量（也可以多元）的相似度的指标，其公式为
'''
def jaccard(p,q):
    c = [a for i in p if v in b]
    return float(len(c))/(len(a)+len(b)-len(b))
#注意：在使用之前必须对两个数据集进行去重
#我们用一些特殊的数据集去测试一下：

p = ['shirt','shoes','pants','socks']
q = ['shirt','shoes']
print jaccard(p,q)
得出结果是:0.5
