# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 18:23:35 2017

@author: Lenovo-Y430p
"""
#1.2 互信息和最大信息系数 (Mutual information and maximal information)，[0,1]
#互信息直接用于特征选择不太方便，最大信息系数首先寻找一种最优的离散化方式，
#然后把互信息取值转换成一种度量方式，取值区间在[0，1]。minepy提供了MIC功能。
import numpy as np
from minepy import MINE  #
m = MINE()
x = np.random.uniform(-1, 1, 10000)
m.compute_score(x, x**2)
print (m.mic())
