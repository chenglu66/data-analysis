# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 19:41:18 2017

@author: Lenovo-Y430p
"""
from sklearn.datasets import load_boston  
from sklearn.linear_model import (LinearRegression, Ridge,   
                                  Lasso, RandomizedLasso)  
from sklearn.feature_selection import RFE, f_regression  
from sklearn.preprocessing import MinMaxScaler  
from sklearn.ensemble import RandomForestRegressor  
import numpy as np  
from minepy import MINE  
  
np.random.seed(0)  
  
size = 750  
X = np.random.uniform(0, 1, (size, 14))  
#x0到x4对输出来说是有用的 
#"Friedamn #1” regression problem  
Y = (10 * np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2] - .5)**2 +  
     10*X[:,3] + 5*X[:,4] + np.random.normal(0,1))  
#Add 3 additional correlated variables (correlated with X1-X3) 
#x10到x14是x0到x4的变种
X[:,9:] = X[:,:5] + np.random.normal(0, .025, (size,5))  
#其余的都是噪声
names = ["x%s" % i for i in range(0,14)]  
ranks = {}  
  
def rank_to_dict(ranks, names, order=1):  
    minmax = MinMaxScaler()  
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]  
    ranks = list(map(lambda x: round(x, 2), ranks))  
    return dict(zip(names, ranks ))  
  
lr = LinearRegression(normalize=True)  
lr.fit(X, Y)  
ranks["reg"] = rank_to_dict(np.abs(lr.coef_), names)  
  
ridge = Ridge(alpha=7)  
ridge.fit(X, Y)  
ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), names)  
  
  
lasso = Lasso(alpha=0.05)  
lasso.fit(X, Y)  
ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_), names)  
  
  
rlasso = RandomizedLasso(alpha=0.04)  
rlasso.fit(X, Y)  
ranks["Stability"] = rank_to_dict(np.abs(rlasso.scores_), names)  
  
#stop the search when 5 features are left (they will get equal scores)  
rfe = RFE(lr, n_features_to_select=5)  
rfe.fit(X,Y)  
ranks["RFE"] = rank_to_dict(list(map(float, rfe.ranking_)), names, order=-1)  
  
rf = RandomForestRegressor()  
rf.fit(X,Y)  
ranks["RF"] = rank_to_dict(rf.feature_importances_, names)  
  
  
f, pval  = f_regression(X, Y, center=True)  
ranks["Corr."] = rank_to_dict(f, names)  
''' 
mine = MINE()  
mic_scores = []  
for i in range(X.shape[1]):  
    mine.compute_score(X[:,i], Y)  
    m = mine.mic()  
    mic_scores.append(m)  
  
ranks["MIC"] = rank_to_dict(mic_scores, names)  
  
'''  
r = {}  
for name in names:  
    r[name] = round(np.mean([ranks[method][name]   
                             for method in ranks.keys()]), 2)  
  
methods = sorted(ranks.keys())  
ranks["Mean"] = r  
methods.append("Mean")  
  
print ("\t%s" % "\t".join(methods))  
for name in names:  
    print ("%s\t%s" % (name, "\t".join(map(str,   
                         [ranks[method][name] for method in methods]))))  
