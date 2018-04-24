# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 09:53:02 2018

@author: zhenyang
"""

from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt
data, target = load_boston().data, load_boston().target

datamean=np.mean(data,axis=0)
datastd=np.std(data,axis=0)
for i in range(0,len(data)):
    data[i]=(data[i]-datamean)/datastd

e_vals, e_vecs =np.linalg.eig(np.dot(data.T,data))

np.set_printoptions(threshold=np.nan)
print("eigenvalue:\n",e_vals)
print("first component:\n",e_vecs[:,0])
print("second component:\n",e_vecs[:,1])

plt.scatter(data.dot(e_vecs[:,0]), data.dot(e_vecs[:,1]), c=target / max(target))
plt.show()