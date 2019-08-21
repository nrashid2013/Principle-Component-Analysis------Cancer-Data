# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 09:48:22 2019


"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#%matplotlib inline

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer.keys())
print(cancer['DESCR'])
df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
print(df.head())





from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)
print(scaled_data)



from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca=pca.transform(scaled_data)
print(scaled_data.shape)
print(x_pca.shape)
print(scaled_data)
print(x_pca)

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0], x_pca[:,1], c=cancer['target'])
plt.xlabel('First principle components')
plt.ylabel('Second principle compoments')
plt.show()
