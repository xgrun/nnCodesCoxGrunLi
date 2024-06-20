# -*- coding: utf-8 -*-
"""
Authors: Nicholas Cox and Xavier Grundler
Description: Code for training and testing a deep neural network for an inversion problem using the Scikit-Learn package. Also produces graphs of the results.
"""

#import Scikit-Learn and supporting packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as splitSet
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import make_scorer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler as SS
from eli5.sklearn import PermutationImportance

columns = ["xsection","incompressability","v1n","v2n","v1p","v2p"] #names for data, explained in Cox, Grundler and Li
datasetDirectory = fr"./xytrain90.dat"
dataset = pd.read_csv(datasetDirectory, sep = ',', names = columns)

#this scaling removes the mean and scales data to unit variance
scaling = SS()
scaling2 = SS()

#for inversion, the inputs are v1 and v2 and the outputs are X and K
xset = dataset[["v1p","v2p"]].to_numpy()
yset = dataset[["xsection","incompressability"]].to_numpy()

#split data: 75% training, 25% testing
xtr, xte, ytr, yte = splitSet(xset, yset, test_size = .25)

#scale the training data to be between -1 and 1, then apply that same transformation to the testing data
xtr = scaling.fit_transform(xtr)
xte = scaling.transform(xte)
ytr = scaling2.fit_transform(ytr)
yte = scaling2.transform(yte)

#configure multi-layer perceptron regressor and train it
model = MLPRegressor(hidden_layer_sizes=(2,6,6,2), activation = 'tanh', solver= 'lbfgs', verbose = True, max_iter = 150).fit(xtr,ytr)

#find the importance and standard deviation of each input feature (column) in determining the output using eli5
perm = PermutationImportance(model, random_state = 133239, n_iter = 30, scoring = make_scorer(mse)).fit(xtr,ytr)
importantFeats = perm.feature_importances_
importantStd = perm.feature_importances_std_
print("feature_importances [v1,v2]:",importantFeats)
print("feature_importances_std [v1,v2]:",importantStd)

#test model accuracy
score = model.score(xte, yte) #R^2=1-SSE/TSS
print("score:",score)

#rescale test data for graphing
true = scaling2.inverse_transform(yte)
truex = true[:,0]
truek = true[:,1]

#line of perfection for graphing purposes
skpred = scaling2.inverse_transform(model.predict(xte).reshape(-1,2)) #make predictions
skpredx = skpred[:,0] #divide into X and K
skpredk = skpred[:,1]
perfLinex = np.linspace(np.min(skpredx) - .05*np.min(skpredx), np.max(skpredx) + .05*np.max(skpredx)) #make lines
perfLinek = np.linspace(np.min(skpredk) - .05*np.min(skpredk), np.max(skpredk) + .05*np.max(skpredk))

#prediction vs. perfection for X, graph
plt.figure()
plt.plot(perfLinex,perfLinex, color = 'red')
plt.scatter(skpredx,truex,facecolors = 'none',edgecolors = 'black',label = F"R\u00B2: {np.round(score,3)}")
plt.xlabel(r"Predicted X")
plt.ylabel(r"True X")
plt.legend(loc = 'upper left',fontsize = 8)
plt.savefig(f'graphOfX.pdf',format = 'pdf')

#prediction vs. perfection for K, graph
plt.figure()
plt.plot(perfLinek,perfLinek,color = 'red')
plt.scatter(skpredk,truek,facecolors = 'none',edgecolors = 'black',label = F"R\u00B2: {np.round(score,3)}")
plt.xlabel(r"Predicted K (MeV)")
plt.ylabel(r"True K (MeV)")
plt.legend(loc = 'upper left',fontsize = 8)
plt.savefig(f'graphOfK.pdf',format = 'pdf')

#write the configuration of the model
with open("modelConfiguration.txt","w") as modConfig:
    print(model.get_params(),file=modConfig)
