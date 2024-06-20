# -*- coding: utf-8 -*-
"""
Authors: Nicholas Cox and Xavier Grundler
Description: Code for training and testing a deep neural networkas an emulator using the Scikit-Learn package. Also produces graphs of the results.
"""

#import Scikit-Learn and supporting packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as splitSet
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import make_scorer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MaxAbsScaler as MAS
from eli5.sklearn import PermutationImportance

columns = ["xsection","incompressability","v1n","v2n","v1p","v2p"] #names for data, explained in Cox, Grundler and Li
datasetDirectory = fr"./xytrain90.dat"
dataset = pd.read_csv(datasetDirectory, sep = ',', names = columns)

#this scaling divides each column in a numpy array by its largest value, designed to maintain sparsity in datasets during preprocessing
scaling = MAS()
scaling2 = MAS()

#for emulation, the inputs are X and K and the outputs are v1 and v2
xset = dataset[["xsection","incompressability"]].to_numpy()
yset = dataset[["v1p","v2p"]].to_numpy()

#scale the training data to be between -1 and 1, then apply that same transformation to the testing data
xset = scaling.fit_transform(xset)
yset = scaling2.fit_transform(yset)

#split data: 75% training, 25% testing
xtr, xte, ytr, yte = splitSet(xset, yset, test_size = .25)

#configure multi-layer perceptron regressor and train it
model = MLPRegressor(hidden_layer_sizes=(2,6,6,2), activation = 'tanh', solver= 'lbfgs', verbose = True, max_iter = 80).fit(xtr,ytr)

#find the importance and standard deviation of each input feature (column) in determining the output using eli5
perm = PermutationImportance(model, random_state = 133239, n_iter = 30, scoring = make_scorer(mse)).fit(xtr,ytr)
importantFeats = perm.feature_importances_
importantStd = perm.feature_importances_std_
print("feature_importances [X,K]:",importantFeats)
print("feature_importances_std: [X,K]",importantStd)

#test model accuracy
score = model.score(xte, yte) #R^2=1-SSE/TSS
print("score:",score)

#rescale test data for graphing
truev = scaling2.inverse_transform(yte)
truev1 = truev[:,0]
truev2 = truev[:,1]

#line of perfection for graphing purposes
pred = scaling2.inverse_transform(model.predict(xte).reshape(-1,2)) #make predictions
predv1 = pred[:,0] #divide into v1 and v2
predv2 = pred[:,1]
perfLinev1 = np.linspace(np.min(predv1) - .05*np.min(predv1), np.max(predv1) + .05*np.max(predv1)) #make lines
perfLinev2 = np.linspace(np.min(predv2) - .05*np.min(predv2), np.max(predv2) + .05*np.max(predv2))

#prediction vs. perfection for v1, graph
plt.figure()
plt.plot(perfLinev1,perfLinev1,color = 'red')
plt.scatter(predv1,truev1,facecolors = 'none',edgecolors = 'black',label = F"R\u00B2: {np.round(score,3)}")
plt.xlabel(r"Predicted $v_1$")
plt.ylabel(r"True $v_1$")
plt.legend(loc = 'upper left',fontsize = 8)
plt.savefig(f'graphOfv1.pdf',format = 'pdf')

#prediction vs. perfection for v2, graph
plt.figure()
plt.plot(perfLinev2,perfLinev2,color = 'red')
plt.scatter(predv2,truev2,facecolors = 'none',edgecolors = 'black',label = F"R\u00B2: {np.round(score,3)}")
plt.xlabel(r"Predicted $v_2$")
plt.ylabel(r"True $v_2$")
plt.legend(loc = 'upper left',fontsize = 8)
plt.savefig(f'graphOfv2.pdf',format = 'pdf')

#calculate emulator error for entire data set
yset = scaling2.inverse_transform(yset)
fullPred = scaling2.inverse_transform(model.predict(xset).reshape(-1,2))
dnnError = np.zeros(89)
runNum = np.zeros(89)
for i in range(89):
    dnnError[i] = (fullPred[i,0] - yset[i,0])**2 + (fullPred[i,1] - yset[i,1])**2
    runNum[i] = i + 1

#plot error
plt.figure()
plt.scatter(runNum,dnnError,facecolors = 'none',edgecolors = 'red')
plt.xlabel(r"Run Number")
plt.ylabel(r"DNN Error")
plt.savefig(f'predError.pdf',format = 'pdf')

#write the configuration of the model
with open("modelConfiguration.txt","w") as modConfig:
    print(model.get_params(),file=modConfig)
