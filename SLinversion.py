# -*- coding: utf-8 -*-
"""
Authors: Nicholas Cox and Xavier Grundler
Description: Code for training and testing a deep neural network for an inversion problem using the Scikit-Learn package. Also produces graphs of the results.
"""

#import Scikit-Learn (Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp.2825-2830, 2011) and supporting packages
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as splitSet
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import make_scorer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler as SS
import os
import sys
import time
from eli5.sklearn import PermutationImportance

columns = ["i","xsection","incompressability","F1n","v2n","F1p","v2p"] #names for data, explained in Cox, Grundler and Li
trainData = pd.read_csv(fr"./trainingData.txt",names = columns)
testData = pd.read_csv(fr"./trainingData.txt",names = columns)

#change and make directory for output
os.chdir(f"../xavie/tamuc/drLi/nick/allOutputs/SLinv")
now = dt.datetime.now()
os.makedirs(f"./trial_{now}")
os.chdir(f"./trial_{now}")

#set print options
sys.stdout = open("out.txt","w")
np.set_printoptions(threshold=np.inf)

#this scaling removes the mean and scales data to unit variance
scaling = SS()
scaling2 = SS()

#seperate input and output parameters
xtr = trainData[["F1p","v2p"]].to_numpy()
ytr = trainData[["xsection","incompressability"]].to_numpy()
xte = testData[["F1p","v2p"]].to_numpy()
yte = testData[["xsection","incompressability"]].to_numpy()

#scale data
xtr = scaling.fit_transform(xtr)
ytr = scaling2.fit_transform(ytr)
xte = scaling.transform(xte)
yte = scaling2.transform(yte)

start = time.perf_counter() #begin counter for training model

#configure multi-layer perceptron regressor and train it
model = MLPRegressor(hidden_layer_sizes=(2,6,6,2), activation = 'tanh', solver= 'lbfgs', verbose = True, max_iter = 150, random_state = 7).fit(xtr,ytr)

end = time.perf_counter() #edn counter for training model

print("Time to train model: ",end-start,"s") #print time to train model

#find the importance and standard deviation of each input feature (column) in determining the output using eli5
perm = PermutationImportance(model, random_state = 133239, n_iter = 30, scoring = make_scorer(mse)).fit(xtr,ytr)
importantFeats = perm.feature_importances_
importantStd = perm.feature_importances_std_
print("feature_importances [F1,v2]:",importantFeats)
print("feature_importances_std [F1,v2]:",importantStd)

#test model accuracy
score = model.score(xte, yte) #R^2=1-SSE/TSS
print("score:",score)

#rescale test data for graphing
true = scaling2.inverse_transform(yte)
truex = true[:,0]
truek = true[:,1]

start = time.perf_counter() #begin counter to time how long it takes to make a prediction

pred = scaling2.inverse_transform(model.predict(xte).reshape(-1,2)) #make predictions

end = time.perf_counter() #end counter

predx = pred[:,0] #divide into X and K
predk = pred[:,1]
perfLinex = np.linspace(np.min(predx) - .05*np.min(predx), np.max(predx) + .05*np.max(predx)) #make lines of perfection
perfLinek = np.linspace(np.min(predk) - .05*np.min(predk), np.max(predk) + .05*np.max(predk))

numPred = np.size(predx)
print("Time per prediction: ",(end-start)/numPred,"s") #output normalized time per prediction

#record unscaled data
unscaledData = pd.DataFrame({"truex":truex,"predx":predx,"truek":truek,"predk":predk})
unscaledData.to_csv("unscaledData.csv")

#prediction vs. perfection for X, graph
plt.figure()
plt.plot(perfLinex,perfLinex, color = 'red')
plt.scatter(predx,truex,facecolors = 'none',edgecolors = 'black',label = F"R\u00B2: {np.round(score,3)}")
plt.xlabel(r"X (Scikit-Learn)",fontsize = 12)
plt.ylabel(r"X (IBUU simulation)",fontsize = 12)
plt.legend(loc = 'upper left',fontsize = 12)
plt.savefig(f'graphOfX.pdf',format = 'pdf')

#prediction vs. perfection for K, graph
plt.figure()
plt.plot(perfLinek,perfLinek,color = 'red')
plt.scatter(predk,truek,facecolors = 'none',edgecolors = 'black',label = F"R\u00B2: {np.round(score,3)}")
plt.xlabel(r"K (MeV) (Scikit-Learn)",fontsize = 12)
plt.ylabel(r"K (MeV) (IBUU simulation)",fontsize = 12)
plt.legend(loc = 'upper left',fontsize = 12)
plt.savefig(f'graphOfK.pdf',format = 'pdf')

#find mean of true values in test data set
meanX = np.sum(true[:,0])/numPred
meanK = np.sum(true[:,1])/numPred

#calculate inversion error for test data set
dnnError = np.zeros(numPred)
predCount = np.zeros(numPred)
XMSE = 0
KMSE = 0
for i in range(numPred):
    dnnError[i] = (pred[i,0]/meanX - true[i,0]/meanX)**2 + (pred[i,1]/meanK - true[i,1]/meanK)**2 #all values divided by the mean of the true test data because X and K have different orders of magnitude
    predCount[i] = i + 1
    XMSE += (pred[i,0] - true[i,0])**2
    KMSE += (pred[i,1] - true[i,1])**2

#calculate inversion error for test data set
mseTest = np.sum(dnnError)/numPred
XMSE /= numPred
KMSE /= numPred

print("numPred = ",numPred)
print("MSE_test = ",mseTest)
print("X MSE = ",XMSE)
print("K MSE = ",KMSE)

#plot error
plt.figure()
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.scatter(predCount,dnnError,facecolors = 'none',edgecolors = 'red',label = F"MSE: {np.format_float_scientific(mseTest,precision=2)}")
plt.xlabel(r"Scikit-Learn Prediction Number",fontsize = 12)
plt.ylabel(r"DNN Error",fontsize = 12)
plt.legend(loc = 'upper left',fontsize = 12)
plt.savefig(f'predError.pdf',format = 'pdf')

#write the configuration of the model
with open("modelConfiguration.txt","w") as modConfig:
    print(model.get_params(),file=modConfig)

sys.stdout.close()
