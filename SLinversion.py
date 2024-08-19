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

columns = ["xsection","incompressability","F1n","v2n","F1p","v2p"] #names for data, explained in Cox, Grundler and Li
datasetDirectory = fr"./xytrain90.dat"
dataset = pd.read_csv(datasetDirectory, sep = ',', names = columns)

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

#for inversion, the inputs are F1 and v2 and the outputs are X and K
xset = dataset[["F1p","v2p"]].to_numpy()
yset = dataset[["xsection","incompressability"]].to_numpy()

#scale data
xset = scaling.fit_transform(xset)
yset = scaling2.fit_transform(yset)

#split data: 75% training, 25% testing
xtr, xte, ytr, yte = splitSet(xset, yset, test_size = .25)

#record training and testing data
trainDF = pd.DataFrame({"xtrX":xtr[:,0],"ytrX":ytr[:,0],"xtrK":xtr[:,1],"ytrK":ytr[:,1]})
trainDF.to_csv("train.csv")
testDF = pd.DataFrame({"xteX":xte[:,0],"yteX":yte[:,0],"xteK":xte[:,1],"yteK":yte[:,1]})
testDF.to_csv("test.csv")

start = time.perf_counter() #begin counter for training model

#configure multi-layer perceptron regressor and train it
model = MLPRegressor(hidden_layer_sizes=(2,6,6,2), activation = 'tanh', solver= 'lbfgs', verbose = True, max_iter = 150).fit(xtr,ytr)

end = time.perf_counter() #edn counter for training model

print("Time to train model: ",end-start,"s") #print time to train model

#test model accuracy
score = model.score(xte, yte) #R^2=1-SSE/TSS
print("score:",score)

#rescale test data for graphing
true = scaling2.inverse_transform(yte)
trueX = true[:,0]
trueK = true[:,1]

start = time.perf_counter() #begin counter to time how long it takes to make a prediction

pred = scaling2.inverse_transform(model.predict(xte).reshape(-1,2)) #make predictions

end = time.perf_counter() #end counter

predX = pred[:,0] #divide into X and K
predK = pred[:,1]
perfLinex = np.linspace(np.min(predX) - .05*np.min(predX), np.max(predX) + .05*np.max(predX)) #make lines of perfection
perfLinek = np.linspace(np.min(predK) - .05*np.min(predK), np.max(predK) + .05*np.max(predK))

numPred = np.size(predX)
print("Time per prediction: ",(end-start)/numPred,"s") #output normalized time per prediction

#record unscaled data
unscaledData = pd.DataFrame({"trueX":trueX,"predX":predX,"trueK":trueK,"predK":predK})
unscaledData.to_csv("unscaledData.csv")

#prediction vs. perfection for X, graph
plt.figure()
plt.plot(perfLinex,perfLinex, color = 'red')
plt.scatter(predX,trueX,facecolors = 'none',edgecolors = 'black',label = F"R\u00B2: {np.round(score,3)}")
plt.xlabel(r"X (Scikit-Learn)",fontsize = 12)
plt.ylabel(r"X (IBUU simulation)",fontsize = 12)
plt.legend(loc = 'upper left',fontsize = 12)
plt.savefig(f'graphOfX.pdf',format = 'pdf')

#prediction vs. perfection for K, graph
plt.figure()
plt.plot(perfLinek,perfLinek,color = 'red')
plt.scatter(predK,trueK,facecolors = 'none',edgecolors = 'black',label = F"R\u00B2: {np.round(score,3)}")
plt.xlabel(r"K (MeV) (Scikit-Learn)",fontsize = 12)
plt.ylabel(r"K (MeV) (IBUU simulation)",fontsize = 12)
plt.legend(loc = 'upper left',fontsize = 12)
plt.savefig(f'graphOfK.pdf',format = 'pdf')

#calculate variance of DNN and IBUU predictions
varTrueX = np.var(trueX,ddof=1)
varPredX = np.var(predX,ddof=1)
varTrueK = np.var(trueK,ddof=1)
varPredK = np.var(predK,ddof=1)

#calculate sum ov variances
sumVarX = varTrueX + varPredX
sumVarK = varTrueK + varPredK

#initialize arrays for recording DNN error
dnnErrorX = np.zeros(numPred)
dnnErrorK = np.zeros(numPred)
predCount = np.zeros(numPred) #for plotting DNN error as function of run number

#calculate DNN error for each prediction and sum of squared errors
for i in range(numPred):
    dnnErrorX[i] = ((predX[i] - trueX[i])**2)/sumVarX
    dnnErrorK[i] = ((predK[i] - trueK[i])**2)/sumVarK
    predCount[i] = i + 1

print("numPred = ",numPred)

#plot DNN error
fig, (axX, axK) = plt.subplots(2,sharex=True) #two subplots with shared x axis

axX.scatter(predCount,dnnErrorX,facecolors='none',edgecolors='red') #make scatter plot for X
axX.ticklabel_format(axis='y',style='sci',scilimits=(0,0)) #use scientific notation for y axis
axX.set_ylabel(F"DNN Error X",fontsize=12) #set y axis title

axK.scatter(predCount,dnnErrorK,facecolors='none',edgecolors='red') #make scatter plot for K
axK.ticklabel_format(axis='y',style='sci',scilimits=(0,0)) #use scientific notation for y axis
axK.set_xlabel(F"Scikit-Learn Prediction Number",fontsize=12) #set x axis title
axK.set_ylabel(F"DNN Error K",fontsize=12) #set y axis title

fig.savefig(f'predError.pdf',format='pdf') #save plot

#write the configuration of the model
with open("modelConfiguration.txt","w") as modConfig:
    print(model.get_params(),file=modConfig)

sys.stdout.close()
