# -*- coding: utf-8 -*-
"""
Authors: Nicholas Cox and Xavier Grundler
Description: Code for training and testing a deep neural networkas an emulator using the Scikit-Learn package. Also produces graphs of the results.
"""

#import Scikit-Learn (Scikit-learn: Machine Learning in Python, Perdgosa et al., JMLR 12, pp.2825-2830, 2011) and supporting packages
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as splitSet
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import make_scorer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MaxAbsScaler as MAS
import os
import sys
import time

columns = ["xsection","incompressability","F1n","v2n","F1p","v2p"] #names for data, explained in Cox, Grundler and Li
datasetDirectory = fr"./xytrain90.dat"
dataset = pd.read_csv(datasetDirectory, sep = ',', names = columns)

#change and make directory for output
os.chdir(sys.argv[1]) #target directory passed from command line
now = dt.datetime.now()
os.makedirs(f"./trial_{now}")
os.chdir(f"./trial_{now}")

#set print options
sys.stdout = open("out.log","w")
np.set_printoptions(threshold=np.inf)

#this scaling divides each column in a numpy array by its largest value, designed to maintain sparsity in datasets during preprocessing
scaling = MAS()
scaling2 = MAS()

#for emulation, the inputs are X and K and the outputs are F1 and v2
xset = dataset[["xsection","incompressability"]].to_numpy()
yset = dataset[["F1p","v2p"]].to_numpy()

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
model = MLPRegressor(hidden_layer_sizes=(2,6,6,2), activation = 'tanh', solver= 'lbfgs', verbose = True, max_iter = 80).fit(xtr,ytr)

end = time.perf_counter() #end counter for training model

print("Time to train model: ",end-start,"s") #print time to train model

#test model accuracy
score = model.score(xte, yte) #R^2=1-SSE/TSS
print("score:",score)

#rescale test data for graphing
truev = scaling2.inverse_transform(yte)
trueF1 = truev[:,0]
truev2 = truev[:,1]

start = time.perf_counter() #begin counter to time how long it takes to make a prediction

pred = scaling2.inverse_transform(model.predict(xte).reshape(-1,2)) #make predictions

end = time.perf_counter() #end counter

predF1 = pred[:,0] #divide into F1 and v2
predv2 = pred[:,1]
perfLineF1 = np.linspace(np.min(predF1) - .05*np.min(predF1), np.max(predF1) + .05*np.max(predF1)) #make lines of perfection
perfLinev2 = np.linspace(np.min(predv2) - .05*np.min(predv2), np.max(predv2) + .05*np.max(predv2))

numPred = np.size(predF1)
print("Time per prediction: ",(end-start)/numPred,"s") #output normalized time per prediction

#record unscaled data
unscaledData = pd.DataFrame({"trueF1":trueF1,"predF1":predF1,"truev2":truev2,"predv2":predv2})
unscaledData.to_csv("unscaledData.csv")

#prediction vs. perfection for F1, graph
plt.figure()
plt.plot(perfLineF1,perfLineF1,color = 'red')
plt.scatter(predF1,trueF1,facecolors = 'none',edgecolors = 'black',label = F"R\u00B2: {np.round(score,3)}")
plt.xlabel(r"$F_1$ (Scikit-Learn)",fontsize = 12)
plt.ylabel(r"$F_1$ (IBUU simulation)",fontsize = 12)
plt.legend(loc = 'upper left',fontsize = 12)
plt.savefig(f'graphOfF1.pdf',format = 'pdf')

#prediction vs. perfection for v2, graph
plt.figure()
plt.plot(perfLinev2,perfLinev2,color = 'red')
plt.scatter(predv2,truev2,facecolors = 'none',edgecolors = 'black',label = F"R\u00B2: {np.round(score,3)}")
plt.xlabel(r"$v_2$ (Scikit-Learn)",fontsize = 12)
plt.ylabel(r"$v_2$ (IBUU simulation)",fontsize = 12)
plt.legend(loc = 'upper left',fontsize = 12)
plt.savefig(f'graphOfv2.pdf',format = 'pdf')

#calculate variance of DNN and IBUU predictions
varTrueF1 = np.var(trueF1,ddof=1)
varPredF1 = np.var(predF1,ddof=1)
varTruev2 = np.var(truev2,ddof=1)
varPredv2 = np.var(predv2,ddof=1)

#calculate sum of variances
sumVarF1 = varTrueF1 + varPredF1
sumVarv2 = varTruev2 + varPredv2

#initialize arrays for recording DNN error
dnnErrorF1 = np.zeros(numPred)
dnnErrorv2 = np.zeros(numPred)
predCount = np.zeros(numPred) #for plotting DNN error as function of run number

#initialize MSE
mseF1 = 0
msev2 = 0

#calculate DNN error for each prediction and sum of squared errors
for i in range(numPred):
    dnnErrorF1[i] = ((predF1[i] - trueF1[i])**2)/sumVarF1
    dnnErrorv2[i] = ((predv2[i] - truev2[i])**2)/sumVarv2
    mseF1 += (predF1[i] - trueF1[i])**2
    msev2 += (predv2[i] - truev2[i])**2
    predCount[i] = i + 1

#save DNN error in csv
dnnDF = pd.DataFrame({"dnnErrorF1":dnnErrorF1,"dnnErrorv2":dnnErrorv2})
dnnDF.to_csv(f"dnnError.csv")

#finish MSE calculation
mseF1 /= numPred
msev2 /= numPred

print("numPred = ",numPred)
print("F1 MSE = ",mseF1)
print("v2 MSE = ",msev2)

#plot DNN error
fig, (axF1, axv2) = plt.subplots(2,sharex=True) #two subplots with shared x axis

axF1.scatter(predCount,dnnErrorF1,facecolors='none',edgecolors='red',label=F'$F_1$ MSE: {np.format_float_scientific(mseF1,precision=2)}') #make scatter plot for F1
axF1.ticklabel_format(axis='y',style='sci',scilimits=(0,0)) #use scientific notation for y axis
axF1.set_ylabel(F"DNN Error $F_1$",fontsize=12) #set y axis title
axF1.legend(loc='upper left',fontsize=12) #set location of legend

axv2.scatter(predCount,dnnErrorv2,facecolors='none',edgecolors='red',label=F'$v_2$ MSE: {np.format_float_scientific(msev2,precision=2)}') #make scatter plot for v2
axv2.ticklabel_format(axis='y',style='sci',scilimits=(0,0)) #use scientific notation for y axis
axv2.set_xlabel(F"Scikit-Learn Prediction Number",fontsize=12) #set x axis title
axv2.set_ylabel(F"DNN Error $v_2$",fontsize=12) #set y axis title
axv2.legend(loc='upper left',fontsize=12) #set location of legend

fig.savefig(f'predError.pdf',format='pdf') #save plot

#write the configuration of the model
with open("modelConfiguration.txt","w") as modConfig:
    print(model.get_params(),file=modConfig)

sys.stdout.close()
