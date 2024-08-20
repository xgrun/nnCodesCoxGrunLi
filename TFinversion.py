# -*- coding: utf-8 -*-
"""
Authors: Nicholas Cox and Xavier Grundler
Description: Code for training and testing a deep neural network for an inversion problem using the TensorFlow package with Keras as the backend. Also produces graphs of the results
"""

#import TensorFlow (TensorFlow: Large-scale machine learning on heterogeneous systems, Abadi et al., software available from tensorflow.org, 2015) and supporting packages
#Keras (Keras, Chollet, https://github.com/fcholet/keras, 2015) used as backend for TensorFlow
#Scikit-Learn (Scikit-learn: Mashine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011) used for preprocessing
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as splitSet
from sklearn.preprocessing import StandardScaler as SS
import os
import sys
import tensorflow as tf
import time

columns = ["xsection","incompressability","v1n","v2n","v1p","v2p"] #names for data, explained in Cox, Grundler and Li
dataset_number = fr"./xytrain90.dat"
dataset = pd.read_csv(dataset_number, sep = ',', names = columns)

#change and make directory for output
os.chdir(sys.argv[1]) #target directory passed from command line
now = dt.datetime.now()
os.makedirs(f"./trial_{now}")
os.chdir(f"./trial_{now}")

#set print options
sys.stdout = open("out.log","w")
np.set_printoptions(threshold=np.inf)

#this scaling removes the mean and scales data to unit variance
scaling = SS()
scaling2 = SS()

#for inversion, the inputs are v1 and v2 and the outputs are X and K
xset = dataset[["v1p","v2p"]]
yset = dataset[["xsection","incompressability"]]

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

#configure the model structure, layers and activation
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation = 'linear'),
    tf.keras.layers.Dense(6, activation = 'tanh'),
    tf.keras.layers.Dense(6, activation = 'tanh'),
    tf.keras.layers.Dense(2)
    ])

#finish configuring the model, optimizer and loss
model.compile(optimizer = 'adam',
              loss = tf.keras.losses.MeanSquaredError(),
              metrics = ['mse'])

start = time.perf_counter() #begin counter for training model

#train the model
model.fit(xtr, ytr, epochs = 250)

end = time.perf_counter() #end counter for training model

print("Time to train model: ",end-start,"s") #print time to train model

#test model
tloss,tacc = model.evaluate(xte,yte,verbose=2)
print(tloss,tacc) #output loss and metrics from testing the model

start = time.perf_counter() #begin counter to time how long it takes to make a prediction

#make predictions with the model
pred = model.predict(xte)

end = time.perf_counter() #end counter

#score the model
sse = np.sum(np.square(pred - yte))
tss = np.sum(np.square(yte-yte.mean()))
r2_score = 1-sse/tss
print("R^2=",r2_score)

#rescale predictions and test data for graphing
pred = scaling2.inverse_transform(pred)
predX = pred[:,0]
predK = pred[:,1]
true = scaling2.inverse_transform(yte)
trueX = true[:,0]
trueK = true[:,1]

numPred = np.size(predX)
print("Time per prediction: ",(end-start)/numPred,"s") #output normalized time per prediction

#record unscaled data
unscaledData = pd.DataFrame({"trueX":trueX,"predX":predX,"trueK":trueK,"predK":predK})
unscaledData.to_csv("unscaledData.csv")

#prediction vs. perfection for X, graph
plt.figure()
plt.plot(predX,predX,color = "red")
plt.scatter(predX,trueX,facecolors = 'none',edgecolors = 'black',label = F"R\u00B2: {np.round(r2_score,3)}")
plt.xlabel(r"X (TensorFlow)",fontsize = 12)
plt.ylabel(r"X (IBUU simulation)",fontsize = 12)
plt.legend(loc = 'upper left',fontsize = 12)
plt.savefig(f"graphOfX.pdf",format="pdf")

#prediction vs. perfection for K, graph
plt.figure()
plt.plot(predK,predK,color = "red")
plt.scatter(predK,trueK,facecolors = 'none',edgecolors = 'black',label = F"R\u00B2: {np.round(r2_score,3)}")
plt.xlabel(r"K (MeV) (TensorFlow)",fontsize = 12)
plt.ylabel(r"K (MeV) (IBUU simulation)",fontsize = 12)
plt.legend(loc = 'upper left',fontsize = 12)
plt.savefig(f"graphOfK.pdf",format="pdf")

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
axK.set_xlabel(F"TensorFlow Prediction Number",fontsize=12) #set x axis title
axK.set_ylabel(F"DNN Error K",fontsize=12) #set y axis title

fig.savefig(f'predError.pdf',format='pdf') #save plot

#write the configuration of the model
with open("modelConfiguration.txt","w") as modConfig:
    print(model.get_config(),file=modConfig)

sys.stdout.close()
