# -*- coding: utf-8 -*-
"""
Authors: Nicholas Cox and Xavier Grundler
Description: Code for training and testing a deep neural network as an emulator using the TensorFlow package with Keras as the backend. Also produces graphs of the results.
"""

#import TensorFlow and supporting packages, Scikit-Learn used for preprocessing
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

columns = ["xsection","incompressability","F1n","v2n","F1p","v2p"] #names for data, explained in Cox, Grundler and Li
dataset_number = fr"./xytrain90.dat"
dataset = pd.read_csv(dataset_number, sep = ',', names = columns)

#change and make directory for output
os.chdir(f"../xavie/tamuc/drLi/nick/allOutputs/TFemu")
now = dt.datetime.now()
os.makedirs(f"./trial_{now}")
os.chdir(f"./trial_{now}")

#set print options
sys.stdout = open("out.txt","w")
np.set_printoptions(threshold=np.inf)

#this scaling removes the mean and scales data to unit variance
scaling = SS()
scaling2 = SS()

#for emulation, the inputs are X and K and the outputs are F1 and v2
xset = dataset[["xsection","incompressability"]]
yset = dataset[["F1p","v2p"]]

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
model.fit(xtr, ytr, epochs = 150)

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
predF1 = pred[:,0]
predv2 = pred[:,1]
true = scaling2.inverse_transform(yte)
trueF1 = true[:,0]
truev2 = true[:,1]

numPred = np.size(predF1)
print("Time per prediction: ",(end-start)/numPred,"s") #output normalized time per prediction

#record unscaled data
unscaledData = pd.DataFrame({"trueF1":trueF1,"predF1":predF1,"truev2":truev2,"predv2":predv2})
unscaledData.to_csv("unscaledData.csv")

#prediction vs. perfection for F1, graph
plt.figure()
plt.plot(predF1,predF1,color = "red")
plt.scatter(predF1,trueF1,facecolors = 'none',edgecolors = 'black',label = F"R\u00B2 {np.round(r2_score,3)}")
plt.xlabel(r"$F_1$ (TensorFlow)",fontsize = 12)
plt.ylabel(r"$F_1$ (IBUU simulation)",fontsize = 12)
plt.legend(loc = 'upper left',fontsize = 12)
plt.savefig(f"graphOfF1.pdf",format="pdf")

#prediction vs. perfection for v2, graph
plt.figure()
plt.plot(predv2,predv2,color = "red")
plt.scatter(predv2,truev2,facecolors = 'none',edgecolors = 'black',label = F"R\u00B2 {np.round(r2_score,3)}")
plt.xlabel(r"$v_2$ (TensorFlow)",fontsize = 12)
plt.ylabel(r"$v_2$ (IBUU Simulation)",fontsize = 12)
plt.legend(loc = 'upper left',fontsize = 12)
plt.savefig(f"graphOfv2.pdf",format="pdf")

#calculate emulator error for test data set
dnnError = np.zeros(numPred)
predCount = np.zeros(numPred)
for i in range(numPred):
    dnnError[i] = (pred[i,0] - true[i,0])**2 + (pred[i,1] - true[i,1])**2
    predCount[i] = i + 1

#calculate MSE for test set
mseTest = np.sum(dnnError)/numPred
print("numPred = ",numPred)
print("MSE_test = ",mseTest)

#plot error
plt.figure()
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.scatter(predCount,dnnError,facecolors = 'none',edgecolors = 'red',label = F"MSE: {np.format_float_scientific(mseTest,precision=2)}")
plt.xlabel(r"TensorFlow Prediction Number",fontsize = 12)
plt.ylabel(r"DNN Error",fontsize = 12)
plt.legend(loc = 'upper left',fontsize = 12)
plt.savefig(f'predError.pdf',format = 'pdf')

#find the importance of each input feature (column) in determining the output
permutationarray = np.zeros([1,2])
for n in range(2):
    permutationarray[0,n] = 1
    permutationimportance = model.predict(permutationarray)
    permutationarray[0,n] = 0
    if n == 0:
        print("x importance = ", permutationimportance)
    else:
        print("k importance = ", permutationimportance)

#write the configuration of the model
with open("modelConfiguration.txt","w") as modConfig:
    print(model.get_config(),file=modConfig)

sys.stdout.close()
