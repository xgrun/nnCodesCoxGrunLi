# -*- coding: utf-8 -*-
"""
Authors: Nicholas Cox and Xavier Grundler
Description: Code for training and testing a deep neural network for an inversion problem using the TensorFlow package with Keras as the backend. Also produces graphs of the results
"""

#import TensorFlow and supporting packages, Scikit-Learn used for preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as splitSet
from sklearn.preprocessing import StandardScaler as SS
import tensorflow as tf

columns = ["xsection","incompressability","v1n","v2n","v1p","v2p"] #names for data, explained in Cox, Grundler and Li
dataset_number = fr"./xytrain90.dat"
dataset = pd.read_csv(dataset_number, sep = ',', names = columns)

#this scaling removes the mean and scales data to unit variance
scaling = SS()
scaling2 = SS()

#for inversion, the inputs are v1 and v2 and the outputs are X and K
xset = dataset[["v1p","v2p"]]
yset = dataset[["xsection","incompressability"]]

#split data: 75% training, 25% testing
xtr, xte, ytr, yte = splitSet(xset, yset, test_size = .25)

#scale the training data to be between -1 and 1, then apply that same transformation to the testing data
xtr = scaling.fit_transform(xtr)
xte = scaling.transform(xte)
ytr = scaling2.fit_transform(ytr)
yte = scaling2.transform(yte)

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

#train the model
model.fit(xtr, ytr, epochs = 250)

#test model
tloss,tacc = model.evaluate(xte,yte,verbose=2)
print(tloss,tacc) #output loss and metrics from testing the model

#make predictions with the model
predictions = model.predict(xte)

#score the model
sse = np.sum(np.square(predictions - yte))
tss = np.sum(np.square(yte-yte.mean()))
r2_score = 1-sse/tss
residues = yte - predictions
print("R^2=",r2_score)

#rescale predictions and test data for graphing
predictions = scaling2.inverse_transform(predictions)
predX = predictions[:,0]
predK = predictions[:,1]
true = scaling2.inverse_transform(yte)
trueX = true[:,0]
trueK = true[:,1]

#prediction vs. perfection for X, graph
plt.figure()
plt.plot(predX,predX,color = "red")
plt.scatter(predX,trueX,facecolors = 'none',edgecolors = 'black',label = F"R\u00B2: {np.round(r2_score,3)}")
plt.xlabel(r"Predicted X")
plt.ylabel(r"True X")
plt.legend(loc = 'upper left',fontsize = 8)
plt.savefig(f"graphOfX.pdf",format="pdf")

#prediction vs. perfection for K, graph
plt.figure()
plt.plot(predK,predK,color = "red")
plt.scatter(predK,trueK,facecolors = 'none',edgecolors = 'black',label = F"R2 {np.round(r2_score,3)}")
plt.xlabel(r"Predicted K (MeV)")
plt.ylabel(r"True K (MeV)")
plt.legend(loc = 'upper left',fontsize = 8)
plt.savefig(f"graphOfK.pdf",format="pdf")

#find the importance of each input feature (column) in determining the output
permutationarray = np.zeros([1,2])
for n in range(2):
    permutationarray[0,n] = 1
    permutationimportance = model.predict(permutationarray)
    permutationarray[0,n] = 0
    if n == 0:
        print("v1p importance = ", permutationimportance)
    else:
        print("v2p importance = ", permutationimportance)

#write the configuration of the model
with open("modelConfiguration.txt","w") as modConfig:
    print(model.get_config(),file=modConfig)
