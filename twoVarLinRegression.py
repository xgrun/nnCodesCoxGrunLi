"""
Author: Xavier Grundler
Description: Code to calculate linear multiple regression with two independent variables
"""

#import necessary functions
import pandas as pd
import numpy as np

#this function does a linear multiple regression with two independent variables
def regression(n,X1,X2,Y):
    #deal with the independent variables
    X1squared = X1*X1
    X2squared = X2*X2
    X1timesX2 = X1*X2
    sumX1 = np.sum(X1)
    sumX2 = np.sum(X2)
    sumX1squared = np.sum(X1squared)
    sumX2squared = np.sum(X2squared)
    sumX1X2 = np.sum(X1timesX2)

    #deal with the dependent variables
    X1timesY = X1*Y
    X2timesY = X2*Y
    sumY = np.sum(Y)
    sumX1Y = np.sum(X1timesY)
    sumX2Y = np.sum(X2timesY)

    #to solve for a,b,c in the equation Y=a+bX1+cX2
    coefficientMatrix = np.array([n,sumX1,sumX2,sumX1,sumX1squared,sumX1X2,sumX2,sumX1X2,sumX2squared]).reshape(3,3)
    augment = np.array([sumY,sumX1Y,sumX2Y])
    solution = np.linalg.solve(coefficientMatrix,augment)

    print("\na*",coefficientMatrix[0,0]," + b*",coefficientMatrix[0,1]," +c*",coefficientMatrix[0,2]," = ",augment[0])
    print("a*",coefficientMatrix[1,0]," + b*",coefficientMatrix[1,1]," +c*",coefficientMatrix[1,2]," = ",augment[1])
    print("a*",coefficientMatrix[2,0]," + b*",coefficientMatrix[2,1]," +c*",coefficientMatrix[2,2]," = ",augment[2])
    print("\nY = ",solution[0]," + ",solution[1],"*X1 + ",solution[2],"*X2")

    #to find R^2
    yAverage = np.sum(Y)/n
    yPredicted = solution[0]+solution[1]*X1+solution[2]*X2
    yMinusYPred_squared = (Y-yPredicted)**2
    yMinusYAvrg_squared = (Y-yAverage)**2
    Rsquared = 1 - (np.sum(yMinusYPred_squared))/(np.sum(yMinusYAvrg_squared))
    print("\nR^2 = ",Rsquared)
    adjRsquared = ((n-1)*Rsquared-2)/(n-2-1) #takes into account number of independent variables in the model, in this case 2
    print("adjusted R^2 = ",adjRsquared)
    
    return

#import data
labels = ("crossSection","incompressability","v1Neutron","v2Neutron","v1Proton","v2Proton")
dataDirectory = fr"./xytrain90.dat"
xyTrainData = pd.read_csv(dataDirectory,sep=",",names=labels)

#create arrays
x = xyTrainData["crossSection"].to_numpy()
k = xyTrainData["incompressability"].to_numpy()
v1n = xyTrainData["v1Neutron"].to_numpy()
v2n = xyTrainData["v2Neutron"].to_numpy()
v1p = xyTrainData["v1Proton"].to_numpy()
v2p = xyTrainData["v2Proton"].to_numpy()
n = np.size(x)

print("v1n = a + b*X + c*K")
regression(n,x,k,v1n)

print("\n\nv2n = a + b*X + c*K")
regression(n,x,k,v2n)

print("\n\nv1p = a + b*X + c*K")
regression(n,x,k,v1p)

print("\n\nv2p = a + b*X + c*K")
regression(n,x,k,v2p)

print("\n\nX = a + b*v1n + c*v2n")
regression(n,v1n,v2n,x)

print("\n\nX = a + b*v1p + c*v2p")
regression(n,v1p,v2p,x)

print("\n\nK = a + b*v1n + c*v2n")
regression(n,v1n,v2n,k)

print("\n\nK = a + b*v1p + c*v2p")
regression(n,v1p,v2p,k)
