#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:48:31 2022

@author: yadhu
"""

import numpy as np
import pandas as pd



def linearRegression(tData,vData,quality):
    
    print("function")
    print(tData.shape)
    print(quality.shape)
    N,P = tData.shape
    b = np.random.uniform(0,0,(P))
    print(" B ",b)
    b[0]=1
    y= np.zeros(N)
    # while True:
    output = tData.dot(b)
    sqrErrorTrial = (output-quality)**2   
    tempDot = tData.T.dot(tData)
    tempInverse = np.linalg.inv(tempDot)
    tempMul = np.dot(tempInverse,tData.T)
    calWeight = np.dot(tempMul,quality)
    output = tData.dot(calWeight)
    sqrErrorTrial = (output-quality)**2
    print("calculated weigts",calWeight)
    print("output",output)
    print("squeare error",sqrErrorTrial)
    
    
    
def update_weights(features, targets, weights, lr,predictions,P,N):
    '''
    Features:(200, 3)
    Targets: (200, 1)
    Weights:(3, 1)
    '''
    dB = np.zeros([P,N])
    for w in range(1,P):
        dB[w] = -features[:,w].dot(targets - predictions) 
        weights[w] -= (lr * np.mean(dB[w,:]))
    
    
    # predictions = predict(features, weights)

    # #Extract our features
    # x1 = features[:,0]
    # x2 = features[:,1]
    # x3 = features[:,2]

    # # Use dot product to calculate the derivative for each weight
    # d_w1 = -x1.dot(targets - predictions)
    # d_w2 = -x2.dot(targets - predictions)
    # d_w2 = -x2.dot(targets - predictions)

    # # Multiply the mean derivative by the learning rate
    # # and subtract from our weights (remember gradient points in direction of steepest ASCENT)
    # weights[0][0] -= (lr * np.mean(d_w1))
    # weights[1][0] -= (lr * np.mean(d_w2))
    # weights[2][0] -= (lr * np.mean(d_w3))

    return weights
    
    
def normalize(features):
    print("data")
    print(features)
    for feature in features.T:
        fmean = np.mean(feature)
        frange = np.amax(feature) - np.amin(feature)

        #Vector Subtraction
        feature -= fmean

        #Vector Division
        feature /= frange
    print("Normalised Data")
    print(features)    
    return features



if __name__ == "__main__":
    print("Welcome to linear regression")
    df1 = pd.read_csv("winequality-white_labels_missing.csv",delimiter=";")
    df = df1.drop_duplicates(subset=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol'])    
    N,P = df.shape
    endingIndex = np.where(df["quality"].isnull())[0][0]
    validationItems = N-endingIndex
    quality = df.loc[0:endingIndex,["quality"]].to_numpy()
    trainingDf = df.iloc[0:endingIndex,:] 
    quality = trainingDf.loc[:,["quality"]].to_numpy()
    print("quality size",quality.shape)
    validationDf = df.iloc[endingIndex:,:]
    trainingData = trainingDf.to_numpy()
    trainingData = np.c_[ np.ones(endingIndex),trainingData ] 
    validationData = validationDf.to_numpy()
    # normalize(validationData)
    validationData = np.c_[ np.ones(validationItems),validationData ] 
    
    linearRegression(trainingData, validationData,quality)
    
    