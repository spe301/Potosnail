# Potosnail

This is a collection of helper functions that can accelerate Machine Learning and allow me to move much faster

# Installation and import 

pip install Potosnail==0.0.5

import potosnail

# Code Structure
I will list each class and the functions within each class.

1. MachineLearning
  * CompareModels
    input: data(pandas dataframe), target_str, task(classification or regression)
    output: a dataframe showing training ant testing accuracy for different vanilla sklearn models
  * Optimize
    input: model(instantiated sklearn model), parameters(dict), data(pandas dataframe), target_str, metric(default=accuracy)
    output: hypertuned model
  * SeeModel
    input: all arguments from Optimize, task(classification or regression)
    output: hypertuned model, fitted hypertuned model, training & testing accuracy, a confusion matrix, feature importances
  * ClusterIt
    input: data(pandas dataframe), clusters(n categories you want the data segmented into)
    output: a visual representation of clustering the data with K-means
    
2. DeepLearning
  * Powers
    input: n(any positive integer)
    output: how many times n is divisible by 2
  * DeepTabularRegression
    input: nodes(n nodes you want in the 1st layer), activation, regularizer, stacking(make first 2 layers the same), dropout,
    nlayers, closer(extra layer before output layer), loss, optimizer, y_col(same as target_str)
  * DeepTabularClassification
  * FastNN
  * RNN
  * FastRNN
  * CNN
  * FastCNN
  * TestDL
  * CollectPerformance
  * ClassifyImage
  * ClassifyText
  * MulticlassOutput
  * ModelReadyText1
  * ModelReadyText2
  * ModelReadyPixels
3. DataHelper
  * HoldOut
  * MakeNewDf
  * ScaleData
  * VifIt
  * SmoteIt
  * MakeDirs
4. Evaluater
  * ScoreModel
  * BuildConfusion
  * BuildConfusionDL
  * BuildTree
  * GetCoefficients
  * GetImportance
  * EvaluateRegressor
  * BinaryCBA
  * DoCohen
  * ViewAccuracy
  * ViewLoss
  * AUC
