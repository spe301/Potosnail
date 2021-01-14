# Potosnail

<img src="Images/download.jpg/">
This is a collection of helper functions that can accelerate Machine Learning and allow me to move much faster

# Installation and import 

pip install Potosnail==0.0.5

import potosnail

# Tutorials
Check out the notebooks in the tutorials folder to see some implementation.

# Code Structure

## Why do these model building functions take in 10 arguments?


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
    input: nodes(n nodes you want in the 1st layer), activation, regularizer, stacking(make first 2 layers the same), dropout (true or false),
    nlayers, closer(extra layer before output layer), loss, optimizer, y_col(same as target_str)
    output: keras model
  * DeepTabularClassification
    input: output_dim, nodes(n nodes you want in the 1st layer), activation, regularizer, stacking(make first 2 layers the same), dropout,
    nlayers, closer(extra layer before output layer), loss, optimizer
    output: keras model
  * FastNN
   input: task, loss, output_dim(default=None), nodes(default=64), activation(default='relu'), regularizer(default=None), stacking(default=False), dropout(default=False),          nlayers(default=4), closer(default=False), optimizer(default='adam')
   output: keras model
  * RNN
   input: output_dim, embedding, nodes, activation, regularizer, stacking, dropout, optimizer, method(LSTM or GRU), bidirectional(true or false)
   output: keras model
  * FastRNN
   input: requires outpud_dim and embedding, nodes, activation, regularizer, stacking, dropout, optimizer, method, and bidirectional 
   output: keras model
  * CNN
   input: output_dim, base_filters(n filters for first convlutional block), kernel_size(int), activation, nblocks, pool(int), dropout, closer, optimizer, metrics
   output: keras model
  * FastCNN
   input: requires output_dim, base_filters(default=32), kernel_size(default=3), activation(default='relu'), nblocks(default=3), pool(default=2), dropout(default=True),            closer(default=False), optimizer(default='adam'), metrics(default='accuracy') 
   output: keras model
  * TestDL
    input: params, func(keras model building function), task, X(input array), y(output array), X_val(default=None), y_val(default=None), batch_size(default=64),                     epochs(default=50) 
    output: tuned keras gridsearch, use .best_estimator at the end to get the model or .best_params to get it's parameter combination
  * CollectPerformance
    input: params, func(must be a keras model building function that takes in 10 arguments), X, y, epochs(default=50), batch_size(default=64), patience(how many epochs a model can stagnate before fitting stops, default=5), regression(default=False)
    output: a dataframe showing the performance for all hyperparameter combinations
  * ClassifyImage
   input: model_dir(directory to saved .h5 file), image(image must be a tensor), classes(list of strings)
   output: the image and it's class probability (ie. picture of a cat, 'cat', 95%)
  * ClassifyText
   input: model_dir, text_str, pad(how many words you want to use) https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences to learn more
   output: predicted class of the given string
  * MulticlassOutput
    input: labels(list column or (n,1) shaped array)
    output: a One hot encoded array
  * ModelReadyText1
    input: text(list or column), labels, pad(reccommended: average n words throughout the dataset)
    output: array to tokenized bodies of text
  * ModelReadyText2
   input: text, labels, num_words(reccomended: n words in vocab)
   output: a vector of 1's and 0's representing wheather each word in the vocabulary is used in the text or not
  * ModelReadyPixels
    input: directories(directories to the file of training and testing for each class, can be made with DataHelper().MakeDirs()), classes, target_size(default=(150, 150))
    output: image tensors and OneHotEncoded labels
3. DataHelper
  * HoldOut
    input: data(any dataframe, array, or list)
    output: train data and test data
  * MakeNewDf
    input: X, y, k(n features you want to keep)
    output: a dataframe with only the best k features
  * ScaleData
    input: strategy('standard', 'minmax', 'mean', 'pca'), data(dataframe), y_var(string), dim(default=None, only nessecary if using pca strategy)
    output: a dataframe with the y_var column dropped and scaled accordingly
  * VifIt
   input: X(dataframe with y dropped)
   output: Vif(variance inflation-factor) scores for all features
  * SmoteIt
    input: X(dataframe with y dropped), y(dataframe[ycol])
    output: X without class imbalance(ie. if only 100 of 1,000 patients in a dataset have cancer, SmoteIt will return the data + 900 synthetic cancer patient datapoints)
  * MakeDirs
    input: train_dir, test_dir, classes
    output: directories(directories to the file of training and testing for each class)
4. Evaluater
  * ScoreModel
    input: model(fitted sklearn model), X, y
    output: training and validation scores for the model
  * BuildConfusion 
    input: fitted_model(must be a classifier), Xval, yval, cmap(default='plasma')
    output: a confusion matrix
  * BuildConfusionDL
    input: model(fitted keras model), X, y, normalize(default='true'), normalize(cmap='plasma')
    output: a confusion matrix
  * BuildTree
    input: tree(fitted sklearn.tree model)
    output: a tree plot
  * GetCoefficients
    input: model(sklearn LinearRegression model), X, y
    output: model coefficients(m values and beta coefficient)
  * GetImportance
    input: model(sklearn.ensemble model), X, y
    output: feature importance visualization
  * EvaluateRegressor
    input: model(sklearn regression model), X, Xval, y, yval
    output: a dataframe comparing predictions on Xval to yval and RMSE score
  * BinaryCBA
    input: trained_model(must be a binary classifier), X, y, value(value of product/service), discount(proposed discount to customers predicted to churn)
    output: money gained - money lost if model were used
  * DoCohen
    input: group1, group2 (both a list or column)
    output: cohen's D  https://www.youtube.com/watch?v=IetVSlrndpI&list=LL&index=4&t=307s to learn more
  * ViewAccuracy
    input: history, epochs
    output: graph showing accuracy throughout training
  * ViewLoss
    input: history, epochs
    output: graph showing loss throughout training
  * AUC
    input: model(sklearn classifier), Xval, yval
    output: AUC score
