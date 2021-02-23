# Potosnail

<img src="Images/download.jpg/">
This is a collection of helper functions that can accelerate Machine Learning and allow me to move much faster

author: Spencer Holley

contact: aacjpw@gmail.com

# Installation and import 

pip install Potosnail==0.0.9

import potosnail

# Tutorials and Case Studies
Check out the notebooks in the tutorials folder to see some implementation. If you want to see more real-world use cases for potosnail, chech out the Case_Studies folder

## Why do these model building functions take in 10 arguments?
<img src="Images/4tqs0v.jpg/">
If you use DeepTabularRegression, DeepTabularClassification, RNN, or CNN from DeepLearning you'll notice that these funtions require 10 arguments, this is so that these functions work well with the CollectPerformance function, this is because I had to code a gridsearch from scratch and there needed to be a fixed number of for loops for a fixed number of parameters. if you don't like this you can use FastNN (for the DeepTabulars), FastRNN, and FastCNN instead, you can run a gridsearch on them with DeepLearning().TestDl().

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
  * ClusterIt2D
    input: data, clusters (number of clusters)
    output: your dataset reduced to 2 features for an easier visuals
   * AMC
     input: X, y, task (classification or regression)
     output: The best sklearn model suitable for the task
    
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
  * PipeIt
    input: scaler, model, X, y
    output: training and testing score of the given model fitted on the scaled data (X)
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
  * AMF
  input: X
  output: the dataset with all features with VIF of 5.5 and up filtered out
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
    input: X(dataframe with y dropped), y(dataframe[ycol]), bool_arr (optional)
    output: X without class imbalance(ie. if only 100 of 1,000 patients in a dataset have cancer, SmoteIt will return the data + 900 synthetic cancer patient datapoints). if bool_arr is not empty, the function will run SMOTENC.
  * MakeDirs
    input: train_dir, test_dir, classes
    output: directories(directories to the file of training and testing for each class)
  * Getcats
    input: X
    output: list of all categorical features
   * Scrape
    input: url
    output: the main bodies of text in a given website, assuming the request is accepted and the site allows scraping
   * GetVocab
     input: df, data_str (name of the column that contains the text)
     output: number of unique words in the text corpus
    * Binarize
      input: df, columns_list (the columns with non-numerical binary values, ie. gender)
      output: new dataframe with all non-numerical binary values
    * OHE
      input: series (a column of categorical string values)
      output: a one hot encoded dataframe
    * Stars2Binary
      input: series (list or array of ratings, ie, scale of 1-5 or 1-10)
      output: a list 1's and 0's. 1 being a higher rating and one being lower. this turns rating prediction into a classification problem rather than a regression.
    
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
  * ACE
    input: fitted_model, metric (accuracy, recall, or precision), Xval, yval
    output: the model's score of the given metric
   * PipeIt: scaler, model, X, y, quiet(default=False)
