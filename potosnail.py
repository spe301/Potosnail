import pandas as pd
import numpy as np
from math import log
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import  KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import  KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer, OneHotEncoder
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
import os
import random
import re
from urllib.request import urlopen
from bs4 import BeautifulSoup
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from tensorflow.keras.regularizers import L1, L2
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from pytrends.request import TrendReq
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import find_peaks
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa import ar_model, stattools, arima_model
import datetime
import tweepy
from textblob import TextBlob


class MachineLearning:
    
    def CompareModels(self, X, y, task): 
        '''returns out the box accuracy of sklearn models'''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        n = len(np.unique(y))
        if task == 'classification':
            if n == 2:
                methods = [KNeighborsClassifier(), GaussianNB(), DecisionTreeClassifier(), RandomForestClassifier(),
                           AdaBoostClassifier(), GradientBoostingClassifier(), LogisticRegression(),
                           SVC()]
                strs = ['KNN', 'NB', 'DT', 'RF', 'AB', 'GB', 'Log', 'SVM']
            else:
                methods = [KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(),
                           AdaBoostClassifier(), GradientBoostingClassifier(), SVC()]
                strs = ['KNN', 'DT', 'RF', 'AB', 'GB', 'XGB', 'SVM']
        if task == 'regression':
            methods = [LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor(),
                       AdaBoostRegressor(), GradientBoostingRegressor(), SVR()]
            strs = ['Lin', 'KNN', 'DT', 'RF', 'AB', 'GB', 'SVM']
        train_acc = []
        test_acc = []
        for i in range(len(methods)):
            model = methods[i].fit(X_train, y_train)
            train_acc.append(model.score(X_train, y_train))
            test_acc.append(model.score(X_test, y_test))
            c1 = pd.DataFrame(strs)
            c2 = pd.DataFrame(train_acc)
            c3 = pd.DataFrame(test_acc)
            results = pd.concat([c1, c2, c3], axis='columns')
            results.columns = ['Model', 'train_acc', 'test_acc']
        return results
    
    def GetGrids(self, X, y, task):
        al = Algorithms()
        n = X.shape[1]
        neighbors1, neighbors2, neighbors3 = al.Neighbors(X, y, 'classification')
        knc = {'n_neighbors': [5, neighbors1, neighbors2, neighbors3], 
            'weights': ['uniform', 'distance'], 'p': [1, 2]}
        gnb = {'var_smoothing': [1e-13, 1e-11, 1e-9, 1e-7, 1e-5]}
        dtc = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_depth': [2, 5, 8, None]}
        rfc = {'n_estimators': al.Estimators(X, 50), 'criterion': ['gini', 'entropy'], 
            'max_depth': [2, 5, 8, None]}
        abc = {'n_estimators': al.Estimators(X, 50), 'learning_rate': [0.1, 0.5, 1], 'algorithm': ['SAMME', 'SAMME.R']}
        gbc = {'learning_rate': [0.1, 0.5, 1], 'n_estimators': al.Estimators(X, 100), 
            'criterion': ['friedman_mse', 'mse', 'mae']}
        lor = {'penalty': ['l1', 'l2', None], 'C': [0.1, 0.5, 1, 2, 10], 
            'fit_intercept': [True, False]}
        svc = {'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto'], 
            'C': [0.1, 1, 10]}
        lir = {'fit_intercept': [True, False]}
        knr = {'n_neighbors': al.Neighbors(X, y, 'regression')}
        dtr = {'max_depth': [2, 5, 8, None], 'criterion': ['mse', 'mae', 'poisson'], 
            'splitter': ['best', 'random']} 
        rfr = {'n_estimators': al.Estimators(X, 50), 'criterion': ['mse', 'mae'], 
            'max_depth': [2, 5, 8, None]}
        abr = {'n_estimators': al.Estimators(X, 50), 'learning_rate': [0.1, 0.5, 1], 
            'loss': ['linear', 'square', 'exponential']}
        gbr = {'learning_rate': [0.1, 0.5, 1], 'n_estimators': al.Estimators(X, 100), 
            'criterion': ['friedman_mse', 'mse', 'mae']}
        svr = {'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto'], 
            'C': [0.1, 1, 10]}
        if task == 'regression':
            grid = {'lir': [lir, LinearRegression()], 'knr': [knr, KNeighborsRegressor()], 
                    'dtr': [dtr, DecisionTreeRegressor()], 'rfr': [rfr, RandomForestRegressor()], 
                    'abr': [abr, AdaBoostRegressor()], 'gbr': [gbr, GradientBoostingRegressor()], 
                    'svr': [svr, SVR()]}
        if task == 'classification':
            grid = {'knc': [knc, KNeighborsClassifier()], 'dtc': [dtc, DecisionTreeClassifier()], 
                    'rfc': [rfc, RandomForestClassifier()], 'abc': [abc, AdaBoostClassifier()], 
                    'gbc': [gbc, GradientBoostingClassifier()], 'svc': [svc, SVC()]}
            if n == 2:
                grid['gnb'] = [gnb, GaussianNB()]
                grid['lor'] = [lor, LogisticRegression()]
        return grid
    
    def Optimize(self, model, parameters, X, y, metric='accuracy'): #make verbose a kwarg
        '''facilitates a gridsearch on any sklearn model'''
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
            return GridSearchCV(model, parameters, cv=3, scoring=metric, n_jobs=-1, verbose=2).fit(X_train, y_train).best_estimator_
        except:
            return GridSearchCV(model, parameters, cv=3, n_jobs=-1, verbose=2).fit(X_train, y_train).best_estimator_
           
    def SeeModel(self, model, parameters, data, target_str, task, ensemble=False):
        '''runs a gridsearch on a sklearn model and evaluates model preformance'''
        dh = DataHelper()
        ev = Evaluater()
        ml = MachineLearning()
        train, test = dh.HoldOut(data)
        mod = ml.Optimize(model, parameters, train, target_str)
        X = train.drop([target_str], axis='columns')
        y = train[target_str]
        score = ev.ScoreModel(mod, X, y)
        Xval = test.drop([target_str], axis='columns')
        yval = test[target_str]
        if task == 'regression':
            results = ev.EvaluateRegressor(mod, X, Xval, y, yval)
            return mod, score, results
        if task == 'classification':
            fit = mod.fit(X, y)
            cm = ev.BuildConfusion(fit, Xval, yval)
            if ensemble == True:
                importance = ev.GetImportance(mod, X, y)
            else:
                importance = None
        return mod, score, fit, cm, importance
    
    def ClusterIt(data, clusters):
        k = KMeans(n_clusters=clusters).fit(data)
        pred = k.predict(data)
        centers = k.cluster_centers_
        X2 = pd.DataFrame(data)
        y = pd.DataFrame(pred)
        y.columns = ['cluster']
        results = pd.concat([X2, y], axis='columns')
        return results

    def Cluster2D(self, data, clusters):
        dh = DataHelper()
        ml = MachineLearning()
        reduced = dh.ScaleData('pca', data, dim=2)
        return ml.ClusterIt(reduced, clusters)

    def AMC(self, X, y, task):
        gmps = MachineLearning().GetGrids(X, y, task)
        methods = list(gmps.keys())
        results = {}
        show = {}
        for method in methods:
            model = gmps[method][1]
            model.fit(X, y)
            score = model.score(X, y)
            results[score] = gmps[method]
        return results[max(results)]


class DeepLearning:
    
    def DeepTabularRegression(self, nodes, activation, regularizer, stacking, dropout, nlayers, closer, loss, optimizer, y_var_str):
        '''Builds a FeedForward net that does regression on tabular data'''
        output_dim = 1
        oa = 'linear'
        model = models.Sequential()
        if regularizer == 'L1':
            model.add(layers.Dense(nodes, activation=activation, kernel_regularizer=L1(0.005)))
        if regularizer == 'L2':
            model.add(layers.Dense(nodes, activation=activation, kernel_regularizer=L2(0.005)))
        if regularizer == None:
            model.add(layers.Dense(nodes, activation=activation))
        if stacking == True:
            model.add(layers.Dense(nodes, activation=activation))
        if dropout == True:
            model.add(layers.Dropout(0.5))
        if nlayers > 2:
            model.add(layers.Dense(int(nodes/2), activation=activation))
        if nlayers > 3:
            model.add(layers.Dense(int(nodes/4), activation=activation))
        if nlayers > 4:
            for i in range(4, nlayers):
                model.add(layers.Dense(int(nodes/4), activation=activation))
        if closer == True:
            model.add(layers.Dense(2, activation=activation))
        model.add(layers.Dense(output_dim, activation=oa))
        model.compile(loss=loss, optimizer=optimizer)
        return model
    
    def DeepTabularClassification(self, output_dim, nodes, activation, regularizer, stacking, dropout, nlayers, closer, loss, optimizer):
        '''Builds a FeedForward net that does classification on tabular data'''
        if output_dim == 2:
            oa = 'sigmoid'
        else:
            oa = 'softmax'
        model = models.Sequential()
        if regularizer == 'L1':
            model.add(layers.Dense(nodes, activation=activation, kernel_regularizer=L1(0.005)))
        if regularizer == 'L2':
            model.add(layers.Dense(nodes, activation=activation, kernel_regularizer=L2(0.005)))
        if regularizer == None:
            model.add(layers.Dense(nodes, activation=activation))
        if stacking == True:
            model.add(layers.Dense(nodes, activation=activation))
        if dropout == True:
            model.add(layers.Dropout(0.5))
        if nlayers > 2:
            model.add(layers.Dense(int(nodes/2), activation=activation))
        if nlayers > 3:
            model.add(layers.Dense(int(nodes/4), activation=activation))
        if nlayers > 4:
            for i in range(4, nlayers):
                model.add(layers.Dense(int(nodes/4), activation=activation))
        if closer == True:
            model.add(layers.Dense(output_dim*2, activation=activation))
        model.add(layers.Dense(output_dim, activation=oa))
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        return model

    def PipeIt(self, scaler, model, X, y):
        '''an sklearn pipeline that returns the train and test score with scaled data'''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        pipe = Pipeline([('scaler', scaler), ('model', model)]).fit(X_train, y_train)
        return "Training: {}, Validation: {}".format(pipe.score(X_train, y_train), pipe.score(X_test, y_test))
    
    def FastNN(self, task, loss, output_dim=None, nodes=64, activation='relu', regularizer=None, stacking=False, dropout=False, nlayers=4, closer=False, optimizer='adam'):
        '''Build a FeedForward Network without filling in a bunch of parameters'''
        dl = DeepLearning()
        if task == 'regression':
            model = dl.DeepTabularRegression(nodes, activation, regularizer, stacking, dropout, nlayers, closer, loss, optimizer, 'target')
        if task == 'classification':
            output_dim = output_dim
            model = dl.DeepTabularClassification(output_dim, nodes, activation, regularizer, stacking, dropout, nlayers, closer, loss, optimizer)
        return model
    
    def RNN(self, output_dim, embedding, nodes, activation, regularizer, stacking, dropout, optimizer, method, bidirectional):
        '''builds a neural network with LSTM or GRU layer(s)'''
        al = Algorithms()
        if output_dim > 16:
          pen = output_dim*2
        else:
          pen = 16
        if output_dim == 2:
          oa = 'sigmoid'
          loss = 'binary_crossentropy'
        if output_dim == 1:
            oa = 'linear'
            loss = 'MSE'
        if output_dim >= 3:
          oa = 'softmax'
          loss = 'categorical_crossentropy'  
        model = models.Sequential()
        model.add(layers.Embedding(embedding, nodes))
        if method == 'LSTM':
          if regularizer == None:
            if bidirectional == False:
              model.add(layers.LSTM(nodes, activation=activation, return_sequences=stacking))
            else:
              model.add(layers.Bidirectional(layers.LSTM(nodes, activation=activation, return_sequences=stacking)))
          if regularizer == 'L1':
            if bidirectional == False:
              model.add(layers.LSTM(nodes, activation=activation, kernel_regularizer=L1(0.005), return_sequences=stacking))
            else:
              model.add(layers.Bidirectional(layers.LSTM(nodes, activation=activation, return_sequences=stacking)))
          if regularizer == 'L2':
            if bidirectional == False:
              model.add(layers.LSTM(nodes, activation=activation, kernel_regularizer=L2(0.005), return_sequences=stacking))
            else:
              model.add(layers.Bidirectional(layers.LSTM(nodes, activation=activation, return_sequences=stacking)))
        if method == 'GRU':
          if regularizer == None:
            if bidirectional == False:
              model.add(layers.GRU(nodes, activation=activation, return_sequences=stacking))
            else:
              model.add(layers.Bidirectional(layers.GRU(nodes, activation=activation, return_sequences=stacking)))
          if regularizer == 'L1':
            if bidirectional == False:
              model.add(layers.GRU(nodes, activation=activation, kernel_regularizer=L1(0.005), return_sequences=stacking))
            else:
              model.add(layers.Bidirectional(layers.GRU(nodes, activation=activation, return_sequences=stacking)))
          if regularizer == 'L2':
            if bidirectional == False:
              model.add(layers.GRU(nodes, activation=activation, kernel_regularizer=L2(0.005), return_sequences=stacking))
            else:
              model.add(layers.Bidirectional(layers.GRU(nodes, activation=activation, return_sequences=stacking)))
        if dropout == True:
          model.add(layers.Dropout(0.5))
        if stacking == True:
          nodes = nodes//2
          if method == 'LSTM':
            if bidirectional == False:
              model.add(layers.LSTM(nodes, activation=activation, return_sequences=stacking))
            else:
              model.add(layers.Bidirectional(layers.LSTM(nodes, activation=activation, return_sequences=stacking)))
          if method == 'GRU':
            if bidirectional == False:
              model.add(layers.GRU(nodes, activation=activation, return_sequences=stacking))
            else:
              model.add(layers.Bidirectional(layers.GRU(nodes, activation=activation, return_sequences=stacking)))
        p = al.Powers(nodes)
        for i in range(p):
          nodes /= 2
          nodes = int(nodes)
          if nodes > 16:
            if nodes > output_dim:
              model.add(layers.Dense(nodes, activation=activation))
        model.add(layers.Dense(pen, activation=activation))
        model.add(layers.Dense(output_dim, activation=oa))
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        return model
    
    def FastRNN(self, output_dim, embedding, nodes=64, activation='tanh', regularizer=None, stacking=False, dropout=None, optimizer='adam', method='GRU', bidirectional=True):
        '''Build a Recurrent Network without filling in a bunch of parameters'''
        dl = DeepLearning()
        return dl.RNN(output_dim, embedding, nodes, activation, regularizer, stacking, dropout, optimizer, method, bidirectional)
    
    def CNN(self, output_dim, base_filters, kernel_size, activation, nblocks, pool, dropout, closer, optimizer, metrics):
        '''Builds a Convlolutional Network'''
        if output_dim == 2:
            oa = 'sigmoid'
            loss = 'binary_crossentropy'
        if output_dim == 1:
            oa = 'linear'
            loss = 'MSE'
        else:
            oa = 'softmax'
            loss = 'categorical_crossentropy'
        model = models.Sequential()
        model.add(layers.Conv2D(base_filters, (kernel_size, kernel_size), activation=activation))
        model.add(layers.MaxPooling2D(pool, pool))
        if nblocks > 1:
            for i in range(nblocks-1):
                model.add(layers.Conv2D(base_filters*2, (kernel_size, kernel_size), activation=activation))
                model.add(layers.MaxPooling2D(pool, pool))
        model.add(layers.Flatten())
        if dropout == True:
            model.add(layers.Dropout(0.5))
        model.add(layers.Dense(base_filters*2, activation=activation))
        if base_filters/2 >= output_dim*2:
            model.add(layers.Dense(base_filters/2, activation=activation))
        if closer == True:
            model.add(layers.Dense(output_dim*2, activation=activation))
        else:
            if base_filters/2 < output_dim*2:
                model.add(layers.Dense(base_filters/2, activation=activation))
        model.add(layers.Dense(output_dim, activation=oa))  
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return model
    
    def FastCNN(self, output_dim, base_filters=32, kernel_size=3, activation='relu', nblocks=3, pool=2, dropout=False, closer=False, optimizer='adam', metrics='accuracy'):
        '''Build a Convolutional Network without filling in a bunch of parameters'''
        dl = DeepLearning()
        return dl.CNN(output_dim, base_filters, kernel_size, activation, nblocks, pool, dropout, closer, optimizer, metrics)
    
    def TestDL(self, params, func, task, X, y, batch_size=64, epochs=50, cv=3, patience=5):
        '''wraps the keras wrapper functions and GridSearchCV into a simple one liner, one line plus the parameter grid'''
        early_stopping = [EarlyStopping(patience=patience), ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5')]
        if task == 'classification':
            k = KerasClassifier(func)
        if task == 'regression':
            k = KerasRegressor(func)
        grid = GridSearchCV(k, params, cv=cv)
        grid.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=early_stopping)
        return grid 
    
    def CollectPerformance(self, params, func, X, y, epochs=50, batch_size=32, patience=5, regression=False):
        '''puts model training results from a gridsearch into a DataFrame'''
        early_stopping = [EarlyStopping(patience=patience), ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5')]
        n = list(params.keys())
        fits = len(params[n[0]])*len(params[n[1]])*len(params[n[2]])*len(params[n[3]])*len(params[n[4]])*len(params[n[5]])*len(params[n[6]])*len(params[n[7]])*len(params[n[8]])*len(params[n[9]])
        print('preforming {} total fits ...'.format(fits))
        lst1 = []
        lst2 = []
        lst3 = []
        lst4 = []
        lst5 = []
        lst6 = []
        lst7 = []
        lst8 = []
        lst9 = []
        lst10 = []
        acc = []
        loss = []
        val_acc = []
        val_loss = []
        epics = []
        bs = []
        progress = 0
        for i in range(len(params[n[0]])):
          var1 = params[n[0]][i]
          for i in range(len(params[n[1]])):
            var2 = params[n[1]][i]
            for i in range(len(params[n[2]])):
              var3 = params[n[2]][i]
              for i in range(len(params[n[3]])):
                var4 = params[n[3]][i]
                for i in range(len(params[n[4]])):
                  var5 = params[n[4]][i]
                  for i in range(len(params[n[5]])):
                    var6 = params[n[5]][i]
                    for i in range(len(params[n[6]])):
                      var7 = params[n[6]][i]
                      for i in range(len(params[n[7]])):
                        var8 = params[n[7]][i]
                        for i in range(len(params[n[8]])):
                          var9 = params[n[8]][i]
                          for i in range(len(params[n[9]])):
                            var10 = params[n[9]][i]
                            lst1.append(var1)
                            lst2.append(var2)
                            lst3.append(var3)
                            lst4.append(var4)
                            lst5.append(var5)
                            lst6.append(var6)
                            lst7.append(var7)
                            lst8.append(var8)
                            lst9.append(var9)
                            lst10.append(var10)
                            history = func(params[n[0]][i], params[n[1]][i], params[n[2]][i], params[n[3]][i], params[n[4]][i], 
                                        params[n[5]][i], params[n[6]][i], params[n[7]][i], params[n[8]][i], params[n[9]][i]).fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=early_stopping)
                            progress += 1
                            print('{} of {} fits complete!'.format(progress, fits))
                            if regression == False:
                                acc.append(history.history['accuracy'][-1])
                            loss.append(history.history['loss'][-1])
                            if regression == False:
                                val_acc.append(history.history['val_accuracy'][-1])
                            val_loss.append(history.history['val_loss'][-1])
                            epics.append(len(history.history['loss']))
                            bs.append(batch_size)
        if regression == False:                    
            results = {n[0] : lst1, n[1] : lst2, n[2] : lst3, n[3] : lst4, 
                       n[4] : lst5, n[5] : lst6, n[6] : lst7, n[7] : lst8, 
                       n[9]: lst10, 'epochs': epics, 'batch_size': bs, n[8] : lst9, 
                       'accuracy': acc, 'loss': loss, 'val_accuracy': val_acc, 'val_loss': val_loss}
        else:
            results = {n[0] : lst1, n[1] : lst2, n[2] : lst3, n[3] : lst4, 
                       n[4] : lst5, n[5] : lst6, n[6] : lst7, n[7] : lst8, 
                       n[9]: lst10, 'epochs': epics, 'batch_size': bs, n[8] : lst9, 
                       'loss': loss, 'val_loss': val_loss}
        df = pd.DataFrame(results)
        df.columns = list(results.keys())
        return df
    
    def ClassifyImage(self, model_dir, image, classes):
        '''uses a pretrained model to classify an individual image'''
        model = models.load_model(model_dir)
        plt.imshow(image[0])
        clsix = int(round(model.predict(image)[0][0]))
        pred = classes[clsix]
        return pred, model.predict(image)[0][0]
    
    def ClassifyText(self, model_dir, text_str, pad):
        '''uses a pretrained model to classify an individual body of text'''
        model = models.load_model(model_dir)
        text = [text_str]
        t = Tokenizer()
        t.fit_on_texts(text)
        tokens = t.texts_to_sequences(text)
        tokens2 = pad_sequences(tokens, maxlen=pad)
        result = model.predict(tokens2)
        return result
    
    def MulticlassOutput(self, labels):
      '''output preprocessing'''
      enc = OneHotEncoder()
      y = labels.reshape(-1, 1)
      oh = enc.fit_transform(y).toarray()
      return oh
    
    def ModelReadyText1(self, text, labels, pad):
      '''converts text into tokenized sequences'''
      t = Tokenizer()
      t.fit_on_texts(text)
      tokens = t.texts_to_sequences(text)
      tokens2 = pad_sequences(tokens, maxlen=pad)
      dl = DeepLearning()
      y = dl.MulticlassOutput(np.array(labels))
      return tokens2, y

    def ModelReadyText2(self, text, labels, num_words):
      '''converts text into one-hot-encoded vectors'''
      dl = DeepLearning()
      text = list(text)
      t = Tokenizer(num_words=num_words)
      t.fit_on_texts(text)
      oh = t.texts_to_matrix(text)
      y = dl.MulticlassOutput(np.array(labels))
      return oh, y
    
    def ModelReadyPixles(self, directories, classes, target_size=(150, 150)):
        '''gets images ready to be fed into a CNN'''
        n_classes = len(classes)
        if n_classes == 2:
            class_mode = 'binary'
        else:
            class_mode = 'categorical'
        idg = ImageDataGenerator(rescale=1./255)
        lens = []
        n = len(directories)
        for i in range(2, n):
            lens.append(len(os.listdir(directories[i])))
            bstr = sum(lens[:n_classes])
            bste = sum(lens[n_classes:])
            trig = idg.flow_from_directory(batch_size=bstr, directory=directories[0], shuffle=True, target_size=target_size, class_mode=class_mode)
            teig = idg.flow_from_directory(batch_size=bste, directory=directories[1], shuffle=True, target_size=target_size, class_mode=class_mode)
            tri, trl = next(trig)
            tei, tel = next(teig)
        dl = DeepLearning()
        tray = dl.MulticlassOutput(trl)
        tey = dl.MulticlassOutput(tel)
        return tri, tei, tray, tey


class DataHelper:

    def SmoteIt(self, X, y, yvar='target'):
        '''must be a binary dataset'''
        info = dict(pd.Series(y).value_counts())
        info2 = {y:x for x,y in info.items()}
        less = info2[min(info.values())]
        mx = max(info2)
        mn = min(info2)
        ix = mx - mn
        df = pd.DataFrame(X)
        df[yvar] = list(y)
        if ix<=mn:
            df2 = df.loc[df[yvar]==less][:ix]
            df3 = pd.concat([df, df2])
        if ix<=mn:
            df2 = df.loc[df[yvar]==less][:ix]
            df3 = pd.concat([df, df2])
        else:
            ix = mx % mn
            df1 = df.loc[df[yvar]==less]
            df2 = df1[:ix]
            dfs = [df2]
            for i in range((mx // mn)-1):
                dfs.append(df1)
            df4 = pd.concat(dfs)
            df3 = pd.concat([df, df4])
        return df3

    def AMF(self, X):
        '''automatically filters out features with high multicolinearity'''
        vif_scores = pd.DataFrame(DataHelper().VifIt(X))
        vif_scores.columns = ['vif']
        vif_scores['fname'] = list(X.columns)
        try:
            df = X[list(vif_scores.loc[vif_scores['vif'] < 5.5]['fname'])]
        except:
            return 'your data sucks!!'
        return df
    
    def HoldOut(self, data):
        '''puts 10% of the data into a seperate dataset for testing purposes'''
        train, test = train_test_split(data, test_size=0.1)
        return train, test
    
    def MakeNewDF(self, X, y, k):
        '''drops less important features with sklearn's SelectKBest'''
        selector = SelectKBest(k=k).fit(X, y)
        mask = selector.get_support()
        selected = []
        for i in range(len(mask)):
            if mask[i] == True:
                selected.append(X.columns[i])
        df = pd.DataFrame(selector.transform(X))
        df.columns = selected
        return df
    
    def ScaleData(self, strategy, X, dim=None): #fix column problem
        '''Scales data via minmax, standard, mean, or PCA scaling'''
        if strategy == 'minmax':
            return pd.DataFrame(MinMaxScaler().fit(X).transform(X))
        if strategy == 'standard':
            return pd.DataFrame(StandardScaler().fit(X).transform(X))
        if strategy == 'pca':
            try:
                return pd.DataFrame(PCA(n_components=dim).fit_transform(X))
            except:
                return 'please pass an integer for dim'
    
    def VifIt(self, X):
        '''returns VIF scores to help prevent multicolinearity'''
        vif = pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)
        return vif
    
    def MakeDirs(self, train_dir, test_dir, classes, data_type='images'):
        '''makes filepaths, intended for loading in image data'''
        classes.reverse()
        rc = classes
        if data_type == 'images':
            dirs = [train_dir, test_dir]
            for c in rc:
                dirs.append(os.path.join(train_dir, c))
            for i in range(len(classes)):
                dirs.append(os.path.join(test_dir, rc[i]))
        return dirs
    
    def GetCats(self, X):
        '''returns an array of booleans with 'True' indicating that a feature is categorical'''
        feats = list(X.columns)
        boolys = []
        for feat in feats:
            boolys.append(list(np.unique(X[feat])) == [0.0, 1.0])
        return np.array(boolys)

    def Scrape(self, url):
        '''Scrapes a wikikedia article'''
        source = urlopen(url).read()
        soup = BeautifulSoup(source, 'lxml')
        text = soup.findAll('p')
        article = ''
        for i in range(len(text)):
            segment = text[i].text
            article += segment.replace('\n', '').replace('\'', '').replace(')', '')
            article = article.lower()
            clean = re.sub("([\(\[]).*?([\)\]])", '', article)
            clean2 = re.sub(r'\[(^)*\]', '', clean)
        return clean

    def GetVocab(self, df, data_str):
        '''returns the vocab size in a text corpus'''
        words = []
        for i in range(len(df)):
            word_lst = list(df[data_str])[i].replace('\n', ' ').split(' ')
            for word in word_lst:
                words.append(word.replace('.', '').replace(',', '').replace(' ', '').replace('"', '').replace(':', '').replace(';', '').replace('!', ''))
        return len(np.unique(words))

    def Binarize(self, df, columns_list):
        for col in columns_list:
            booly = list(df[col].apply(lambda x: x==df[col][0], False))
            inty = list(map(int, booly))
            df[col] = inty
        return df

    def OHE(self, series):
        ohe = OneHotEncoder()
        oh = pd.DataFrame(ohe.fit_transform(np.array(series).reshape(-1, 1)).toarray())
        oh.columns = list(np.unique(series))
        return oh

    def Stars2Binary(self, series):
        thresh = int(round(max(series)/2))
        booly = list(series.apply(lambda x: x>=thresh, True))
        return list(map(int, booly))

                
class Evaluater:
    
    def ScoreModel(self, model, X, y):
        '''returns accuracies for any sklearn model'''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        algo = model.fit(X_train, y_train)
        return 'Training: {}, Validation: {}'.format(algo.score(X_train, y_train), algo.score(X_test, y_test))
    
    def BuildConfusion(self, fitted_model, Xval, yval, cmap='Blues'):
        '''returns a Confusion Matrix given a pretrained sklearn model'''
        return plot_confusion_matrix(fitted_model, Xval, yval, cmap=cmap)
        
    def BuildConfusionDL(self, model, X, y, normalize='true', cmap='Blues'):
      '''displays a confusion matrix to evaluate a deep learning classifier'''
      yreal = y.argmax(axis=1)
      pred = model.predict(X)
      prediction = pred.argmax(axis=1)
      cm = confusion_matrix(yreal, prediction, normalize=normalize)
      plot = sns.heatmap(cm, annot=True, cmap=cmap);
      plot.set_ylabel('True')
      plot.set_xlabel('Predict')
      return plot
    
    def BuildTree(self, tree):
        '''a copy of plot_tree() from sklearn'''
        try:
            return plot_tree(tree)
        except:
            return 'Please pass a fitted model from the tree class'
    
    def GetCoefficients(self, model, X, y):
        '''returns coefficients from a sklearn model'''
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
            model.fit(X_train, y_train)
            return 'coefficients: {}'.format(model.coef_)
        except:
            return 'Please pass LinearRegression, LogisticRegression, or an SVM with a linear kernel'
        
    def GetImportance(self, model, X, y):
        '''returns feature importances from an ensemble sklearn model'''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        model.fit(X_train, y_train)
        try:
            FI = model.feature_importances_
            n_features = X_train.shape[1]
            plt.figure(figsize=(8,8))
            plt.barh(range(n_features), FI, align='center') 
            plt.yticks(np.arange(n_features), X_train.columns.values) 
            plt.xlabel('Feature importance')
            plt.ylabel('Feature')
        except:
            return 'Please pass an ensemble class'
        
    def EvaluateRegressor(self, model, X, Xval, y, yval):
        '''returns rmse and accuracy of a sklearn regression model'''
        model.fit(X, y)
        pred = model.predict(Xval)
        n = len(Xval)
        a = list(yval)
        e = []
        pe = []
        RMSE = 0
        for i in range(n):
            e.append(abs(pred[i] - a[i]))
            RMSE += ((a[i] - pred[i])**2) / n
            if pred[i] > a[i]:
                pe.append(a[i]/pred[i])
            else:
                pe.append(100 - ((pred[i]/a[i])*100))
        p = pd.DataFrame(pred)
        a = pd.DataFrame(a)
        e = pd.DataFrame(e)
        pe = pd.DataFrame(pe)
        results = pd.concat([p, a, e, pe], axis='columns')
        results.columns = ['predicted', 'actual', 'error', '%error']
        score = model.score(Xval, yval)*100
        return results, RMSE, round(score, 2)
    
    def BinaryCBA(self, model, X, y, rate, total_cost, discount, quiet=True):
        '''calculates the potential revenue with and of using the model to make decisions'''
        cm = confusion_matrix(y, model.predict(X))
        tn = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tp = cm[1][1]
        cost = (fp*discount)+(fn*total_cost)
        benefit = tp*((rate+total_cost)-discount)
        if quiet == False:
            return 'the cost is ${} and the benefit is ${}'.format(cost, benefit)
        else:
            return cost, benefit
        
    def DoCohen(self, group1, group2):
        '''calculates Cohen's D between 2 population samples'''
        n1 = len(group1)
        sd1 = np.std(group1)
        n2 = len(group2)
        sd2 = np.std(group2)
        num = (n1 - 1)*(sd1**2) + (n2 - 1)*(sd2**2)
        denom = (n1 + n2)-2
        pooled_sd = np.sqrt(num/denom)
        numerator = abs(np.mean(group1) - np.mean(group2))
        return numerator/pooled_sd
    
    def ViewAccuracy(self, history):
      '''plots a model's accuracy throughout training'''
      plt.plot(list(range(len(history.history['accuracy']))), history.history['accuracy'], label='train');
      plt.plot(list(range(len(history.history['accuracy']))), history.history['val_accuracy'], label='val');
      plt.legend(loc='best')
      plt.xlabel('epochs')
      plt.ylabel('accuracy')
      return None


    def ViewLoss(self, history):
      '''plots a model's loss throughout training'''
      plt.plot(list(range(len(history.history['loss']))), history.history['loss'], label='train');
      plt.plot(list(range(len(history.history['loss']))), history.history['val_loss'], label='val');
      plt.legend(loc='best')
      plt.xlabel('epochs')
      plt.ylabel('loss')
      return None
  
    def AUC(self, model, Xval, yval):
      '''displays AUC to evaluate a classification model'''
      pred = model.predict(Xval)
      fpr, tpr, threshold = roc_curve(yval, pred)
      return auc(fpr, tpr)

    def ACE(self, fitted_model, metric, Xval, yval, merged=True):
        '''Automated Classifier Evaluation'''
        pred = fitted_model.predict(Xval)
        cm = confusion_matrix(yval, pred)
        if metric == 'accuracy':
            score1 = fitted_model.score(Xval, yval)
        if metric == 'recall':
            score1 = cm[1][1] / (cm[1][0] + cm[1][1])
        if metric == 'precision':
            score1 = cm[0][0] / (cm[0][0] + cm[0][1])
        return score1

    def PipeIt(self, scaler, model, X, y, quiet=False):
        '''an sklearn pipeline that returns the train and test score with scaled data'''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        pipe = Pipeline([('scaler', scaler), ('model', model)]).fit(X_train, y_train)
        if quiet == True:
            return pipe.score(X_test, y_test)
        else:
            return "Training: {}, Validation: {}".format(pipe.score(X_train, y_train), pipe.score(X_test, y_test))
        
    def InspectTree(self, tree, X, y, forest=False):
        '''inspects a decision tree or random forrest'''
        ev = Evaluater()
        clf = tree.fit(X, y)
        if forest == True:
            try:
                return ev.GetImportance(clf, X, y)
            except:
                return 'please pass a RandomForestClassifier or other ensemble model for "tree"'
        else:
            try:
                return ev.BuildTree(clf)
            except:
                return 'please pass a DecisionTreeClassifier or other tree model for "tree"'

    def BinaryBarGraph(self, df, opt1, opt2, var):
        yes = len(df.loc[df[var] == opt1])/len(df)
        no = 1 - yes
        X = [opt1, opt2]
        y = [yes, no]
        sns.barplot(X, y);
        plt.show()
        return 'The sample is {}% {} and {}% {}'.format(int(round(yes*100)), opt1, int(round(no*100)), opt2)

    def GetTopN(self, df, var, n):
        dict(df[var].value_counts())
        X = list({v: k for k, v in dict(df[var].value_counts()).items()}.values())[:n]
        y = list({v: k for k, v in dict(df[var].value_counts()).items()}.keys())[:n]
        sns.barplot(X, y);
        plt.show()


class Algorithms:

    def GetDC(self, y):
        vc = list(dict(pd.Series(y.argmax(axis=1)).value_counts()).values())
        return max(vc) / sum(vc)
    
    def Powers(self, n):
        '''returns how many times n is divisible by 2'''
        k = int(log(n, 2))
        return k

    def Neighbors(self, X, y, task):
        '''returns n_neighbors to test in a KNN'''
        if task == 'classification':
            nclasses = len(np.unique(y))
            sizes = list(y.value_counts())
            neighbors1 = int(min(sizes)/10)
            neighbors2 = int(min(sizes)/nclasses)
            neighbors3 = int(neighbors1/2)
            return neighbors1, neighbors2, neighbors3
        if task == 'regression':
            n = int(0.05 * len(X))
            return list(range(1, n))

    def Estimators(self, X, default):
        '''returns range of n_estimators to test'''
        step = int(len(X)/100)
        tests = [default]
        for i in range(1, 11):
            tests.append(i*step)
        return tests

    def GetMetric(self, y, fn):
        '''determines if the metric should be accuracy or recall'''
        total = len(y)
        sizes = list(y.value_counts())
        if max(sizes) > total*0.55:
            if fn == False:
                metric = 'recall'
            if fn == True:
                metric = 'precision'
        else:
            metric = 'accuracy'
        return metric

    def PickScaler(self, X, y, model):
        '''decides how to scale data'''
        ev = Evaluater()
        n_features = len(list(X.columns))
        if n_features >= 256:
            return 'pca'
        else:
            ss = ev.PipeIt(StandardScaler(), model, X, y, quiet=True)
            mm = ev.PipeIt(MinMaxScaler(), model, X, y, quiet=True)
            if ss >= mm:
                return 'standard'
            else:
                return 'minmax'

    def ReduceTo(self, X):
        '''divides a n columns 5 and converts it to an int'''
        return int(len(list(X.columns))/5)

    def ToTry(self, X):
        '''returns a list of k features to select from'''
        n_features = len(list(X.columns))
        if n_features >= 15:
            k = int(round(n_features/3))
            step = int(round(k/3))
            k_features = [k]
            for i in range(3):
                k += step
                k_features.append(k)
        else:
            k_features = [n_features-1, n_features-2, n_features-3, n_features-4]
        return k_features

    def Imbalanced(self, y):
        '''determines if a dataset is imbalanced'''
        total = len(y)
        sizes = list(y.value_counts())
        if max(sizes) > total*0.55:
            return True
        else:
            return False
        
    def ScoreClf(self, model, metric, X, y):
        '''Quantifies the quality of a sklearn classifier'''
        ev = Evaluater()
        if metric == 'accuracy':
            score = model.score(X, y)
        else:
            cm = confusion_matrix(y, model.predict(X))
            if metric == 'recall':
                s1 = cm[1][1] / (cm[1][1]+cm[1][0]) 
                s2 = cm[0][0] / (cm[0][0]+cm[0][1]) 
            if metric == 'precision':
                s1 = cm[0][0] / (cm[0][0]+cm[0][1])
                s2 = cm[1][1] / (cm[1][1]+cm[1][0])
            if s2 > 0.52:
                score = s1
            else:
                score = (s1*0.5) + (ev.AUC(model, X, y)*0.5)
        return score
    
    def Nodes(self, X, generation=False):
        if generation == True:
            if len(X) < 1000000:
                return [256, 512, 1024]
            else:
                return [512, 1024, 2048]
        else:
            if len(X) > 500000:
                return [128, 256, 512]
            if len(X) > 100000:
                return [64, 128, 256]
            else:
                return [32, 64, 128]
            
    def Epochs(self, structured):
        if structured == True:
            return list(range(25, 300, 25))
        else:
            return list(range(5, 60, 5))
        
    def BatchSize(self, X):
        if len(X) > 1000000:
            return [256, 512, 1024]
        if len(X) > 500000:
            return [128, 256, 512]
        if len(X) > 200000:
            return [64, 128, 256]
        if len(X) > 100000:
            return [64, 128]
        if len(X) > 50000:
            return [32, 64, 128]
        if len(X) > 10000:
            return [32, 64]
        if len(X) > 1000:
            return [16, 32, 64]
        else:
            return [8, 16, 32, 64]


class Wrappers: 

    def TCP(self, X, y):
        '''fully preprocesses labeld text data and finds vocab size, specific to classification problems'''
        dl = DeepLearning()
        avg = 0
        words = []
        for i in range(len(X)):
            avg += len(list(X)[i].split(' '))/len(X)
        vocab = len(set(','.join(X).split(' ')))
        pad = int(avg)
        text, labels = dl.ModelReadyText1(X, y, pad)    
        return text, labels, vocab

    def TRP(self, X, y):
        '''fully preprocesses labeld text data and finds vocab size, specific to regression problems'''
        dl = DeepLearning()
        avg = 0
        words = []
        for i in range(len(X)):
            avg += len(list(X)[i].split(' '))/len(X)
            word_lst = list(X)[i].replace('\n', ' ').split(' ')
            for word in word_lst:
                words.append(word.replace('.', '').replace(',', '').replace(' ', '').replace('"', '').replace(':', '').replace(';', '').replace('!', ''))
        pad = int(avg)
        text, _ = dl.ModelReadyText1(X, y, pad)    
        return text, np.array(y), len(np.unique(words))

    def Vanilla(self, df, target_str, task):
        '''returns the best vanilla model and a corresponding parameter grid'''
        dh = DataHelper()
        ml = MachineLearning()
        train, test = dh.HoldOut(df)
        X = train.drop([target_str], axis='columns')
        Xval = test.drop([target_str], axis='columns')
        y = train[target_str]
        yval = test[target_str]
        mg = ml.AMC(X, y, task)
        grid = mg[0]
        vanilla = mg[1]
        return vanilla, grid, X, Xval, y, yval
    
    def SmoteStack(self, model, grid, X, y, Xval, yval, yvar='target', metric='recall'):
        dh = DataHelper()
        ev = Evaluater()
        ml = MachineLearning()
        df = dh.SmoteIt(X, y, yvar=yvar)
        X2 = df.drop([yvar], axis='columns')
        y2 = df[yvar]
        tm = ml.Optimize(model, grid, X2, y2) 
        score = ev.ACE(tm, metric, Xval, yval)
        return model, score, X2, Xval, y2, yval
    
    def FeatureEngineering(self, X, Xval, y, yval, model, grid, task, target_str, metric='accuracy', optimize=True):
        al = Algorithms()
        dh = DataHelper()
        ev = Evaluater()
        ml = MachineLearning()
        try_list = al.ToTry(X)
        results = {}
        for num in try_list:
            train = dh.MakeNewDF(X, y, num).drop([target_str], axis='columns')
            test = Xval[list(train.columns)]
            if optimize == True:
                fm = ml.Optimize(model, grid, train, y)
            else:
                fm = model.fit(X, y)
            if task == 'classification':
                score = ev.ACE(fm, metric, test, yval)
            if task == 'regression':
                score = ev.EvaluateRegressor(fm, train, test, y, yval)[2]
            train[target_str] = list(y)
            test[target_str] = list(yval)
            results[score] = [fm, train, test, score]
        return results[max(results)]
    
    def RegLoop(self, df, target_str, quiet=True):
        task = 'regression'
        results = {}
        wr = Wrappers()
        ev = Evaluater()
        ml = MachineLearning()
        al = Algorithms()
        dh = DataHelper()
        vanilla, grid, X, Xval, y, yval = wr.Vanilla(df, target_str, task)
        model1 = vanilla.fit(X, y)
        score1 = ev.EvaluateRegressor(model1, X, Xval, y, yval)[2]
        results[score1] = [model1, X, y, Xval, yval, None, None]
        if quiet == False:
            print('{}% accuracy, untuned model, raw data'.format(score1))
        model2 = ml.Optimize(vanilla, grid, X, y)
        score2 = ev.EvaluateRegressor(model2, X, Xval, y, yval)[2]
        results[score2] = [model2, X, y, Xval, yval, None, None]
        if quiet == False:
            print('{}% accuracy, tuned model, raw data'.format(score2))
        scaler = al.PickScaler(X, y, vanilla)
        if scaler == 'pca':
            dim = al.ReduceTo(X)
        else: 
            dim = None
        X3 = dh.ScaleData(scaler, X, dim=dim)
        Xv3 = dh.ScaleData(scaler, Xval, dim=dim)
        model3 = ml.Optimize(vanilla, grid, X3, y)
        score3 = ev.EvaluateRegressor(vanilla, X3, Xv3, y, yval)[2]
        results[score3] = [model3, X3, y, Xv3, yval, scaler, dim]
        if quiet == False:
            print('{}% accuracy, tuned model, data is scaled with {} scaler'.format(score3, scaler))
            if scaler == 'pca':
                print('feartures have been reduced to {}'.format(dim))
        try:
            vdf = pd.DataFrame(dh.VifIt(X3.drop([target_str], axis='columns')))
        except:
            vdf = pd.DataFrame(dh.VifIt(X3))
        vdf.columns = ['vif']
        cols = list(vdf.loc[vdf['vif']<5.5].index)
        X4 = X3[cols]
        Xv4 = Xv3[cols]
        model4 = ml.Optimize(vanilla, grid, X4, y)
        score4 = ev.EvaluateRegressor(vanilla, X4, Xv4, y, yval)[2]
        if quiet == False:
            print('{}% accuracy, tuned model, features have been reduced to {}'.format(score4, len(cols)))
        results[score4] = [model4, X4, y, Xv4, yval, scaler, dim]
        if len(list(X4.columns)) < 10:
            return results
        else:
            FE = wr.FeatureEngineering(X4, Xv4, y, yval, vanilla, grid, task, target_str)
            results[FE[3]] = [FE[0], FE[1], FE[2], scaler, dim]
            if quiet == False:
                print('{}% accuracy, tuned model, features have been reduced to {}'.format(FE[2], len(list(FE[1].columns))))
            return results[max(results)]
        
    def ClfLoop(self, df, target_str, fn=False, quiet=True):
        task ='classification'
        wr = Wrappers()
        ev = Evaluater()
        al = Algorithms()
        ml = MachineLearning()
        dh = DataHelper()
        results = {}
        vanilla, grid, X, Xval, y, yval = wr.Vanilla(df, target_str, task)
        metric = 'precision'
        model1 = vanilla.fit(X, y)
        metric = al.GetMetric(y, fn=fn)
        score1 = ev.ACE(model1, metric, Xval, yval)
        results[score1] = [model1, X, Xval, y, yval, None, None]
        if quiet==False:
            print('raw data, baseline model: {}'.format(score1))
        model2 = ml.Optimize(vanilla, grid, X, y)
        score2 = ev.ACE(model2, metric, Xval, yval)
        results[score2] = [model2, X, Xval, y, yval, None, None]
        if quiet==False:
            print('raw data, tuned model: {}'.format(score2))
        scaler = al.PickScaler(X, y, model2)
        if scaler == 'pca':
            dim = al.ReduceTo(X)
        else:
            dim = None
        X3 = dh.ScaleData(scaler, X, dim=dim)
        Xval3 = dh.ScaleData(scaler, Xval, dim=dim)
        model3 = ml.Optimize(vanilla, grid, X3, y)
        score3 = ev.ACE(model3, metric, Xval3, yval)
        results[score3] = [model3, X3, Xval3, y, yval, scaler, dim]
        if quiet==False:
            if scaler == 'pca':
                print('data has been reduced to {} features with pca'.format(dim))
            else:
                print('data has been scaled with {} scaler'.format(scaler))
        if al.Imbalanced(y) == True:
            print('Smoting!!')
            print()
            model4, score4, X4, Xval3, y4, yval = wr.SmoteStack(vanilla, grid, X3, y, Xval3, yval, yvar=target_str)
            state = 'smoted'
        else:    
            state = 'not smoted'
            model4, score4, X4, y4 = model3, score3, X3, y
        df4 = X4
        df4[target_str] = list(y4)
        results[score4] = [model4, X4, Xval3, y4, yval, scaler, dim]
        if quiet==False:
            print('data is {}, tuned model: {}'.format(state, score4))
        FE = wr.FeatureEngineering(X4, Xval3, y4, yval, vanilla, grid, task, target_str, metric=metric)
        model5 = FE[0]
        score5 = FE[3]
        results[score5] = [model5, FE[1], FE[2], scaler, dim]
        if quiet==False:
            print('data is {}, tuned model, using {} features: {}'.format(state, FE[1].shape[1], score5))
        return results[max(results)]
    
    def WrapML(self, df, target_str, task, fn=False, quiet=True):
        wr = Wrappers()
        if task == 'regression':
            results = wr.RegLoop(df, target_str, quiet=quiet)
        if task == 'classification':
            results = wr.ClfLoop(df, target_str, fn=fn, quiet=quiet)
        return results


class Stats:
    
     def PDF(self, X, bins=100):
         '''plots a probability density function'''
         hist, bins = np.histogram(X, bins=bins, normed=True)
         bin_centers = (bins[1:]+bins[:-1])*0.5
         plt.plot(bin_centers, hist)

     def Z(self, sample, x):
        mu = np.mean(sample)
        st = np.std(sample)
        try:
            return list((x-mu)/st)[0]   
        except:
            return (x-mu)/st
         
     def IQR(self, values):
        arr = np.array(values)
        uq = np.quantile(arr, 0.75)
        lq = np.quantile(arr, 0.25)
        return uq - lq

     def SqDev(self, values):
        '''measures variance in a dataset'''
        avg = np.mean(values)
        result = 0
        n = len(values)-1
        for num in values:
            result += (num - avg)**2
        return result/n
    
    
class Datasets:
    
    def load_wikipedia(self):
        df = pd.read_csv('https://raw.githubusercontent.com/spe301/AI-generated-AI/main/Data/Wikipedia.csv')
        X, y, vocab = Wrappers().TCP(df['Text'], df['AI'])
        dataset = {'data': X, 'target': y, 'classes': ['Human', 'AI'], 'vocab': vocab}
        return dataset
    
    def load_twitter(self):
        df = pd.read_csv('https://raw.githubusercontent.com/spe301/AI-generated-AI/main/Data/Twitter.csv')
        X, y, vocab = Wrappers().TCP(df['tweet'], df['is_there_an_emotion_directed_at_a_brand_or_product'])
        dataset = {'data': X, 'target': y, 'classes': ['negative', 'neutral', 'positive'], 'vocab': vocab}
        return dataset
    
    def load_rnn(self):
        df = pd.read_csv('https://raw.githubusercontent.com/spe301/AI-generated-AI/main/Data/NLP8.csv')
        try:
            df = df.drop(['Unnamed: 0'], axis='columns')
        except:
            pass
        return df
    
    def load_wikipedia2k(self):
        df = pd.read_csv('https://raw.githubusercontent.com/spe301/Wikipedia-Capstone/main/Data/Wikipedia2k.csv')
        X, y, vocab = Wrappers().TCP(df['Text'], df['AI'])
        dataset = {'data': X, 'target': y, 'classses': ['Human, AI'], 'vocab': vocab}
        return dataset
    
    def load_wikipedia11k(self):
        H = pd.read_csv('')[['Text', 'Human', 'AI']]
        A = pd.read_csv('https://raw.githubusercontent.com/spe301/Wikipedia-Capstone/main/Data/Wikipedia2k.csv')
        AI = A.loc[A['AI'] == 1]
        df = pd.concat([H, A])
        X, y, vocab = Wrappers().TCP(df['Text'], df['AI'])
        dataset = {'data': X}


class DataBuilder:
    
    def Generate(self, X, y, params, task, ts=150, epochs=50, batch_size=32, patience=5, regression=False):
      '''facilitates the Data Collection Process'''
      dl = DeepLearning()
      wr = Wrappers()
      if task == 'NLP':
        func = dl.RNN 
      if task == 'CV':
        func = dl.CNN
      if task == 'TC':
        func = dl.DeepTabularClassification
      if task == 'TR':
        func = dl.DeepTabularRegression
      try:
        df = dl.CollectPerformance(params, func, X, y, epochs=epochs, batch_size=batch_size, patience=patience, regression=regression)
      except:
        df = dl.CollectPerformance(params, func, X, y.reshape(-1), epochs=epochs, batch_size=batch_size, patience=patience, regression=regression)
      size = []
      n_features = []
      batch = []
      depth = []
      dominances = []
      for i in range(len(df)):
        size.append(X.shape[0])
        n_features.append(X.shape[1])
        dominances.append(max(pd.DataFrame(y).value_counts())/len(y))
        if task == 'CV':
          depth.append(X.shape[3])
      df['len_dataset'] = size
      df['n_features'] = n_features
      if regression == False:
        df['dominant_class'] = dominances
      if task == 'CV':
        df['thickness'] = depth
      return df

    def ResultsDL(self, X, y, params, task, ts=150, epochs=50, batch_size=32, patience=5, regression=False):
        '''facilitates the Data Collection Process'''
        if task == 'NLP':
            if regression==False:
                X2, y2, _ = Wrappers().TCP(X, y)
            else:
                X2, y2, _ = Wrappers().TRP(X, y)
            func = DeepLearning().RNN 
        if task == 'CV':
            X2, _, y2, _ = DeepLearning().ModelReadyPixles(X, y, target_size=(ts, ts))
            func = DeepLearning().CNN
        if task == 'TC':
            X2 = np.array(X)
            y2 = DeepLearning().MulticlassOutput(np.array(y))
            func = DeepLearning().DeepTabularClassification
        if task == 'TR':
            X2 = np.array(X)
        y2 = np.array(y)
        func = DeepLearning().DeepTabularRegression
        try:
            df = DeepLearning().CollectPerformance(params, func, X2, y2, epochs=epochs, batch_size=batch_size, patience=patience, regression=regression)
        except:
            df = DeepLearning().CollectPerformance(params, func, X2, y2.reshape(-1), epochs=epochs, batch_size=batch_size, patience=patience, regression=regression)
        size = []
        n_features = []
        batch = []
        depth = []
        dominances = []
        for i in range(len(df)):
            size.append(X2.shape[0])
            n_features.append(X2.shape[1])
            dominances.append(max(pd.DataFrame(y2).value_counts())/len(y2))
            if task == 'CV':
                depth.append(X2.shape[3])
            df['len_dataset'] = size
            df['n_features'] = n_features
            if regression == False:
                df['dominant_class'] = dominances
            if task == 'CV':
                df['thickness'] = depth
        return df

    def ResultsNLP(self, df, gridding=False):
      dh = DataHelper()
      '''gets the nlp results dataset ready for modeling'''
      df['regularizer'] = df['regularizer'].fillna('None')
      df['stacking'] = df['stacking'].astype(int)
      df['dropout'] = df['dropout'].astype(int)
      df['bidirectional'] = df['bidirectional'].astype(int)
      act = dh.OHE(df['activation'])
      reg = dh.OHE(df['regularizer'])
      opt = dh.OHE(df['optimizer'])
      method = dh.OHE(df['method'])
      df = df.drop(['activation', 'regularizer', 'optimizer', 'method'], axis='columns')
      df = pd.concat([df, act, reg, opt, method], axis='columns')
      if gridding == True:
        return df
      df['val_loss'] = df['val_loss'].fillna(max(df['val_loss']))
      df['loss'] = df['loss'].fillna(max(df['loss']))
      kpi_list = ['accuracy', 'loss', 'val_accuracy', 'val_loss']
      kpi = df[kpi_list]
      scores = []
      for i in range(len(df)):
        ts = (1 - (kpi['loss'][i] / max(kpi['loss'])) + kpi['accuracy'][i])/2
        vs = (1 - (kpi['val_loss'][i] / max(kpi['val_loss'])) + kpi['val_accuracy'][i])/2
        score = (ts+vs) - abs(ts-vs)
        scores.append(score)
      df2 = df.drop(kpi_list, axis='columns')
      df2['quality'] = scores
      return df2

    def BuildCombos(self, params, len_dataset, n_features, dominant_class):
      '''puts all possible gridsearch combinations in a dataframe'''
      n = list(params.keys())
      lst1 = []
      lst2 = []
      lst3 = []
      lst4 = []
      lst5 = []
      lst6 = []
      lst7 = []
      lst8 = []
      lst9 = []
      lst10 = []
      lst11 = []
      lst12 = []
      for i in range(len(params[n[0]])):
        var1 = params[n[0]][i]
        for i in range(len(params[n[1]])):
          var2 = params[n[1]][i]
          for i in range(len(params[n[2]])):
            var3 = params[n[2]][i]
            for i in range(len(params[n[3]])):
              var4 = params[n[3]][i]
              for i in range(len(params[n[4]])):
                var5 = params[n[4]][i]
                for i in range(len(params[n[5]])):
                  var6 = params[n[5]][i]
                  for i in range(len(params[n[6]])):
                    var7 = params[n[6]][i]
                    for i in range(len(params[n[7]])):
                      var8 = params[n[7]][i]
                      for i in range(len(params[n[8]])):
                        var9 = params[n[8]][i]
                        for i in range(len(params[n[9]])):
                          var10 = params[n[9]][i]
                          for i in range(len(params[n[10]])):
                            var11 = params[n[10]][i]
                            for i in range(len(params[n[11]])):
                              var12 = params[n[11]][i]
                              lst1.append(var1)
                              lst2.append(var2)
                              lst3.append(var3)
                              lst4.append(var4)
                              lst5.append(var5)
                              lst6.append(var6)
                              lst7.append(var7)
                              lst8.append(var8)
                              lst9.append(var9)
                              lst10.append(var10)
                              lst11.append(var11)
                              lst12.append(var12)
      df = pd.DataFrame(lst1)
      df.columns = [n[0]]
      df[n[1]] = lst2
      df[n[2]] = lst3
      df[n[3]] = lst4
      df[n[4]] = lst5
      df[n[5]] = lst6
      df[n[6]] = lst7
      df[n[7]] = lst8
      df[n[8]] = lst9
      df[n[9]] = lst10
      df[n[10]] = lst11
      df[n[11]] = lst12
      df['len_dataset'] = [len_dataset] * len(df)
      df['n_features'] = [n_features] * len(df)
      df['dominant_class'] = [dominant_class] * len(df)
      return df


class Models:
    
    def BuildML(self, net_type):
      wr = Wrappers()
      dh = DataHelper()
      ml = MachineLearning()
      db = DataBuilder()
      ds = Datasets()
      if net_type == 'RNN':
        df = ds.load_rnn()
        df2 = db.ResultsNLP(df)
      vanilla, grid, X, Xval, y, yval = wr.Vanilla(df2, 'quality', 'regression')
      Xs = dh.ScaleData('standard', X)
      model = ml.Optimize(vanilla, grid, Xs, y)
      return model


    def BuildRNN(self, model, vocab_dim, X, y, quiet=True): 
      db = DataBuilder()
      dh = DataHelper()
      dl = DeepLearning()
      al = Algorithms()
      gr = Grids()
      len_dataset = X.shape[0]
      n_features = X.shape[1]
      dominant_class = al.GetDC(y)
      grid = gr.GridRNN(X, y, vocab_dim)
      combos = db.ResultsNLP(db.BuildCombos(grid, len_dataset, n_features, dominant_class), gridding=True)
      c2 = dh.ScaleData('standard', combos)
      combos['quality'] = model.predict(c2)
      best = combos.loc[combos['quality'] == max(combos['quality'])].iloc[0]
      res = dict(best)
      if res['relu'] == 1:
        activation = 'relu'
      if res['tanh'] == 1:
        activation = 'tanh'
      if res['L1'] == 1:
        regularizer = 'L1' 
      if res['L2'] == 1:
        regularizer = 'L2' 
      if res['None'] == 1:
        regularizer = 'None' 
      if res['adam'] == 1:
        optimizer = 'adam'
      if res['rmsprop'] == 1:
        optimizer = 'rmsprop'
      if res['sgd'] == 1:
        optimizer = 'sgd'
      if res['GRU'] == 1:
        method = 'GRU'
      if res['LSTM'] == 1:
        method = 'LSTM' 
      dm = dl.RNN(int(res['output_dim']), int(res['embedding']), int(res['nodes']), activation, regularizer, 
                  bool(res['stacking']), bool(res['dropout']), optimizer, method, bool(res['bidirectional']))
      epochs = int(res['epochs'])
      bs = int(res['batch_size'])
      try:
          history = dm.fit(X, y, epochs=epochs, batch_size=bs, validation_split=0.2)
      except:
          history = dm.fit(X, y.reshape(-1), epochs=epochs, batch_size=bs, validation_split=0.2)
      if quiet == False:
        print(res)
        print(Evaluater().ViewLoss(history))
        try:
          print(Evaluater().ViewAccuracy(history))
        except:
          pass
      return dm


class Grids:
    
    def GridRNN(self, X, y, v):
        al = Algorithms()
        the_grid = {'output_dim': [y.shape[1]], 'embedding': [v], 
                    'nodes': al.Nodes(X, generation=False), 'activation': ['relu', 'tanh'], 
                    'regularizer': ['L1', None, 'L2'], 'stacking': [True, False], 
                    'dropout': [True, False], 'optimizer': ['adam', 'rmsprop', 'sgd'], 'method': ['LSTM', 'GRU'], 
                    'bidirectional': [True, False], 'epochs': al.Epochs(False), 
                    'batch_size': al.BatchSize(X)}
        return the_grid

    def GridNN(self, X, y, task):
        al = Algorithms()
        o = y.shape[1]
        if task == 'classification':
            if o == 2:
                lf = 'binary_crossentropy'
            else:
                lf = 'categorical_crossentropy'
            the_grid = {'output_dim': [o], 'nodes': al.Nodes(X), 'activation': ['relu', 'tanh'], 'regularizer': [None, 'L1', 'L2'], 'stacking': [False, True], 
           'dropout': [False, True], 'nlayers': list(range(3, 8)), 'closer': [True, False], 'loss_func': [lf], 'optimizer': ['adam', 'rmsprop', 'sgd']}
        if task == 'regression':
            the_grid = {'nodes': al.Nodes(X), 'activation': ['relu', 'tanh'], 'regularizer': [None, 'L1', 'L2'], 'stacking': [False, True], 'dropout': [False, True], 
           'nlayers': list(range(3, 8)), 'closer': [False, True], 'loss_func': ['MSE', 'MAE'], 'optimizer': ['rmsprop', 'adam', 'sgd'], 'y_col': ['string']}
        return the_grid
    
    def GridCNN(self, X, y):
        al = Algorithms()
        if len(X) > 500000:
            ks = 5
            pool = 5
        else:
            ks = 3
            pool = 2
        the_grid = {'output_dim': [y.shape[1]], 'base_filters': al.Nodes(X), 'kernel_size': [ks], 'activation': ['relu', 'tanh'], 'nblocks': [2, 3, 4, 5], 
            'pool':[pool], 'dropout': [False, True], 'closer': [True, False], 'optimizer': ['adam', 'rmsprop', 'sgd'], 'metrics': ['accuracy']}
        return the_grid

class Google:
    
    def GetReport(self, keywords, span='today 5-y', geo='', quiet=True):
        '''observe a search term's popularity in the past 5 years'''
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload(keywords, cat=0, timeframe=span, geo=geo, gprop='')
        ts = pytrends.interest_over_time().drop(['isPartial'], axis='columns')
        if quiet == False:
            print(ts.plot())
        return ts
    
    def Collect(self, keyword, quiet=True):
        row = {}
        tsdf = Google().BuildTS(keyword)
        row['term'] = keyword
        current_popularity = list(tsdf[keyword][:260])[-1]
        row['current_popularity'] =  current_popularity
        row['change_3mo'] = '{}%'.format(round(((tsdf[keyword][271] - current_popularity) / current_popularity) * 100, 1))
        row['change_6mo'] = '{}%'.format(round(((tsdf[keyword][283] - current_popularity) / current_popularity) * 100, 1))
        row['change_9mo'] = '{}%'.format(round(((tsdf[keyword][295] - current_popularity) / current_popularity) * 100, 1))
        row['change_12mo'] = '{}%'.format(round(((tsdf[keyword][307] - current_popularity) / current_popularity) * 100, 1))
        row['change_24mo'] = '{}%'.format(round(((tsdf[keyword][355] - current_popularity) / current_popularity) * 100, 1))
        try:
            row['popularity_2y'] = round((((tsdf[keyword][355] - current_popularity) / current_popularity) + 1) * current_popularity)
        except:
            row['popularity_2y'] = round(tsdf[keyword][355])
        sentiment, subjectivity, sentiments = Twitter().AnalyzeTwitter(keyword)
        row['sentiment'] = round(sentiment, 2)
        row['subjectivity'] = round(subjectivity, 2)
        row['sentiments_std'] = round(np.std(sentiments), 2)
        if quiet == True:
            return row
        else:
            return tsdf, row
    
    def CollectLoop(self, terms_list):
        '''tells us how popularity for a given list of search terms are expected to change'''
        df = pd.DataFrame(Google().Collect(terms_list[0]), index=[0])
        for term in terms_list[1:]:
            temp = pd.DataFrame(Google().Collect(term), index=[0])
            df = pd.concat([df, temp])
        return df.reset_index().drop(['index'], axis='columns')
    
    def PlotOne(self, keyword):
        '''the output a user gets when looking at one term'''
        ts, results = Google().Collect(keyword, quiet=False)
        subj = results['subjectivity']
        obj = 1 - subj
        X = ['%subjective', '%objective']
        y = [subj, obj]
        X2 = ['sentiment']
        y2 = results['sentiment']
        if results['popularity_2y'] > results['current_popularity']:
            future = 'increase'
        else:
            future = 'decrease'
        fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = results['sentiment'],
        mode = "gauge+number",
        title = {'text': "Sentiment of '{}' based on tweets".format(keyword)},
        gauge = {'axis': {'range': [-1, 1]},
                 'steps' : [
                     {'range': [-1, 0], 'color': "red"},
                     {'range': [0, 1], 'color': "lightgreen"}]}))
        fig.show()
        fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = results['subjectivity'],
        mode = "gauge+number",
        title = {'text': "Subjectivity of '{}' based on tweets".format(keyword)},
        gauge = {'axis': {'range': [0, 1]},
                 'steps' : [
                     {'range': [0, 0.5], 'color': "yellow"},
                     {'range': [0.5, 1], 'color': "blue"}]}))
        fig.show()
        fig = px.line(ts, x='index', y=keyword, range_y=[0, 100])
        fig.show()
        
    def PlotMany(self, keywords):
        df = Google().CollectLoop(keywords)
        fig = px.bar(df, x='term', y='current_popularity', color='sentiment', range_y=[0, 100])
        fig.show()
        for i in range(len(keywords)):
            ser = Google().Collect(keywords[i], quiet=False)[0]
            fig = px.line(ser, x='index', y=keywords[i], range_y=[0, 100])
            fig.show()
            
    def GetPeaks(self, ser):
        kw = list(ser.columns)[0]
        varience = 0
        for i in range(len(ser)):
            varience += abs(np.mean(ser)[0] - ser.iloc[i][0])
        delta = abs(np.mean(ser.iloc[235:])[0] - np.mean(ser.iloc[:27])[0])
        si = varience/delta
        x = np.array(list(Google().GetReport([kw])[kw]))
        peaks, _ = find_peaks(x, prominence=10, distance=52)
        return peaks, si
        
    def CheckSeasonality(self, ser, quiet=True):
        peaks, si = Google().GetPeaks(ser)
        n_peaks = len(peaks)
        if quiet == False:
            print(peaks, si)
        if si > 250:
            if n_peaks < 3:
                return False
            else:
                return True
        else:
            if n_peaks > 4:
                return True
            else:
                if Google().CovidCheck(peaks) == False:
                    return False
                else:
                    p1 = len(Google().GetPeaks(ser.loc[:np.datetime64('2020-03-08')])[0])
                    p2 = len(Google().GetPeaks(ser.loc[np.datetime64('2020-07-26'):])[0])
                    return p1+p2 > 4
        
    def BuildTS(self, keyword):
        ser = Google().GetReport([keyword])
        s = Google().CheckSeasonality(ser)
        if s == True:
            my_order = (2,1,2) #probably wrong, also needs to be programatic
            my_seasonal_order = (2, 1, 2, 52) #probably wrong, also needs to be programatic
            model = SARIMAX(ser, order=my_order, seasonal_order=my_seasonal_order).fit()
            pred = model.predict(start=len(ser), end=356)
            ser_ = pd.DataFrame(ser)
            pred_ = pd.DataFrame(pred)
            pred_.columns = [keyword]
            ser_.columns = [keyword]
            return pd.concat([ser_, pred_]).reset_index()
        if s == False:
            model = ar_model.AutoReg(ser, lags=4).fit()
            pred = model.predict(start=len(ser), end=356)
            ser_ = pd.DataFrame(ser)
            pred_ = pd.DataFrame(pred)
            pred_.columns = [keyword]
            ser_.columns = [keyword]
            return pd.concat([ser_, pred_]).reset_index()
        
    def PredictSearches(self, to_predict):
        if type(to_predict) == str:
            return Google().PlotOne(to_predict)
        if type(to_predict) == list:
            if len(to_predict) == 1:
                return Google().PlotOne(to_predict[0])
            else:
                return Google().PlotMany(to_predict)
            
    def CovidCheck(self, peaks):
        today = datetime.date.today().strftime("%Y-%m-%d")
        delta = np.datetime64(today) - np.datetime64('2021-03-28')
        delta = int(delta.astype(int)/7)
        peaks = np.array(peaks)
        spike = np.array(list(range(205-delta, 226-delta)))
        affected = np.intersect1d(spike, peaks)
        regular = 0
        try:
            for peak in affected:
                if affected-52 in list(peaks):
                    regular += 1
        except:
            return False
        return len(affected)!=0 and regular==0
    
    
class Twitter:
    
    def AnalyzeTwitter(self, keyword):
        '''find the average sentimental value and subjectivity of a given search term'''
        c1 = 'aHXduTrDkva3ItY52tUtYVPvA'
        c2 = 'Qs6d4oNT3zXxDqOhita7IG07CfAJGceoqIs1sGuA4OURlbLP6d'
        a1 = '1181578611171762177-sGQaj7E9fpWi2aEB3MfWL4nTRovXYk'
        a2 = 'wa77yBJZJSOKOAzdaJYDruc9U1HrGhzyDhWgKvSQpm2hv'
        auth = tweepy.OAuthHandler(c1, c2)
        auth.set_access_token(a1, a2)
        api = tweepy.API(auth)
        topic = api.search(keyword)
        sent = 0
        sub = 0
        sents = []
        for i in range(len(topic)):
            tweet = topic[i]._json['text'].replace('@', '')
            blob = TextBlob(tweet)
            sents.append(blob.sentiment[0])
            sent += blob.sentiment[0]/len(topic)
            sub += blob.sentiment[1]/len(topic)
        return sent, sub, sents