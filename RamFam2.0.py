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
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, XGBRegressor
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os
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
import itertools
import matplotlib
import seaborn as sns
from matplotlib.pylab import rcParams
plt.style.use('fivethirtyeight')

class MachineLearning:
    
    def CompareModels(self, data, target, task): 
        '''returns out the box accuracy of sklearn models'''
        X = data.drop([target], axis='columns')
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        n = len(np.unique(y))
        if task == 'classification':
            if n == 2:
                methods = [KNeighborsClassifier(), GaussianNB(), DecisionTreeClassifier(), RandomForestClassifier(),
                           AdaBoostClassifier(), GradientBoostingClassifier(), XGBClassifier(), LogisticRegression(),
                           SVC()]
                strs = ['KNN', 'NB', 'DT', 'RF', 'AB', 'GB', 'XGB', 'Log', 'SVM']
            else:
                methods = [KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(),
                           AdaBoostClassifier(), GradientBoostingClassifier(), XGBClassifier(), SVC()]
                strs = ['KNN', 'DT', 'RF', 'AB', 'GB', 'XGB', 'SVM']
        if task == 'regression':
            methods = [LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor(),
                       AdaBoostRegressor(), GradientBoostingRegressor(), XGBRegressor(), SVR()]
            strs = ['Lin', 'KNN', 'DT', 'RF', 'AB', 'GB', 'XGB', 'SVM']
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
    
    def Optimize(self, model, parameters, data, target, metric='accuracy'):
        '''facilitates a gridsearch on any sklearn model'''
        X = data.drop([target], axis='columns')
        y = data[target]
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
            return GridSearchCV(model, parameters, cv=5, scoring=metric, n_jobs=-1, verbose=2).fit(X_train, y_train).best_estimator_
        except:
            return GridSearchCV(model, parameters, cv=5, n_jobs=-1, verbose=2).fit(X_train, y_train).best_estimator_
           
    def SeeModel(self, model, parameters, data, target, task, ensemble=False):
        '''runs a gridsearch on a sklearn model and evaluates model preformance'''
        dh = DataHelper()
        ev = Evaluater()
        ml = MachineLearning()
        train, test = dh.HoldOut(data)
        mod = ml.Optimize(XGBClassifier(), parameters, train, target)
        X = train.drop([target], axis='columns')
        y = train[target]
        score = ev.ScoreModel(mod, X, y)
        Xval = test.drop([target], axis='columns')
        yval = test[target]
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
    
    def ClusterIt(self, data, clusters, labeled=False):
        '''clusters a given dataset with KMeans'''
        k = KMeans(n_clusters=clusters).fit(data)
        pred = k.predict(data)
        centers = k.cluster_centers_
        X2 = pd.DataFrame(data)
        y = pd.DataFrame(pred)
        y.columns = ['cluster']
        results = pd.concat([X2, y], axis='columns')
        plt.scatter(data[0], data[1], s=10, c=pred);
        plt.scatter(centers[:, 0], centers[:, 1], s=70, c='black');
        plt.show()
        return results


class DeepLearning:
    
    def Powers(self, n):
        '''help the model building functions determine nlayers'''
        k = int(log(n, 2))
        return k
    
    def DeepTabularRegression(self, nodes, activation, regularizer, stacking, dropout, nlayers, closer, loss, optimizer, y_col):
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
    
    def RNN(self, output_dim, embedding, nodes, activation, regularizer, stacking, dropout, optimizer, method, bidirectional):
        '''builds a neural network with LSTM or GRU layer(s)'''
        dl = DeepLearning()
        if output_dim > 16:
          pen = output_dim*2
        else:
          pen = 16
        if output_dim == 2:
          oa = 'sigmoid'
          loss = 'binary_crossentropy'
        else:
          oa = 'softmax'
          loss = 'sparse_categorical_crossentropy'  
        model = models.Sequential()
        model.add(layers.Embedding(embedding, nodes))
        if method == 'LSTM':
          if regularizer == None:
            if bidirectional == False:
              model.add(layers.LSTM(nodes, activation=activation, return_sequences=stacking))
            else:
              model.add(layers.Bidirectional(layers.LSTM(nodes)))
          if regularizer == 'L1':
            if bidirectional == False:
              model.add(layers.LSTM(nodes, activation=activation, kernel_regularizer=L1(0.005), return_sequences=stacking))
            else:
              model.add(layers.Bidirectional(layers.LSTM(nodes)))
          if regularizer == 'L2':
            if bidirectional == False:
              model.add(layers.LSTM(nodes, activation=activation, kernel_regularizer=L2(0.005), return_sequences=stacking))
            else:
              model.add(layers.Bidirectional(layers.LSTM(nodes)))
        if method == 'GRU':
          if regularizer == None:
            if bidirectional == False:
              model.add(layers.GRU(nodes, activation=activation, return_sequences=stacking))
            else:
              model.add(layers.Bidirectional(layers.GRU(nodes)))
          if regularizer == 'L1':
            if bidirectional == False:
              model.add(layers.GRU(nodes, activation=activation, kernel_regularizer=L1(0.005), return_sequences=stacking))
            else:
              model.add(layers.Bidirectional(layers.GRU(nodes)))
          if regularizer == 'L2':
            if bidirectional == False:
              model.add(layers.GRU(nodes, activation=activation, kernel_regularizer=L2(0.005), return_sequences=stacking))
            else:
              model.add(layers.Bidirectional(layers.GRU(nodes)))
        if dropout == True:
          model.add(layers.Dropout(0.5))
        dl = DeepLearning()
        if stacking == True:
          nodes = nodes//2
          if method == 'LSTM':
            if bidirectional == False:
              model.add(layers.LSTM(nodes, activation=activation))
            else:
              model.add(layers.Bidirectional(layers.LSTM(nodes)))
          if method == 'GRU':
            if bidirectional == False:
              model.add(layers.GRU(nodes, activation=activation))
            else:
              model.add(layers.Bidirectional(layers.GRU(nodes)))
        p = dl.Powers(nodes)
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
    
    def CNN(self, output_dim, filters, kernel_size, activation, nblocks, pool, dropout, closer, optimizer, metrics):
        if output_dim == 2:
            oa = 'sigmoid'
            loss = 'binary_crossentropy'
        else:
            oa = 'softmax'
            loss = 'categorical_crossentropy'
        model = models.Sequential()
        model.add(layers.Conv2D(filters, (kernel_size, kernel_size), activation=activation))
        model.add(layers.MaxPooling2D(pool, pool))
        if nblocks > 1:
            for i in range(nblocks-1):
                model.add(layers.Conv2D(filters*2, (kernel_size, kernel_size), activation=activation))
                model.add(layers.MaxPooling2D(pool, pool))
        model.add(layers.Flatten())
        if dropout == True:
            model.add(layers.Dropout(0.5))
        model.add(layers.Dense(filters*2, activation=activation))
        if filters/2 >= output_dim*2:
            model.add(layers.Dense(filters/2, activation=activation))
        if closer == True:
            model.add(layers.Dense(output_dim*2, activation=activation))
        else:
            if filters/2 < output_dim*2:
                model.add(layers.Dense(filters/2, activation=activation))
        model.add(layers.Dense(output_dim, activation=oa))  
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return model
    
    def TestDL(self, params, func, task, X, y, X_val=None, y_val=None, batch_size=64, epochs=50):
        '''wraps the keras wrapper functions and GridSearchCV into a simple one liner, one line plus the parameter grid'''
        early_stopping = [EarlyStopping(patience=10), ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5')]
        if task == 'classification':
            k = KerasClassifier(func)
        if task == 'regression':
            k = KerasRegressor(func)
        grid = GridSearchCV(k, params, cv=3)
        if type(X_val) != np.ndarray:
            grid.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=early_stopping)
        else:
            grid.fit(X, y, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=early_stopping)
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
        t = keras.Tokenizer()
        t.fit_on_texts(text)
        tokens = t.texts_to_sequences(text)
        tokens2 = keras.pad_sequences(tokens, maxlen=pad)
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
      y = dl.MulticlassOutput(labels)
      return tokens2, y

    def ModelReadyText2(self, df, Xcol, ycol, num_words):
      '''converts text into one-hot-encoded vectors'''
      dh = DataHelper()
      text = list(df[Xcol])
      t = Tokenizer(num_words=num_words)
      t.fit_on_texts(text)
      oh = t.texts_to_matrix(text)
      y = dh.MulticlassOutput(df, ycol)
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
    
    def ScaleData(self, strategy, data, y_var=None, dim=None):
        '''Scales data via minmax, standard, mean, or PCA scaling'''
        try:
            X = data.drop([y_var], axis='columns')
        except:
            X = data
        if strategy == 'minmax':
            return MinMaxScaler().fit(X).transform(X)
        if strategy == 'standard':
            return StandardScaler().fit(X).transform(X)
        if strategy == 'mean':
            for col in X.columns:
                X[col] = (X[col] - min(X[col]))/ (max(X[col]) - min(X[col]))
            return X
        if strategy == 'pca':
            return PCA(n_components=dim).fit_transform(X)
    
    def VifIt(self, X):
        '''returns VIF scores to help prevent multicolinearity'''
        vif = pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)
        return vif
    
    def SmoteIt(self):
        return None
    
    def MakeDirs(self, train_dir, test_dir, classes, data_type):
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
                
         
class Evaluater:
    
    def ScoreModel(self, model, X, y):
        '''returns accuracies for any sklearn model'''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        algo = model.fit(X_train, y_train)
        return 'Training: {}, Validation: {}'.format(algo.score(X_train, y_train), algo.score(X_test, y_test))
    
    def BuildConfusion(self, fitted_model, Xval, yval, cmap='plasma'):
        '''returns a Confusion Matrix given a pretrained sklearn model'''
        try:
            return plot_confusion_matrix(fitted_model, Xval, yval, cmap=cmap)
        except:
            return 'Please pass a classifier'
        
    def BuildConfusionDL(self, model, X, y, normalize='true', cmap='plasma'):
      '''displays a confusion matrix to evaluate a deep learning classifier'''
      yhat = []
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
            return 'Please pass a tree class'
    
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
    
    def BinaryCBA(self, trained_model, X, y, value, discount):
        '''uses a trained binary classification model to do cost benefit analysis'''
        pred = trained_model.predict(X)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        tn = tn * (value)
        fp = fp * (0-discount)
        fn = fn * (0-value)
        tp = tp * (value-discount)
        results = tn + fp + fn +tp
        
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
    
    def ViewAccuracy(self, history, epochs):
      '''plots a model's accuracy throughout training'''
      plt.plot(range(epochs), history.history['accuracy'], label='train');
      plt.plot(range(epochs), history.history['val_accuracy'], label='val');
      plt.legend(loc='best')
      plt.xlabel('epochs')
      plt.ylabel('accuracy')
      return None


    def ViewLoss(self, history, epochs):
      '''plots a model's loss throughout training'''
      plt.plot(range(epochs), history.history['loss'], label='train');
      plt.plot(range(epochs), history.history['val_loss'], label='val');
      plt.legend(loc='best')
      plt.xlabel('epochs')
      plt.ylabel('loss')
      return None
  
    def AUC(self, model, Xval, yval):
      '''displays AUC to evaluate a classification model'''
      pred = model.predict(Xval)
      fpr, tpr, threshold = roc_curve(yval, pred)
      return auc(fpr, tpr)

