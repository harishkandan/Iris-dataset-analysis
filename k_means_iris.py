#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:17:37 2017

@author: harish
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

iris_data = load_iris()
lr = LogisticRegression()
class Iris:
    
    def __init__(self):
        '''
        Initiates the Iris object with four class variables
        X_train, X_test, y_train and y_test
        
        '''
        self.X_train, \
        self.X_test, \
        self.y_train, \
        self.y_test = train_test_split(iris_data.data,
                                       iris_data.target,
                                       test_size = 0.3,
                                       random_state = 42)
        
    def k_means_labels(self):
        '''
        Creates two class variables containing cluster labels obtained 
        from k-means clustering for train data and test data respectively
        
        '''
        km = KMeans(n_clusters = 3, 
                    init = 'k-means++').fit(self.X_train)
        self.train_labels = km.labels_
        self.test_labels = km.predict(self.X_test)
        
    def predict(self, model = lr, type = None):
        '''
        Fits the data in the given model (Default model is Logistic regression),
        makes a prediction and returns the accuracy value for the model
        Parameters
        ----------
        model: Default - LogisticRegression
               Classification is performed on the given model
        
        type: string value
              Default- None
              Mentions wether to use just cluster labels, input features or both
              for training the model
        Return
        ------
        accuracy score for that model
        '''
        if type == None:
            model.fit(self.X_train, self.y_train)
            return accuracy_score(model.predict(self.X_test), self.y_test)
        
        elif type == 'labels_only':
            model.fit(self.train_labels.reshape(-1, 1), self.y_train)
            return accuracy_score(model.predict(self.test_labels.reshape(-1, 1)), self.y_test)
        
        elif type == 'data_labels':
            train_df = pd.DataFrame(self.X_train)
            test_df = pd.DataFrame(self.X_test)
            train_df['labels'] = self.train_labels
            test_df['labels'] = self.test_labels
            model.fit(train_df, self.y_train)
            return accuracy_score(model.predict(test_df), self.y_test)