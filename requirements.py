#Import danych
import requests
import time
import os

#Cross walidacja
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

class BlockingTimeSeriesSplit_with_valid():
    def __init__(self, n_splits, train_size=13/20, validation_size=3/20):
      self.n_splits = n_splits
      self.train_size = train_size
      self.validation_size = validation_size
        
    def get_n_splits(self, X, y, groups):
      return self.n_splits
        
    def split(self, X, y=None, groups=None, train_size=0.8):
      n_samples = len(X)
      k_fold_size = math.ceil(n_samples/self.n_splits)
      indices = np.arange(n_samples)
      for i in range(0, self.n_splits):
        start_train = i * k_fold_size
        stop_train = start_train + math.ceil(self.train_size*k_fold_size)
        stop_valid = stop_train + math.ceil(self.validation_size*k_fold_size)
        stop_test = (i+1)*k_fold_size
        yield indices[start_train: stop_train], indices[stop_train: stop_valid], indices[stop_valid: stop_test]

class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
      self.n_splits = n_splits
        
    def get_n_splits(self, X, y, groups):
      return self.n_splits
        
    def split(self, X, y=None, groups=None, train_size=0.8):
      n_samples = len(X)
      k_fold_size = math.ceil(n_samples/self.n_splits)
      indices = np.arange(n_samples)
      margin = 0
      for i in range(self.n_splits):
        start = i * k_fold_size
        stop = start + k_fold_size
        mid = math.ceil(train_size * (stop - start)) + start
        yield indices[start: mid], indices[mid + margin: stop]

class Rolling_window():
      def __init__(self, n_splits, test_size):
        self.n_splits = n_splits
        self.test_size = test_size
          
      def get_n_splits(self, df, y, groups):
        return self.n_splits
          
      def split(self, df, y=None, groups=None):
        n_samples = len(df)
        test_len = int(len(df)*(self.test_size))
        train_len = int(len(df)*(1-self.test_size))
        indices = np.arange(n_samples)
        for i in range(1, self.n_splits+1):
          start = int((i-1)*int(test_len)/self.n_splits)
          stop = int(train_len+i*int(test_len)/self.n_splits)
          mid = int(train_len+(i-1)*int(test_len)/self.n_splits)
          yield indices[start: mid], indices[mid: stop]

class Time_Series():
  def __init__(self, n_splits):
    self.n_splits = n_splits
  
  def get_n_splits(self, df, y, groups):
    return self.n_splits
  
  def split(self, df, y=None, groups=None):
    n_samples = len(df)
    test_len = int(len(df)/(self.n_splits+1))
    indices = np.arange(n_samples)
    for i in range(1, self.n_splits+1):
      mid = int(i*test_len)
      stop = int((i+1)*test_len)
      yield indices[: mid], indices[mid: stop]
#Modelowanie
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import KernelCenterer
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from lightgbm import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder

#Wizualizacja
import plotly.graph_objects as go
import plotly.express as px
import plotly
from chart_studio import plotly
from plotly import express as px
import plotly.figure_factory as ff
from plotly import graph_objs as go

#Ewaluacja
from sklearn.metrics import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

#Inne
import pandas as pd
import numpy as np
import math
from collections import Counter
from itertools import cycle
import random
from datetime import datetime
#import warnings
#warnings.simplefilter('ignore')