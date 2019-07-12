from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from scipy.stats import skew
import scipy.stats as stats
import lightgbm as lgb
import seaborn as sns
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import json
import sys
import csv
import os

# print('matplotlib: {}'.format(matplotlib.__version__))
# print('sklearn: {}'.format(sklearn.__version__))
# print('scipy: {}'.format(scipy.__version__))
# print('seaborn: {}'.format(sns.__version__))
# print('pandas: {}'.format(pd.__version__))
# print('numpy: {}'.format(np.__version__))
# print('Python: {}'.format(sys.version))

pd.set_option('display.float_format', lambda x: '%.3f' % x)
sns.set(style='white', context='notebook', palette='deep')
warnings.filterwarnings('ignore')
sns.set_style('white')

# import Dataset to play with it
train = pd.read_csv('train.csv')
test= pd.read_csv('test.csv')

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

# print(type(train))
# print(type(test))

# shape
#print(train.shape)

# shape
#print(test.shape)

# print(train.info())

#print(train['Fence'].unique())
#print(train["Fence"].value_counts())

train_id=train['Id'].copy()
test_id=test['Id'].copy()

# print(train.head(5)) 

# print(train.tail())

# print(train.sample(5))

# print(train.describe())

# print(train.isnull().sum().head(2))

# print(train.groupby('SaleType').count())

print(train.columns)