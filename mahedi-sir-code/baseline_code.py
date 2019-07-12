#@Author: Md Mahedi Hasan
#@Description: Baseline code for the House Price Prediction Competition


# import necessary python packages
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning
from scipy import stats 
from scipy.stats import norm, skew

# package settings
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
sns.set_style('darkgrid')

# reading data files
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("training data shape")
print(train.shape)

print("testing data shape")
print(test.shape)

#print(train.head(10))
#print(test.head(5))

# saving training and testing ID
train_ID = train['Id']
test_ID = test['Id']

train.drop('Id', axis = 1, inplace=True)
test.drop('Id', axis = 1, inplace=True)

############# Data Preprocessing #############
### let's check for the outliers first
drop_index = train[(train['GrLivArea'] > 4000) & 
                (train['SalePrice']<300000)].index

# we can safely delete these huge outliers mention in drop_index
train = train.drop(drop_index)

fig, ax = plt.subplots()
#ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
#plt.xlabel("living area square feet", fontsize = 14)
#plt.ylabel("house sale price", fontsize = 14)
#plt.show()


### let's check the target variable distribution
#sns.distplot(train['SalePrice'], fit = norm)
(mu, sigma) = norm.fit(train['SalePrice'])
#print("mu: ", mu)
#print('sigma', sigma)

#plt.ylabel('Frequency')
#plt.title('SalePrice distributed')

# get also the QQ-plot
#fig = plt.figure()
#res = stats.probplot(train['SalePrice'], plot = plt)
#plt.show()


# we need to transform this variable and make it more normally distributed.
train["SalePrice"] = np.log1p(train["SalePrice"])
#sns.distplot(train['SalePrice'], fit=norm)
#plt.ylabel("Frequency")
#plt.title("Sale Price Distribution")

#fig = plt.figure()
#res = stats.probplot(train['SalePrice'], plot=plt)
#plt.show()


### missing data handling
ntrain = train.shape[0]
ntest = train.shape[0]
y_train = train.SalePrice.values

train.drop(['SalePrice'], axis = 1, inplace = True)
all_data = pd.concat((train, test)).reset_index(drop=True)

print("concatenated data: ", all_data.shape)

# let's check missing data
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = (all_data_na.drop(all_data_na[all_data_na == 0].index).
                    sort_values(ascending=False)[:30])

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
print("list of total missing data (in percentage)")
print(missing_data)


### imputing missing values
# PoolQC --> NA means missing houses have no Pool in general so "None"
all_data['PoolQC'] = all_data['PoolQC'].fillna("None")

# MiscFeature --> NA means no misc. features so "No"
all_data['MiscFeature'] = all_data['MiscFeature'].fillna("None")

# Alley :  NA means "no alley access"
all_data['Alley'] = all_data['Alley'].fillna("None")

# Fence: NA means "no fence"
all_data['Fence'] = all_data['Fence'].fillna("None")

# FireplaceQu: NA means "no fireplace"
all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna("None")


# GarageType, GarageFinish, GarageQual and GarageCond: NA means "None"
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna("None")

# GarageYrBlt, GarageArea and GarageCars : NA means o
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)


# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath:
# "NA" means 0 for no basement
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
            'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)

# 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'
# categorical meaning NA means 'None'
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')

# masonry veneer: 0 for area and None for category
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

# Utilites: won't help in predictive modelling
all_data['Utilities'] = all_data.drop(['Utilities'], axis = 1)


# Functional: NA means typeical
all_data['Functional'] = all_data['Functional'].fillna('Typ')

# set the most commomn string
# MSZoning: NA replace most common value of the list "RL"
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])


# Electrical: NA means SBrkr
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])


#SaleType: NA means WD
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

# KitchenQual
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])


# Exterior1st and Exterior2nd: NA means most commom string
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

# most important
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median()))


# let's check missing data
print('\n\ncheck again for the missing values')
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = (all_data_na.drop(all_data_na[all_data_na == 0].index).
                    sort_values(ascending=False)[:30])

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
print(missing_data.head(30))



# Transforming some numerical variables that are really categorical
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data', all_data.shape)
#print(all_data['Street'])

all_data = pd.get_dummies(all_data)
print(all_data.shape)

#getting the new train and test sets.
train = all_data[:ntrain]
test = all_data[ntrain:]

# cross validation strategy
#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, 
            scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

# Modelling
from sklearn.linear_model import Lasso,  BayesianRidge, LassoLarsIC
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb


# LASSO Regression
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

#score = rmsle_cv(lasso)
#print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# Kernel Ridge Regression
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# rmsle evaluation function
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# predicting
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))

print(xgb_pred)


sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = xgb_pred
sub.to_csv('submission.csv',index=False)