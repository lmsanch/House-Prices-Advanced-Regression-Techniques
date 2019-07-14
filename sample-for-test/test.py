
# import necessary python packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
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

# print("training data shape")
# print(train.shape)

# print("testing data shape")
# print(test.shape)

# print(train.head(10))
# print(test.head(5))

# saving training and testing ID
train_ID = train['Id']
test_ID = test['Id']

fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.xlabel("living area square feet", fontsize = 14)
plt.ylabel("house sale price", fontsize = 14)
# plt.show()


############# Data Preprocessing #############
### let's check for the outliers first

drop_index = train[(train['GrLivArea'] > 4000) & 
                (train['SalePrice']<300000)].index
# we can safely delete these huge outliers mention in drop_index
train.drop('Id', axis = 1, inplace=True)
test.drop('Id', axis = 1, inplace=True)
train = train.drop(drop_index)


# fig, ax = plt.subplots()
# ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
# plt.xlabel("living area square feet", fontsize = 14)
# plt.ylabel("house sale price", fontsize = 14)
# plt.show()

# print("Train set size:", train.shape)
# print("Test set size:", test.shape)

### let's check the target variable distribution

# df = pd.concat([train.SalePrice, np.log(train.SalePrice + 1).rename('LogSalePrice')], axis=1, names=['SalePrice', 'LogSalePrice'])
# print(df.head())
# sns.distplot(train['SalePrice'], fit = norm)
(mu, sigma) = norm.fit(train['SalePrice'])
# print("mu: ", mu)
# print('sigma', sigma)

# plt.ylabel('Frequency')
# plt.title('SalePrice distributed')

#  get also the QQ-plot
#  fig = plt.figure()
# res = stats.probplot(train['SalePrice'], plot = plt)
# plt.show()

# # we need to transform this variable and make it more normally distributed.
# train["SalePrice"] = np.log1p(train["SalePrice"])
# sns.distplot(train['SalePrice'], fit=norm)
# plt.ylabel("Frequency")
# plt.title("Sale Price Distribution")

# fig = plt.figure()
# res = stats.probplot(train['SalePrice'], plot=plt)
# plt.show()

### missing data handling
ntrain = train.shape[0]
ntest = train.shape[0]
y_train = train.SalePrice.values

train.drop(['SalePrice'], axis = 1, inplace = True)
all_data = pd.concat((train, test)).reset_index(drop=True)

# print("concatenated data: ", all_data.shape)

# # let's check missing data
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = (all_data_na.drop(all_data_na[all_data_na == 0].index).
                    sort_values(ascending=False)[:30])

# missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
# print("list of total missing data (in percentage)")
# print(missing_data)



# nulls = np.sum(all_data.isnull())
# nullcols = nulls.loc[(nulls != 0)]
# dtypes = all_data.dtypes
# dtypes2 = dtypes.loc[(nulls != 0)]
# info = pd.concat([nullcols, dtypes2], axis=1).sort_values(by=0, ascending=False)
# print(info)
# print("There are", len(nullcols), "columns with missing values")



# f, ax = plt.subplots(figsize=(15, 12))
# plt.xticks(rotation='90')
# sns.barplot(x=all_data_na.index, y=all_data_na)
# plt.xlabel('Features', fontsize=15)
# plt.ylabel('Percent of missing values', fontsize=15)
# plt.title('Percent missing data by feature', fontsize=15)
# plt.show()

##data correction

#Correlation map to see how features are correlated with SalePrice
# corrmat = train.corr()
# plt.subplots(figsize=(12,9))
# sns.heatmap(corrmat, vmax=0.9, square=True)
# plt.show()


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
# print('\n\ncheck again for the missing values')
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = (all_data_na.drop(all_data_na[all_data_na == 0].index).
                    sort_values(ascending=False)[:30])

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
# print(missing_data.head(30))

# all_data['Functional'] = all_data['Functional'].fillna('Typ')
# all_data['Electrical'] = all_data['Electrical'].fillna("SBrkr")
# all_data['KitchenQual'] = all_data['KitchenQual'].fillna("TA")

# all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
# all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

# all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

# pd.set_option('max_columns', None)
# # print(all_data[all_data['PoolArea'] > 0 & all_data['PoolQC'].isnull()])
# all_data.loc[2418, 'PoolQC'] = 'Fa'
# all_data.loc[2501, 'PoolQC'] = 'Gd'
# all_data.loc[2597, 'PoolQC'] = 'Fa'
# pd.set_option('max_columns', None)
# all_data[(all_data['GarageType'] == 'Detchd') & all_data['GarageYrBlt'].isnull()]
# basement_columns = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
#                    'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
#                    'TotalBsmtSF']

# tempdf = all_data[basement_columns]
# tempdfnulls = tempdf[tempdf.isnull().any(axis=1)]

# #now select just the rows that have less then 5 NA's, 
# # meaning there is incongruency in the row.
# tempdfnulls[(tempdfnulls.isnull()).sum(axis=1) < 5]

# all_data.loc[332, 'BsmtFinType2'] = 'ALQ' #since smaller than SF1
# all_data.loc[947, 'BsmtExposure'] = 'No' 
# all_data.loc[1485, 'BsmtExposure'] = 'No'
# all_data.loc[2038, 'BsmtCond'] = 'TA'
# all_data.loc[2183, 'BsmtCond'] = 'TA'
# all_data.loc[2215, 'BsmtQual'] = 'Po' #v small basement so let's do Poor.
# all_data.loc[2216, 'BsmtQual'] = 'Fa' #similar but a bit bigger.
# all_data.loc[2346, 'BsmtExposure'] = 'No' #unfinished bsmt so prob not.
# all_data.loc[2522, 'BsmtCond'] = 'Gd' #cause ALQ for bsmtfintype1

# subclass_group = all_data.groupby('MSSubClass')
# Zoning_modes = subclass_group['MSZoning'].apply(lambda x : x.mode()[0])
# # print(Zoning_modes)

# all_data['MSZoning'] = all_data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
# objects = []
# for i in all_data.columns:
#     if all_data[i].dtype == object:
#         objects.append(i)

# all_data.update(all_data[objects].fillna('None'))

# nulls = np.sum(all_data.isnull())
# nullcols = nulls.loc[(nulls != 0)]
# dtypes = all_data.dtypes
# dtypes2 = dtypes.loc[(nulls != 0)]
# info = pd.concat([nullcols, dtypes2], axis=1).sort_values(by=0, ascending=False)
# # print(info)
# # print("There are", len(nullcols), "columns with missing values")
# neighborhood_group = all_data.groupby('Neighborhood')
# lot_medians = neighborhood_group['LotFrontage'].median()
# # print(lot_medians)

# all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
# pd.set_option('max_columns', None)
# all_data[(all_data['GarageYrBlt'].isnull()) & all_data['GarageArea'] > 0]
# pd.set_option('max_columns', None)
# print(all_data[(all_data['MasVnrArea'].isnull())])

# print(all_data.describe())

# print(all_data[all_data['GarageYrBlt'] == 2207])



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

# all_data = pd.get_dummies(all_data)
print(all_data.shape)

#getting the new train and test sets.
train = all_data[:ntrain]
test = all_data[ntrain:]

# cross validation strategy
#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

# all_data.loc[2590, 'GarageYrBlt'] = 2007
# factors = ['MSSubClass', 'MoSold']
# factors = ['MSSubClass']
 


# for i in factors:
#     all_data.update(all_data[i].astype('str'))

# from scipy.stats import skew

# numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
# numerics2 = []
# for i in all_data.columns:
#     if all_data[i].dtype in numeric_dtypes: 
#         numerics2.append(i)

# skew_features = all_data[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)
# skews = pd.DataFrame({'skew':skew_features})
# # print(skews)

# objects3 = []
# for i in all_data.columns:
#     if all_data[i].dtype == object:
#         objects3.append(i)

# # print("Training Set incomplete cases")

# sums_features = all_data[objects3].apply(lambda x: len(np.unique(x)))
# sums_features.sort_values(ascending=False)
# # print(sums_features)

# # print(all_data['Street'].value_counts())
# # print('-----')
# # print(all_data['Utilities'].value_counts())
# # print('-----')
# # print(all_data['CentralAir'].value_counts())
# # print('-----')
# # print(all_data['PavedDrive'].value_counts())

# #features = features.drop(['Utilities'], axis=1)
# all_data = all_data.drop(['Utilities', 'Street'], axis=1)

# all_data['Total_sqr_footage'] = (all_data['BsmtFinSF1'] + all_data['BsmtFinSF2'] +
#                                  all_data['1stFlrSF'] + all_data['2ndFlrSF'])

# all_data['Total_Bathrooms'] = (all_data['FullBath'] + (0.5*all_data['HalfBath']) + 
#                                all_data['BsmtFullBath'] + (0.5*all_data['BsmtHalfBath']))

# all_data['Total_porch_sf'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] +
#                               all_data['EnclosedPorch'] + all_data['ScreenPorch'] +
#                              all_data['WoodDeckSF'])


# #simplified all_data
# all_data['haspool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
# all_data['has2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
# all_data['hasgarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
# all_data['hasbsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
# all_data['hasfireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

# # print(all_data.shape)

# final_features = pd.get_dummies(all_data).reset_index(drop=True)
# # print(final_features.shape)


#Modelling
from sklearn.linear_model import Lasso,  BayesianRidge, LassoLarsIC
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

# ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
# GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
#                                    max_depth=4, max_features='sqrt',
#                                    min_samples_leaf=15, min_samples_split=10, 
#                                    loss='huber', random_state =5)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

# model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
#                               learning_rate=0.05, n_estimators=720,
#                               max_bin = 55, bagging_fraction = 0.8,
#                               bagging_freq = 5, feature_fraction = 0.2319,
#                               feature_fraction_seed=9, bagging_seed=9,
#                               min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
