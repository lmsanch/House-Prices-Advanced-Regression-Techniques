import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import warnings
from scipy.stats import norm, skew #for some statistics
warnings.filterwarnings('ignore')

color = sns.color_palette()
sns.set_style('darkgrid')

#load datasets
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
# train.head()

# #shape of train dataset
# train.shape
train_labels = train.pop('SalePrice') # separate labels from train dataset
data = pd.concat([train, test], keys=['train', 'test'])
# print(data.columns) # check column decorations
# print('rows:', data.shape[0], ', columns:', data.shape[1]) # count rows of total dataset
# print('rows in train dataset:', train.shape[0])
# print('rows in test dataset:', test.shape[0])

nans = pd.concat([train.isnull().sum(), train.isnull().sum() / train.shape[0], test.isnull().sum(), test.isnull().sum() / test.shape[0]], axis=1, keys=['Train', 'Percentage', 'Test', 'Percentage'])
# print(nans[nans.sum(axis=1) > 0])
# print(train_labels.describe())

###not working it's problem
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.title("Sale Price Dist")
# sns.distplot(np.log(train_labels), fit=stats.norm)
# plt.subplot(1, 2, 2)
# stats.probplot(np.log(train_labels), plot=sns.plt)
# plt.show()
# print("Skewness: %f" % train_labels.skew())
# print("Kurtosis: %f" % train_labels.kurt())

# # Skewness and Kurtosis
# skew: 1.882876
# kurt: 6.536282
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points


from subprocess import check_output
# print(check_output(["ls", "./"]).decode("utf8"))

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

##display the first five rows of the train dataset.
train.head(5)
##display the first five rows of the test dataset.
test.head(5)
# train.info()

# plt.subplots(figsize=(12,9))
# sns.distplot(train['SalePrice'], fit=stats.norm)

# Get the fitted parameters used by the function

(mu, sigma) = stats.norm.fit(train['SalePrice'])

# plot with the distribution

# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
# plt.ylabel('Frequency')

#Probablity plot
# fig = plt.figure()
# stats.probplot(train['SalePrice'], plot=plt)
# plt.show()

#we use log function which is in numpy
# train['SalePrice'] = np.log1p(train['SalePrice'])

#Check again for more normal distribution
# plt.subplots(figsize=(12,9))
# sns.distplot(train['SalePrice'], fit=stats.norm)

# # Get the fitted parameters used by the function
# (mu, sigma) = stats.norm.fit(train['SalePrice'])

# plot with the distribution
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
# plt.ylabel('Frequency')


#Probablity plot
# fig = plt.figure()
# stats.probplot(train['SalePrice'], plot=plt)
#plt.show()


 #Let's check if the data set has any missing values. 
# train.columns[train.isnull().any()]
#plot of missing value attributes
# plt.figure(figsize=(12, 6))
# sns.heatmap(train.isnull())
# plt.show()


#missing value counts in each of these columns
# Isnull = train.isnull().sum()/len(train)*100
# Isnull = Isnull[Isnull>0]
# Isnull.sort_values(inplace=True, ascending=False)
# Isnull


#Convert into dataframe
# Isnull = Isnull.to_frame()
# Isnull.columns = ['count']
# Isnull.index.names = ['Name']
# Isnull['Name'] = Isnull.index

# #plot Missing values
# plt.figure(figsize=(13, 5))
# sns.set(style='whitegrid')
# sns.barplot(x='Name', y='count', data=Isnull)
# plt.xticks(rotation = 90)
# plt.show()

#Separate variable into new dataframe from original dataframe which has only numerical values
#there is 38 numerical attribute from 81 attributes
train_corr = train.select_dtypes(include=[np.number])
# print(train_corr.shape)

#Delete Id because that is not need for corralation plot
del train_corr['Id']

#Coralation plot
corr = train_corr.corr()
# print(plt.subplots(figsize=(20,9)))
# print(sns.heatmap(corr, annot=True))
#plt.show()


top_feature = corr.index[abs(corr['SalePrice']>0.5)]
plt.subplots(figsize=(12, 8))
top_corr = train[top_feature].corr()
sns.heatmap(top_corr, annot=True)
# plt.show()

#unique value of OverallQual
train.OverallQual.unique()
sns.barplot(train.OverallQual, train.SalePrice)
#boxplot
plt.figure(figsize=(18, 8))
sns.boxplot(x=train.OverallQual, y=train.SalePrice)

col = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
sns.set(style='ticks')
sns.pairplot(train[col], size=3, kind='reg')

print("Find most important features relative to target")
corr = train.corr()
corr.sort_values(['SalePrice'], ascending=False, inplace=True)
corr.SalePrice

# PoolQC has missing value ratio is 99%+. So, there is fill by None
train['PoolQC'] = train['PoolQC'].fillna('None')

#Arround 50% missing values attributes have been fill by None
train['MiscFeature'] = train['MiscFeature'].fillna('None')
train['Alley'] = train['Alley'].fillna('None')
train['Fence'] = train['Fence'].fillna('None')
train['FireplaceQu'] = train['FireplaceQu'].fillna('None')


#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
train['LotFrontage'] = train.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

#GarageType, GarageFinish, GarageQual and GarageCond these are replacing with None
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    train[col] = train[col].fillna('None')

#GarageYrBlt, GarageArea and GarageCars these are replacing with zero
for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:
    train[col] = train[col].fillna(int(0))


#BsmtFinType2, BsmtExposure, BsmtFinType1, BsmtCond, BsmtQual these are replacing with None
for col in ('BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual'):
    train[col] = train[col].fillna('None')


#MasVnrArea : replace with zero
train['MasVnrArea'] = train['MasVnrArea'].fillna(int(0))

#MasVnrType : replace with None
train['MasVnrType'] = train['MasVnrType'].fillna('None')




#There is put mode value 
train['Electrical'] = train['Electrical'].fillna(train['Electrical']).mode()[0]

#There is no need of Utilities
train = train.drop(['Utilities'], axis=1)

#Checking there is any null value or not
plt.figure(figsize=(10, 5))
sns.heatmap(train.isnull())

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood',
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'GarageType', 'MiscFeature', 
        'SaleType', 'SaleCondition', 'Electrical', 'Heating')

from sklearn.preprocessing import LabelEncoder
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(train[c].values)) 
    train[c] = lbl.transform(list(train[c].values))

#Take targate variable into y
y = train['SalePrice']

#Delete the saleprice
del train['SalePrice']

#Take their values in X and y
X = train.values
y = y.values

# Split data into train and test formate
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)


#Linear Regression¶

#Train the model
from sklearn import linear_model
model = linear_model.LinearRegression()
#Fit the model
model.fit(X_train, y_train)
#Prediction
print("Predict value " + str(model.predict([X_test[142]])))
print("Real value " + str(y_test[142]))


#Score/Accuracy
print("Accuracy --> ", model.score(X_test, y_test)*100)

#RandomForestRegression
#Train the model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=1000)

#Fit
model.fit(X_train, y_train)
#Score/Accuracy
print("Accuracy --> ", model.score(X_test, y_test)*100)


#GradientBoostingRegressor

#Train the model
from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor(n_estimators=100, max_depth=4)
#Fit
GBR.fit(X_train, y_train)
print("Accuracy --> ", GBR.score(X_test, y_test)*100)