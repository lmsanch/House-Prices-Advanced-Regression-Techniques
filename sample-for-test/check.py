import numpy as numpy
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

plt.subplots(figsize=(12,9))
sns.distplot(train['SalePrice'], fit=stats.norm)

# Get the fitted parameters used by the function

(mu, sigma) = stats.norm.fit(train['SalePrice'])

# plot with the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')

#Probablity plot

fig = plt.figure()
stats.probplot(train['SalePrice'], plot=plt)
plt.show()