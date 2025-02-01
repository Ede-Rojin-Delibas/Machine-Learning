import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
#seaborn default for set process
sns.set()
train_data=pd.read_csv('14_Model_Secimi/data/housing/train.csv')
test_data=pd.read_csv('14_Model_Secimi/data/housing/test.csv')
# print(train_data.head())
# print(test_data.head())
#merge all data (without id and salesprice columns)
all_data=pd.concat((train_data.loc[:,'MSSubClass':'SaleCondition'],
                   test_data.loc[:,'MSSubClass':'SaleCondition']))
print(all_data.head())
print(train_data['GrLivArea'].describe())
#std =525.48
#3 standard deviation =525.48*3=1576.44
#3 std to the right and left of the mean.
#-60.98 ile 3091.9 parts outside these values are outliers..
rcParams['figure.figsize'] = (6.0,6.0)
#visualize with seaborn
sns.scatterplot(x='GrLivArea',y='SalePrice',data=train_data)
# plt.show()
#shape of data without outliers
print(train_data.shape) #(1460,81)
#removing the outliers
train_data=train_data.drop(train_data[(train_data['GrLivArea']>3200)].index).reset_index(drop=True)
#print(train_data.shape) #(1447,81)
#Let's combine the test data with the outlier removed.
all_data=pd.concat((train_data.loc[:,'MSSubClass':'SaleCondition'],test_data.loc[:,'MSSubClass':'SaleCondition']))
# print(all_data.head())
#Converting numeric columns to strings (columns that should normally be strings)
all_data['MSSubClass']=all_data['MSSubClass'].astype('str')
all_data['OverallCond']=all_data['OverallCond'].astype('str')
all_data['YrSold']=all_data['YrSold'].astype('str')
all_data['MoSold']=all_data['MoSold'].astype('str')
#encoding categorical columns(LabelEncoder)
from sklearn.preprocessing import LabelEncoder
# print(all_data.columns)
cols=['FireplaceQu', 'BsmtQual','BsmtCond','GarageQual','GarageCond','ExterQual','ExterCond',
      'HeatingQC','PoolQC','KitchenQual','BsmtFinType1','BsmtFinType2','Functional','Fence','BsmtExposure',
      'GarageFinish','LandSlope','LotShape','PavedDrive','Street','Alley','CentralAir','MSSubClass',
      'OverallCond','YrSold','MoSold']
#with loop encoding for all columns
for c in cols:
    lbl=LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c]=lbl.transform(list(all_data[c].values))
# print(all_data.columns)
#before one hot encoder data's shape
# print(all_data.shape) #(2906,79)
all_data=pd.get_dummies(all_data)
#print(all_data.shape) #(2906,218)
# print(all_data.columns)
# Normalization #
#scipy's skew function gives us the skewness, that is, the tail value.
from scipy.stats import skew
# histogram for 'SalePrice'
rcParams['figure.figsize'] = (12.0,6.0)
sns.displot(train_data['SalePrice'],kde=True)
plt.legend(labels=['Skewness: %.2f' %train_data['SalePrice'].skew()],loc='best')
# plt.show()
normalizedSalePrice=np.log1p(train_data["SalePrice"])
rcParams['figure.figsize'] = (12.0,6.0)
sns.displot(normalizedSalePrice,kde=True)
plt.legend(labels=['Skewness: %.2f' %normalizedSalePrice.skew()],loc='best')
# plt.show()
#We can write the log transformed version instead of saleprice.
train_data['SalePrice']=np.log1p(train_data['SalePrice'])
#print(all_data.isnull().any().any()) #True, there is null values
all_data=all_data.fillna(all_data.mean())
#print(all_data.isnull().any().any()) #False
#generating model matrixes
X_train=all_data[:train_data.shape[0]]
X_test=all_data[train_data.shape[0]:]
y=train_data.SalePrice
# print(train_data.head())
# print(X_train.shape) #(1447,218)
#lineer reg
from sklearn.model_selection import cross_val_score
#using k-fold cross validation calculating RMSE
def rmse_cv(model,cv=5):
    rmse=np.sqrt(-cross_val_score(model,X_train,y,scoring="neg_mean_squared_error",cv=cv))
    return rmse
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
rmse=rmse_cv(lr)
print("RMSE Ortalaması:{}, std:{}".format(rmse.mean(),rmse.std())) #0.1259, 0.0121
#fit linear model
lr.fit(X_train,y)
#coefficients
weights=lr.coef_
print(weights)
#print(weights.shape) # (218,)
#Let's take the biggest valuable coefficients(mutlak değer)
coef=pd.Series(weights,index=X_train.columns)
#Let's take the first 10 and last 10 biggest coefficients (important coefficients)
imp_coef=pd.concat([coef.sort_values().head(10),coef.sort_values().tail(10)])
imp_coef.plot(kind='barh')
plt.title("En büyük değerli katsayılar ")
# plt.show()
#take Ridge Reg. (alpha=0.1)
from sklearn.linear_model import Ridge
ridgeModel=Ridge(alpha=0.1)
rmse=rmse_cv(ridgeModel)
#print("RMSE Ortalaması:{}, std: {}".format(rmse.mean(),rmse.std())) #0.1242 ; 0.0115
#A slight improvement was achieved with the ridge
ridgeModel.fit(X_train,y)
coef_ridge=pd.Series(ridgeModel.coef_,index=X_train.columns)
imp_coef_ridge=pd.concat([coef_ridge.sort_values().head(10),coef_ridge.sort_values().tail(10)])
imp_coef_ridge.plot(kind='barh')
plt.title("En Büyük Değerli Katsayılar")
# plt.show()
#Problem: Our coefficients are still large, meaning the alpha value is not fully adjusted.
#Let's try many alpha values with CV and find the best one.(hyperparameter tuning)
alphas=[0.05,0.1,0.3,1,3,5,10,15,30,50,75]
cv_ridge= [rmse_cv(Ridge(alpha=alpha)).mean() for alpha in alphas]
cv_ridge=pd.Series(cv_ridge,index=alphas)
cv_ridge.plot(title="Alpha Değişimi & Ridge Regression RMSE Değeri")
plt.xlabel("Alpha")
plt.ylabel("rmse")
# plt.show()
#grafikten görüldüğü gibi optimum alpha değeri = 10
optimumRidgeAlpha=cv_ridge[cv_ridge==cv_ridge.min()].index.values[0]
print("Optimal Ridge Alpha Değeri: {}".format(optimumRidgeAlpha)) #çalışmadı
# find the lowest RMSE value
min_rmse = min(cv_ridge)  # Find the minimum value on List
#Find the index of this minimum value RMSE and select it from the list of alphas
optimumRidgeAlpha = alphas[cv_ridge.index(min_rmse)]
print("Optimal Ridge Alpha Değeri: {}".format(optimumRidgeAlpha)) #15
ridgeModel=Ridge(alpha=optimumRidgeAlpha)
rmse=rmse_cv(ridgeModel)
print("RMSE Ortalaması: {}, std: {}".format(rmse.mean(),rmse.std())) #0.115; 0.0075
ridgeModel.fit(X_train,y)
coef_ridge=pd.Series(ridgeModel.coef_,index=X_train.columns)
imp_coef_ridge=pd.concat([coef_ridge.sort_values().head(10),coef_ridge.sort_values().tail(10)])
imp_coef_ridge.plot(kind='barh')
plt.title("En büyük değerli katsayılar")
# plt.show()
#With ridge regression, coefficient values ​​were much smaller.
#largest coefficients
ridge_coef=pd.Series(ridgeModel.coef_,index=X_train.columns)
ridge_imp_coef=pd.concat([ridge_coef.sort_values().head(10),ridge_coef.sort_values().tail(10)])
rcParams['figure.figsize']=(8.0,10.0) #size of graph
df=pd.DataFrame({'Ridge Regresyon':ridge_imp_coef, 'Linear Regression':imp_coef})
df.plot(kind='barh')
plt.title("En büyük değerli katsayılar Ridge-Linear Regression")
# plt.show()
#Lasso Regression
from sklearn.linear_model import Lasso
#determine RMSE for Lasso Regression model with alpha=0.1
lassoModel=Lasso(alpha=0.1)
rmse=rmse_cv(lassoModel)
#print("RMSE Ortalaması: {}, std : {}".format(rmse.mean(),rmse.std())) #0.160, 0.0047
alphas=[0.05,0.1,0.3,1,3,5,10,15,30,50,75]
cv_lasso=[rmse_cv(Lasso(alpha=alpha)).mean() for alpha in alphas]
cv_lasso=pd.Series(cv_lasso,index=alphas)
rcParams['figure.figsize']=(12.0,6.0)
cv_lasso.plot(title="Alpha üzerinden Lasso Regression için RMSE")
plt.xlabel("Alpha")
plt.ylabel("RMSE")
# plt.show()
#Theoretically, we would expect RMSE to decrease for some alpha values.
#The reason is that the alpha values we gave are not very suitable. There is no most suitable value in the list we gave.
#Instead of giving alpha values manually, we can find the most suitable alpha by using sklearn's LassoCV class..(Hyperparameter Tuning)
from sklearn.linear_model import LassoCV
lassoModel=LassoCV(alphas=np.linspace(0.0002,0.0022,21),cv=5).fit(X_train,y)
lassoModel.alpha_
optimalLassoAlpha=lassoModel.alpha_
#print("Optimal Lasso Alpha: {}".format(optimalLassoAlpha)) #0.0005
lassoModel=Lasso(alpha=optimalLassoAlpha)
rmse=rmse_cv(lassoModel)
#print("RMSE Ortalaması: {} ,std: {}".format(rmse.mean(),rmse.std())) #0.1129; 0.0071
"""Lineer Regresyon:RMSE=0.1259
   Ridge:RMSE=0.1152
   Lasso:RMSE=0.1129
"""
lassoModel.fit(X_train,y)
lasso_coef_=pd.Series(lassoModel.coef_,index=X_train.columns)
lasso_imp_coef_=pd.concat([lasso_coef_.sort_values().head(10),lasso_coef_.sort_values().tail(10)])
rcParams['figure.figsize']=(8.0,10.0)
df=pd.DataFrame({"Lasso Regression":lasso_imp_coef_})
df.plot (kind="barh")
plt.title("En büyük değerli katsayılar Lasso Regression")
plt.show()
#In Lasso, most of the coefficients have zero values. This is actually the purpose of regularization..
ridge_coef=pd.Series(ridgeModel.coef_,index=X_train.columns)
print("Sıfır olmayan katsayı adedi- Ridge:",sum(ridge_coef!=0))
print("Sıfır katsayı adedi - Ridge:",sum(ridge_coef==0))
lasso_coef=pd.Series(lassoModel.coef_,index=X_train.columns)
print("Sıfır olmayan katsayı adedi- Lasso:",sum(lasso_coef!=0))
print("Sıfır katsayı adedi - Lasso:",sum(lasso_coef==0))
#In Lasso, 123 variables took the value 0. So, it turned out that they were not very important.
# Therefore, Lasso has done Feature Elimination (Variable Reduction) somewhere. One of the places
# where Lasso is frequently used is Feature Elimination.


