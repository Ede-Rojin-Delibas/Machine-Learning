import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
#seaborn default set işlemi
sns.set()
train_data=pd.read_csv('14_Model_Secimi/data/housing/train.csv')
test_data=pd.read_csv('14_Model_Secimi/data/housing/test.csv')
# print(train_data.head())
# print(test_data.head())
#bütün datayı birleştire(ıd ve salesprice sütunları olmadan)
all_data=pd.concat((train_data.loc[:,'MSSubClass':'SaleCondition'],
                   test_data.loc[:,'MSSubClass':'SaleCondition']))
# print(all_data.head())
# print(train_data['GrLivArea'].describe())
#std =525.48
#3 standart sapma=525.48*3=1576.44
#ortalamanın 3 std sağı ve solu olur.
#-60.98 ile 3091.9 bu değerlerin dışında kalan kısımlar outlier'dır.
#figure ün şekli
# rcParams['figure.figsize'] = (6.0,6.0)
#seaborn ile görselleştirme
# sns.scatterplot(x='GrLivArea',y='SalePrice',data=train_data)
# plt.show()
#outlier ları atmadan önce verinin sekli
# print(train_data.shape) #(1460,81)
#outlierların çıkartılması
train_data=train_data.drop(train_data[(train_data['GrLivArea']>3200)].index).reset_index(drop=True)
#print(train_data.shape) #(1447,81)
#outlier atılmış haliyle test datayı birleştirelim
all_data=pd.concat((train_data.loc[:,'MSSubClass':'SaleCondition'],test_data.loc[:,'MSSubClass':'SaleCondition']))
# print(all_data.head())
#numerik sütunların string yapılması(normalde string olması gereken sütunlar)
all_data['MSSubClass']=all_data['MSSubClass'].astype('str')
all_data['OverallCond']=all_data['OverallCond'].astype('str')
all_data['YrSold']=all_data['YrSold'].astype('str')
all_data['MoSold']=all_data['MoSold'].astype('str')
#kategorik sütunları encode etme(LabelEncoder)
from sklearn.preprocessing import LabelEncoder
#encode edilecek sütunlar
# print(all_data.columns)
cols=['FireplaceQu', 'BsmtQual','BsmtCond','GarageQual','GarageCond','ExterQual','ExterCond',
      'HeatingQC','PoolQC','KitchenQual','BsmtFinType1','BsmtFinType2','Functional','Fence','BsmtExposure',
      'GarageFinish','LandSlope','LotShape','PavedDrive','Street','Alley','CentralAir','MSSubClass',
      'OverallCond','YrSold','MoSold']
#döngü ile bütün sütunları encode et
for c in cols:
    lbl=LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c]=lbl.transform(list(all_data[c].values))
# print(all_data.columns)
#one hot encoder öncesi datanın şekli
# print(all_data.shape) #(2906,79)
all_data=pd.get_dummies(all_data)
#print(all_data.shape) #(2906,218)
# print(all_data.columns)
# Normalizasyon #
#scipy in skew fonksiyonu bize skewness yani kuyruk değerini verir.
from scipy.stats import skew
#'SalePrice' için histogram
# rcParams['figure.figsize'] = (12.0,6.0) #şeklin boyutu
# sns.displot(train_data['SalePrice'],kde=True)
# plt.legend(labels=['Skewness: %.2f' %train_data['SalePrice'].skew()],loc='best')
# plt.show()
# normalizedSalePrice=np.log1p(train_data["SalePrice"])
# rcParams['figure.figsize'] = (12.0,6.0)
# sns.displot(normalizedSalePrice,kde=True)
# plt.legend(labels=['Skewness: %.2f' %normalizedSalePrice.skew()],loc='best')
# plt.show()
#saleprice yerine log transform edilmiş halini yazabiliriz.
train_data['SalePrice']=np.log1p(train_data['SalePrice'])
#eksik verileri yönetme:ortalama(mean) ile yönetme
#print(all_data.isnull().any().any()) #True yani eksik veri var
all_data=all_data.fillna(all_data.mean())
#print(all_data.isnull().any().any()) #False
#model matrislerinin oluşturulması
X_train=all_data[:train_data.shape[0]]
X_test=all_data[train_data.shape[0]:]
y=train_data.SalePrice
# print(train_data.head())
# print(X_train.shape) #(1447,218)
#lineer reg
from sklearn.model_selection import cross_val_score
#k-fold cross validation kullanarak RMSE hesapla
def rmse_cv(model,cv=5):
    rmse=np.sqrt(-cross_val_score(model,X_train,y,scoring="neg_mean_squared_error",cv=cv))
    return rmse
from sklearn.linear_model import LinearRegression
# lr=LinearRegression()
# rmse=rmse_cv(lr)
#print("RMSE Ortalaması:{}, std:{}".format(rmse.mean(),rmse.std())) #0.1259, 0.0121
#fit linear model
# lr.fit(X_train,y)
#katsayılar
# weights=lr.coef_
# print(weights)
#print(weights.shape) # (218,)
#en büyük değerli katsayıları alalım(mutlak değer)
# coef=pd.Series(weights,index=X_train.columns)
#ilk 10 ve son 10 büyük katsayıyı alalım (important coefficients)
# imp_coef=pd.concat([coef.sort_values().head(10),coef.sort_values().tail(10)])
#şimdi bu katsayıları görelim
# imp_coef.plot(kind='barh')
# plt.title("En büyük değerli katsayılar ")
# plt.show()
#Ridge Reg. (alpha=0.1) alalım
from sklearn.linear_model import Ridge
# ridgeModel=Ridge(alpha=0.1)
# rmse=rmse_cv(ridgeModel)
#print("RMSE Ortalaması:{}, std: {}".format(rmse.mean(),rmse.std())) #0.1242 ; 0.0115
#ridge ile az da olsa bir iyileşme sağlandı
# ridgeModel.fit(X_train,y)
# coef_ridge=pd.Series(ridgeModel.coef_,index=X_train.columns)
# imp_coef_ridge=pd.concat([coef_ridge.sort_values().head(10),coef_ridge.sort_values().tail(10)])
# imp_coef_ridge.plot(kind='barh')
# plt.title("En Büyük Değerli Katsayılar")
# plt.show()
#Sorun:Katsayılarımız hala büyük, alpha değerinin tam ayarlanamadığı anlamına geliyor.
#CV ile birçok alpha değerini deneyip en iyisini bulalım(hyperparameter tuning)
# alphas=[0.05,0.1,0.3,1,3,5,10,15,30,50,75]
# cv_ridge= [rmse_cv(Ridge(alpha=alpha)).mean() for alpha in alphas]
# cv_ridge=pd.Series(cv_ridge,index=alphas)
# cv_ridge.plot(title="Alpha Değişimi & Ridge Regression RMSE Değeri")
# plt.xlabel("Alpha")
# plt.ylabel("rmse")
# plt.show()
#grafikten görüldüğü gibi optimum alpha değeri = 10
# optimumRidgeAlpha=cv_ridge[cv_ridge==cv_ridge.min()].index.values[0]
# print("Optimal Ridge Alpha Değeri: {}".format(optimumRidgeAlpha)) #çalışmadı
# En düşük RMSE değerini bul
# min_rmse = min(cv_ridge)  # Listenin minimum değerini bul
# Bu minimum değer olan RMSE'nin indeksini bul ve alphas listesinden seç
# optimumRidgeAlpha = alphas[cv_ridge.index(min_rmse)]
#print("Optimal Ridge Alpha Değeri: {}".format(optimumRidgeAlpha)) #15
# ridgeModel=Ridge(alpha=optimumRidgeAlpha)
# rmse=rmse_cv(ridgeModel)
#print("RMSE Ortalaması: {}, std: {}".format(rmse.mean(),rmse.std())) #0.115; 0.0075
# ridgeModel.fit(X_train,y)
# coef_ridge=pd.Series(ridgeModel.coef_,index=X_train.columns)
# imp_coef_ridge=pd.concat([coef_ridge.sort_values().head(10),coef_ridge.sort_values().tail(10)])
# imp_coef_ridge.plot(kind='barh')
# plt.title("En büyük değerli katsayılar")
# plt.show()
#ridge regresyon ile katsayı değerleri çok daha küçük olmuştur.
#en büyük değerli katsayılar
# ridge_coef=pd.Series(ridgeModel.coef_,index=X_train.columns)
# ridge_imp_coef=pd.concat([ridge_coef.sort_values().head(10),ridge_coef.sort_values().tail(10)])
# rcParams['figure.figsize']=(8.0,10.0) #grafiğin boyutu
# df=pd.DataFrame({'Ridge Regresyon':ridge_imp_coef, 'Linear Regression':imp_coef})
# df.plot(kind='barh')
# plt.title("En büyük değerli katsayılar Ridge-Linear Regression")
# plt.show()
#Lasso Regression
from sklearn.linear_model import Lasso
#determine RMSE for Lasso Regression model with alpha=0.1
# lassoModel=Lasso(alpha=0.1)
# rmse=rmse_cv(lassoModel)
#print("RMSE Ortalaması: {}, std : {}".format(rmse.mean(),rmse.std())) #0.160, 0.0047
alphas=[0.05,0.1,0.3,1,3,5,10,15,30,50,75]
# cv_lasso=[rmse_cv(Lasso(alpha=alpha)).mean() for alpha in alphas]
# cv_lasso=pd.Series(cv_lasso,index=alphas)
# rcParams['figure.figsize']=(12.0,6.0)
# cv_lasso.plot(title="Alpha üzerinden Lasso Regression için RMSE")
# plt.xlabel("Alpha")
# plt.ylabel("RMSE")
# plt.show()
#Teorik olarak RMSE nin bazı alpha değerleri için azalıyor olmasını beklerdik.
#Sebebi bizim verdiğimiz alpha değerlerinin çok uygun olmaması.Verdiğimiz listede en uygun değer yok.
#Alpha değerlerini manuel vermek yerine sklearn ün LassoCV classını kullanarak en uygun alphayı bulabiliriz.(Hyperparameter Tuning)
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
#lasso da katsayıların çoğu sıfır değerli oldu.Bu aslinda regularizasyonun amacıdır .
# ridge_coef=pd.Series(ridgeModel.coef_,index=X_train.columns)
# print("Sıfır olmayan katsayı adedi- Ridge:",sum(ridge_coef!=0))
# print("Sıfır katsayı adedi - Ridge:",sum(ridge_coef==0))
# lasso_coef=pd.Series(lassoModel.coef_,index=X_train.columns)
# print("Sıfır olmayan katsayı adedi- Lasso:",sum(lasso_coef!=0))
# print("Sıfır katsayı adedi - Lasso:",sum(lasso_coef==0))
#Lasso da 123 adet değişken 0 değerini aldı.Yani aslında bunların çok da önemli olmadıkları ortaya çıktı.
# Dolayısıyla Lasso bir yerde Feature Ellimination(Değişken Azaltma) yapmıştır.Lasso nun çok sık kullanılma yerlerinden biri de Feature Ellimination dır.


