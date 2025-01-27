import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.model_selection import train_test_split #scikit learn den import
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import math
# df=pd.read_csv('Smarket.csv')
# print(df.describe())
#Bugünki fiyat - Önceki gün fiyat
# df4=df[['Today','Direction']].copy()
#negatifleri yönet
# df4['Today']=df4['Today'].abs()
# X=df4['Today']
# y=df4['Direction']
#boxplot
# y_grouped=df4.groupby('Direction')
# print(y_grouped.count())
# y_grouped.boxplot(column='Today',by='Direction', rot=90,figsize=(6,6))
#görsel başlıklarını temizleme
# plt.title("Boxplot for Today by Direction")
# plt.xlabel("Direction")
# plt.ylabel("Today")
# plt.show()
"""Direction       
Down         602
Up           648 
Görüldüğü gibi hisse senetlerinin bugünkü fiyatının aşağı mı yoksa yukarı mı olacağına
bir önceki gün fiyatı üzerinden gidemiyoruz.Sağlıklı bir sonuç vermiyor. 1250 kayıttan 602
bir önceki güne göre düşmüş, 648 ise yükselmiş. Neredeyse yarı yarıya (BASE ORAN:DEFAULT ORAN %50)"""
#UNSUPERVISED LEARNING
#NCI60 veri setiyle çalışma
# df=pd.read_csv('NCI60.csv')
# print(df.head())
# X=df.iloc[:,1:6831]
# print(X)
# veriyi ölçekleme
# sc=StandardScaler()
# X_scaled=sc.fit_transform(X)
# print(X_scaled)
#PCA : principal Component Analysis
# pca=PCA(n_components=2)
# pca_result=pca.fit_transform(X_scaled)
# print('Eigenvalues')
# print(pca.explained_variance_)
#Variances Percentage
# print('Variances Percentage')
# print(pca.explained_variance_ratio_ * 100)
# principalDf=pd.DataFrame(data=pca_result,columns=['PC1','PC2'])
# print(principalDf.head(10))
# finalDf=pd.concat([principalDf,df[['labs']]],axis=1)
# print(finalDf.head(10))
#Plot PCA
# fig=plt.figure(figsize=(5,5))
# ax=fig.add_subplot(1,1,1)
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_title('En Etkili Genler - Hastalık')
# targets=['COlON','MELANOMA','LEUKEMIA']
# for target in targets:
#     indexler = finalDf['labs'] == target
#     ax.scatter(finalDf.loc[indexler,'PC1'],
#                finalDf.loc[indexler,'PC2'],s=50)
# ax.legend(targets,loc='upper right')
# ax.grid()
# plt.show()
df=pd.read_csv('4_Ogrenme_Learning/data/Advertising.csv')
# print(df.head())
# print(df.describe())
X_1=df['TV']
X_2=df['radio']
X_3=df['newspaper']
# print(type(X_1)) #series
# print(X_1.shape)#series şeklinde listeye çevrilecek
y=df['sales']
#Regresyon için Hazırlık Yapma:shape leri vektöre çevirme(regresyon için)
linear_Regressor=LinearRegression()
#dataframe'lerin şeklini düzenle-> output
y_r=y.values.reshape(-1,1)
# print(y.shape)
# print(y_r.shape)
#dataframelerin şeklini düzenle -> input
X_1_r=X_1.values.reshape(-1,1)
X_2_r=X_2.values.reshape(-1,1)
X_3_r=X_3.values.reshape(-1,1)
# print(X_1_r.shape)
#Satış -TV ilişkisi:data -> X_1
# plt.scatter(X_1,y,c='orange')
# plt.xlabel('TV')
# plt.ylabel('Satış')
# plt.title('Satış - TV')
# plt.show()
#Regresyon
# linear_Regressor.fit(X_1_r,y_r)
# y_pred_1=linear_Regressor.predict(X_1_r)
# plt.plot(X_1,y_pred_1,color='red')
# plt.xlabel('TV')
# plt.ylabel('Satış')
# plt.title('Satış - TV')
# plt.show()
#Satış-Radio ilişkisi:data -> X_2
# plt.scatter(X_2,y,c='green')
# linear_Regressor.fit(X_2_r,y_r)
# y_pred_2=linear_Regressor.predict(X_2_r)
# plt.plot(X_2,y_pred_2,color='red')
# plt.xlabel('Radio')
# plt.ylabel('Satış')
# plt.title('Satış-Radio')
# plt.show()
#Satış-Newspaper ilişkisi:data -> X_3
# plt.scatter(X_3,y,c='pink')
# linear_Regressor.fit(X_3_r,y_r)
# y_pred_3=linear_Regressor.predict(X_3_r)
# plt.plot(X_3,y_pred_3,color='red')
# plt.xlabel('Newspapaer')
# plt.ylabel('Satış')
# plt.title('Satış-Newpaper')
# plt.show()
#Income.csv:Eğitim düzeyi - gelir arasındaki ilişkiye bakacağız.
# income=pd.read_csv('4_Ogrenme_Learning/data/Income.csv')
#print(income.head(5))
# X=income['Education']
# y=income['Income']
# #regresyon için hazırlık:reshaping
# X_r=X.values.reshape(-1,1)
# y_r=y.values.reshape(-1,1)
# #grafik-linear Regression
# plt.figure(figsize=(10,6))
# plt.scatter(X,y)
# lr=LinearRegression() #regresyon
# lr.fit(X_r,y_r)
# y_pred=lr.predict(X_r)
# plt.plot(X,y_pred,color='red')
# plt.xlabel('Education')
# plt.ylabel('Income')
# plt.title('Education-Income')
# plt.show() #oluşan grafik doğrusaldır.
#Basit/tek lineer regresyon Proje
df=pd.read_csv('4_Ogrenme_Learning/data/Advertising.csv')
# print(df.head(10))
# print(df.tail(2)) #son iki satır
#data ile ilgili genel bilgileri görmek için
# print(df.info())
#temel istatistik
# print(df.describe())
#datayı görselleştir
data=df[['TV','sales']]
X=data['TV'] #input ->feature
y=data['sales'] #output
# print(type(X))
#grafik çiz
plt.figure(figsize=(6,6))
sns.scatterplot(data=data,x='TV',y='sales',color='orange')
plt.title('SALES-TV')
# plt.show()
#Modeli Oluştur
#linear regresyon nesnesini yarat
lr=LinearRegression()
#input ve output un şekline bakalım
# print('X in boyutu:', X.shape)
# print('y nin boyutu:', y.shape) #ortaya çıkan sonuç sklearn ün LinearRegression class'ının istediği gibi değil
#yani iki boyutlu değil (200,0) olarak geliyor her ikisi de (200,1) olması lazım
#yeniden boyutlandırmamız lazım
X=X.values.reshape(-1,1) #anlamı baştakini kabul et ve yanına bir sütun ekle
y=y.values.reshape(-1,1)
# print(y.shape)
#train-test split
X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=100)
# print('X train boyutu:',X_train.shape) #tipi numpy.ndarray
# print('y test boyutu:',y_test.shape)
#lineer regresyon modelini çalıştır(fit)
#Regresyonu çalıştır ->fit
lr.fit(X_train,y_train) #betaları vericek yani katsayıları
#katsayıları hesaplama
# print(lr.intercept_)
# print(lr.coef_)
#tahmin yap.
y_pred=lr.predict(X_test)
# print(y_pred) #(60,1)
#gerçek data ile tahmin datasını çiz
#Gerçek data
fig,ax=plt.subplots(figsize=(6,6))
ax.scatter(X_test,y_test,label='Grand Truth',color='red')
#prediction
ax.scatter(X_test,y_pred,label='Prediction',color='green')
plt.title('SALES - TV - PREDICTION')
plt.xlabel('TV')
plt.ylabel('SALES')
plt.legend(loc='upper left')
# plt.show()
#ilk 10 gerçek y değerini görelim
# print(y_test[0:10])
#ilk 10 tahmin değerini görelim
# print(y_pred[0:10])
#Her bir tahmin noktadaki değişimi görelim
indexler=range(1,61)
fig,ax=plt.subplots(figsize=(6,6))
ax.plot(indexler,y_test,label='Grand Truth',color='red',linewidth=2)
#Tahmin -> Prediction
ax.plot(indexler,y_pred,label='Grand Truth',color='green',linewidth=2)
plt.title('SALES - TV - PREDICTION')
plt.xlabel('TV')
plt.ylabel('SALES')
plt.legend(loc='upper left')
# plt.show()
#Hataları çiz
indexler=range(1,61)
#Residuals-Hatalar
fig,ax=plt.subplots(figsize=(6,6))
ax.plot(indexler,y_test-y_pred,label='Residuals',color='green',linewidth=2)
#sıfır doğrusunu çiz
ax.plot(indexler,np.zeros(60),color='black')
#Model doğruluğunu kontrol etmek(RMSE,R^2)
r_2=r2_score(y_test,y_pred) #r^2
# print(r_2)
mse=mean_squared_error(y_test,y_pred) #mse
print(mse)
rmse=math.sqrt(mse) #rmse
print(rmse)

