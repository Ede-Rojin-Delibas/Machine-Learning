import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

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
df=pd.read_csv('NCI60.csv')
# print(df.head())
X=df.iloc[:,1:6831]
# print(X)
# veriyi ölçekleme
sc=StandardScaler()
X_scaled=sc.fit_transform(X)
# print(X_scaled)
#PCA : principal Component Analysis
pca=PCA(n_components=2)
pca_result=pca.fit_transform(X_scaled)
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
income=pd.read_csv('4_Ogrenme_Learning/data/Income.csv')
#print(income.head(5))
X=income['Education']
y=income['Income']
#regresyon için hazırlık:reshaping
X_r=X.values.reshape(-1,1)
y_r=y.values.reshape(-1,1)
#grafik-linear Regression
plt.figure(figsize=(10,6))
plt.scatter(X,y)
lr=LinearRegression() #regresyon
lr.fit(X_r,y_r)
y_pred=lr.predict(X_r)
plt.plot(X,y_pred,color='red')
plt.xlabel('Education')
plt.ylabel('Income')
plt.title('Education-Income')
plt.show() #oluşan grafik doğrusaldır.