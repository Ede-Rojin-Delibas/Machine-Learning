import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import math
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder #for label encoding
from sklearn.preprocessing import OneHotEncoder
# from SMarket import indexler
#OLS -tek değişken olarak tablo okuma
df=pd.read_csv('4_Ogrenme_Learning/data/Advertising.csv',index_col=0)
# data=df[['TV','sales']]
#input ->output
# X=data['TV']
#output
# y=data['sales']
#Hazırlık(Boyut kontrollerini yapmamız lazım) (200,1) şeklinde
# X=X.values.reshape(-1,1)
# y=y.values.reshape(-1,1)
# X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=100)
#Statsmodels OLS
# X_train_ols=sm.add_constant(X_train)
#statsmodels ols modeli oluştur.
# sm_model=sm.OLS(y_train,X_train_ols)
#OLS modelinin sonuçlarını al
# sonuc=sm_model.fit()
#OLS özet tablosunu oluştur
# print(sonuc.summary())
#Çoklu Linear Regresyon Projesi -Advertising.csv : Amaç=TV, Radio, Newspaper bütçesi ile satış arasındaki ilişkiyi Lineer regresyon ile modellemek
#EDA
# sns.pairplot(df)
# plt.show() #satış ve değişkenler arasındaki ilişkiyi bulmaya çalışıyoruz
# X=df[['TV','radio','newspaper']]
# y=df['sales']
# print(df.columns)
# sns.pairplot(df,x_vars=df.columns[:3],y_vars=df.columns[3],height=5)
#modeli oluştur
# lr= LinearRegression() #linear Regresyon nesnesini yarat.
#hazırlık, önce input ve output un boyutuna bakalım
# print(X.shape) (200,3)
# print(y.shape) #y yanlış:(200,)
# y=y.values.reshape(-1,1)
# print(y.shape)
#train-test split
# X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=100)
# print(X_train.shape) #(140,3)
# print(y_train.shape) #(140,1)
# print(X_test.shape)  #(60,3)
# print(y_test.shape)  #(60,1)
#Lineer Regresyon modelini çalıştır(fit)
# lr.fit(X_train,y_train)
# print(lr.intercept_)
# print(lr.coef_) #slope 3 tane olucak çünkü 3 katsayı var.
# katsayilar=pd.DataFrame(lr.coef_,columns=['beta_1(TV)','beta_2(Radio)','beta_3(newspaper)'])
# print(katsayilar) #newspaper ın katsayısı çok ama çok küçük ; katsayının çok küçük olması etki etmediğini gösterir
#Tahmin
# y_pred=lr.predict(X_test)
# print(y_pred)
# print(y_pred.shape) #(60,1)
#gerçek data ile tahmin datasını çiz:gerçek label-y_test, tahmin - y_pred, input-X_test
# print(y_pred[0:10])
# print(y_test[0:10])
#Her bir tahmin arasındaki değişimi görelim
# indexler=range(1,61)
# fig,ax=plt.subplots(figsize=(6,6))
# ax.plot(indexler,y_test,label='Grand Truth',color='red',linewidth=2) #Gerçek data
# ax.plot(indexler,y_pred,label='Prediction',color='green',linewidth=2) #tahmin
# plt.title('Gerçek - Prediction')
# plt.xlabel('Data Index')
# plt.ylabel('Sales')
# plt.legend(loc='upper left')
# plt.show()
#Hataları çiz, Hata: Residuals (y-y^) ,y_test-y_pred
#Her bir tahmin noktasındaki hatayı görelim
# indexler=range(1,61)
# ax.plot(indexler,y_test-y_pred,label='Residuals',color='red',linewidth=2)
#sıfır doğrusunu çiz
# ax.plot(indexler,np.zeros(60),color='black')
# plt.title('Hatalar')
# plt.xlabel('Data Index')
# plt.ylabel('Sales')
# plt.legend(loc='upper left')
# plt.show()
#model doğruluğunu kontrol et
# r_2=r2_score(y_test,y_pred) #r_2 yi hesaplama
# print(r_2) #%90 doğruluk,çok iyi
# mse=mean_squared_error(y_test,y_pred)#MSE->RMSE
# print(mse)
# rmse=math.sqrt(mse)#RMSE
# print(rmse)#ortalamada 1.36 yanılıyoruz
# #OLS
# X_train_ols=sm.add_constant(X_train)
# sm_model=sm.OLS(y_train,X_train_ols)
# sonuc=sm_model.fit
# print(sonuc.summary())
#Korelasyon
# sns.heatmap(df.corr(),annot=True)
# plt.show()
#Sonuçlara göre tekrar model kur: newspaper çıkarılarak yeni model oluşturuldu.
# X_train_yeni=X_train[['TV','radio']]#yeni feature matrisi:X_train_yeni
# X_train_yeni.head()
# X_test_yeni=X_test[['TV','radio']]#yeni test matrisi:X_test_yeni
# X_test_yeni.head()
# lr.fit(X_train_yeni,y_train)#modeli tekrar kurgula.
# y_pred_yeni=lr.predict(X_test_yeni)#yeni tahminleri al
# X_train_yeni_ols=sm.add_constant(X_train_yeni)#yeni OLS'i gör
# sm_model=sm.OLS(y_train,X_train_yeni_ols)#yeni ols modeli oluştur
# sonuc=sm_model.fit()#OLS modelinin sonuçlarını al
# print(sonuc.summary())#OLS özet tablosunu yazdır
#şuan 2 değişkenimiz var (tv ve radio) ve ikisi de önemli p değerleri 0.000
#ispat:newspaper gerçekten önemsiz mi:bu sefer sadece newspaper a bakıcaz
# X=df['newspaper']
# y=df['sales']
# X=X.values.reshape(-1,1)#yeniden boyutlandırma
# y=y.values.reshape(-1,1)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=100)
# X_train_ols=sm.add_constant(X_train)
# model=sm.OLS(y_train,X_train_ols)
# result=model.fit()
# print(result.summary())
#kategorik verilerin sayısal verilere dönüştürülmesi
# evlilik_durumu=('Evli','Bekar','Belirtilmemiş')#ham veri
# evlilik_df=pd.DataFrame(evlilik_durumu,columns=['Evlilik_Durumu'])#df yarat
# # print(evlilik_df.info())#object tipinde
# #sutünü 'category' ye dönüştür
# evlilik_df['Evlilik_Durumu']=evlilik_df['Evlilik_Durumu'].astype('category')
# # print(evlilik_df.info()) #category
# evlilik_df['Evlilik_Kategorileri'] = evlilik_df['Evlilik_Durumu'].cat.codes
# print(evlilik_df)
evlilik_durumu=('Evli','Bekar','Belirtilmemiş')#ham veri
evlilik_df=pd.DataFrame(evlilik_durumu,columns=['Evlilik_Durumu'])
#label encoder nesnesi yarat
# label_encoder=LabelEncoder()
# evlilik_df['Evlilik_Kategorilerii_Sklearn']=label_encoder.fit_transform(evlilik_df['Evlilik_Durumu'])
#One-Hot Encoding
#one hot encoder nesnesi yarat
# enc=OneHotEncoder(handle_unknown='ignore')
# enc_result=enc.fit_transform(evlilik_df[['Evlilik_Durumu']])#evlilik durumunu enc ye ver
# enc_df=pd.DataFrame(enc_result.toarray())#evlilik durumunu enc ye ver
# evlilik_df=evlilik_df.join(enc_df)    #yeni olan enc_df yi evlilik_df ye ekle
#her bir kolon için -> binary(0,1) değerler olan sütunlar üret
dummy_df=pd.get_dummies(evlilik_df,columns=['Evlilik_Durumu'])
# print(dummy_df)
evlilik_df=evlilik_df.join(dummy_df)






