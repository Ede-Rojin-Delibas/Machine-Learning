import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('11_Naive_Bayes/data/wine.data',header=None)
#EDA
# print(df.shape)
#sütun adları yok o yüzden onları ekleyelim
col_names=[
    "class",
    "Alcohol",
    "Malic_acid",
    "Ash",
    "Alcalinity_of_ash",
    "Magnesium",
    "Total_phenols",
    "Flavanoids",
    "Nonflavanoid_phenols",
    "Proanthocyanins",
    "Color_intensity",
    "Hue",
    "OD280/OD315_of_diluted_wines",
    "Proline"
]
df.columns=col_names
# print(df.columns)
# print(df.head())
# print(df.info())
# print(df.describe()) #basit istatistikler
#KATEGORIK DEGISKENLER :Kategorik sütunları al -> == 'O'
categorical=[var for var in df.columns if df[var].dtypes=='O']
# print("Kategorik Degisken sayısı: {} ".format(len(categorical))) :0
#numerik sütunları al !='O'
numerical=[var for var in df.columns if df[var].dtype != 'O']
# print('Numerik değişken sayısı: {}\n'.format(len(numerical)-1))
# print('Numerik sütunlar:',numerical[1:])
# print(df[numerical].head())
#feature and label(girdi % sonuç) ayrımı
X=df.drop('class',axis=1)
y=df['class']
# print(X.head())
# print(y.head())
#sınıf adedini görelim
y.value_counts(sort=False)
#train-test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
#boyutları görelim
# print(X_train.shape)
# print(X_test.shape)
#Feature Scaling
cols=X_train.columns
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
#X_train üzerinden hem fit(eğitme) hem de transform yapalım
X_train=sc.fit_transform(X_train)
#X_test üzerinden transform yapalım
X_test=sc.transform(X_test)
#X_train ve X_test in tipleri numpy.ndarray
# print(type(X_train))
#bunları tekrar pandas DataFrame e çevirmemiz lazım
X_train=pd.DataFrame(X_train,columns=[cols])
X_test=pd.DataFrame(X_test,columns=[cols])
# print(type(X_train))
# print(X_train.head())
#model geliştirme
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
#modeli train ile eğitelim
gnb.fit(X_train,y_train)
#sonuçların tahmini
#test_data ile class tahminlerimizi yapalım
y_pred=gnb.predict(X_test)
#tahmin sonucunun analizi(accuracy score)
from sklearn.metrics import accuracy_score
# print('Accuracy Score: {0:0.4f}'.format(accuracy_score(y_test,y_pred)))
#underfitting için (train düşük, test yüksek)
#train data üzerinden tahmin
y_pred_train=gnb.predict(X_train)
# print('Train data accuracy score:{0:0.4f}'.format(accuracy_score(y_train,y_pred_train)))
''' Accuracy Score: 0.9444
    Train data accuracy score:0.9839
'''
#train ve test data için skorları yazalım
print('Train data score:{:.4f}'.format(gnb.score(X_train,y_train))) #%98
print('Test data score:{:.4f}'.format(gnb.score(X_test,y_test)))#%94
#Hem test hem de train data skorları gayet iyi ve birbirlerine yakın, dolayısıyla overfitting ve underfitting riski yok.