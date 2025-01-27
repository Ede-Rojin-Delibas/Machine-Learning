import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn import tree
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('bank-additional.csv',sep=';')
#duration sütununu çıkaralım(gereksiz bir sütun sanırım)
df=df.drop('duration',axis=1)
#EDA(Exploratory Data Analysis)
# print(df.shape) #(4119,20)
# print(df.head())
#sütun adları
col_names=df.columns
# print(col_names)
#y:arandıktan sonra soruluyor müşteriye kredi istiyor musun diye yes ve no onu ifade ediyor.
#target variable 'ın dağılımı
# print(df['y'].value_counts())
#sınıfların yes-no yüzde dağılımlarını görelim
# print(df['y'].value_counts() / float(len(df)))
"""y
    no:3668
    yes:451
    #yüzdeler
    no:0.890
    yes:0.109 
    #yorum: sınıflar arasında büyük bir fark var(%89,%11).Veri düzensiz,inbalanced.Sonuçları yorumlarken dikkatli olmamız gerekiyor. 
   """
# print(df.isnull().sum()) #eksik data kontrolü
# print(len(df.columns))#toplam kolon sayısı
# print(df.select_dtypes(include=['int64','float64']).shape)#numeric kolonların adedi (4116,9)
# numeric_cols=df.select_dtypes(include=['int64','float64']).columns #numerik sütunlar
# print(numeric_cols)
"""Index(['age', 'campaign', 'pdays', 'previous', 'emp.var.rate',
       'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'],
      dtype='object')"""
#kategorik kolonların (object) adedi
# print(df.select_dtypes(include='object').shape) #(4119,11)
#kategorik kolonlar
# cat_cols=df.select_dtypes(include='object').columns
# print(cat_cols)
'''Index(['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
       'month', 'day_of_week', 'poutcome', 'y'],
      dtype='object')''' #y de bu kategorik sütunlar içerisinde ayıracağız y yi
"""VERİ ÖZETİ:
    Toplam 4119 adet veri
    19 değişken
    9 adet numerik değişken
    10 adet kategorik değişken
    1 adet y değişkeni"""
#numerik değişkenlerin dağılımı
# df.hist(column=numeric_cols,figsize=(10,10))
# plt.subplots_adjust(wspace=0.5,hspace=0.5)
# plt.show()
#bunu yapma sebebimiz data içerisinde bir terslik var mı ya da outlier'lar var mı bunu anlamak
# print(df['poutcome'].value_counts())
df['poutcome']=df['poutcome'].map({'failure':-1,'nonexistent':0,'success':1})
# print(df['poutcome'].value_counts())
# print(df['default'].value_counts())#default,mevcutta kredisinin olmaması iyi olarak yorumlandı
df['housing']=df['housing'].map({'yes':-1,'unknown':0, 'no':1})
df['loan']=df['loan'].map({'yes':-1,'unknown':0, 'no':1})
# 'default' sütunundaki stringleri numerik değerlere dönüştürme
df['default'] = df['default'].map({'yes': 1, 'no': 0, 'unknown': -1})
df['default'] = df['default'].fillna(-1)  # NaN değerleri -1 ile dolduralım (isteğe bağlı)
nominal=['job','marital','education','contact','month','day_of_week']
# print(df.shape)#one hot encoding yapmadan önce df in şekli (4119,20)
df=pd.get_dummies(df,columns=nominal)
# print(df.shape)#OHE sonrası (4119,50)
#target variable y yi encode edelim
df['y']=df['y'].map({'yes':1,'no':0})
# print(df.head())
# numeric_cols_ohe=df.select_dtypes(include=['int64','float64']).columns
# numeric_cols_ohe = numeric_cols_ohe.drop('y')
# print(numeric_cols_ohe)
#FEATURE VECTOR AND TARGET VARIABLE
X=df.drop(['y'],axis=1)
y=df['y']
# print(X.shape) #(4119,54)
# print(y.shape)#(4119,)
#TRAIN - TEST SPLIT
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
# print(X_train.shape)#X_train şekli(3295, 54)
# print(X_test.shape )#X_test şekli(824, 54)
#Feature Scaling:değişkenleri aynı ölçülere getirme
cols=X_train.columns #X_train sütunlarını tut
# print(cols)
# print(X_train[numeric_cols_ohe])#veriler
#X_train üzerinde standardScaler eğiticez.Hem X_train hem de X_test i aynı şekilde Scale edicez
#Standart scale yarat
# scaler=StandardScaler()
# X_train=scaler.fit_transform(X_train)#X_train üzerinde fit ve transform yap
# X_test=scaler.transform(X_test)#X_test i transform yap
#StandardScaler transform sonrası veri yapıları değişir
# print(type(X_train))
# print(X_train.dtypes) #tüm sütunların veri tiplerini gösterir
# print(X_train.head()) #default sütununda no değerleri var
#hatadan sonra yeniden scaling yapma
#numeric sütunların belirlenmesi
numeric_cols=X_train.select_dtypes(include=['int64','float64']).columns
#standardScaler
scaler=StandardScaler()
#numerik sütunlara scale uygulayın
X_train[numeric_cols]=scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols]=scaler.transform(X_test[numeric_cols])
#kontrol
# print(X_train.head())
#Decision Tree Classifier(Gini Index ile)
# #DTC modelini criterion gini index olarak instantiate edelim
# clf_gini=DecisionTreeClassifier(criterion='gini',max_depth=3,random_state=0)
# #model'i fit edelim
# clf_gini.fit(X_train,y_train)
# #gini index kullanarak oluşturduğumuz model ile tahmin yapalım
# y_pred_gini=clf_gini.predict(X_test)
# #ROC_AUC skorunu kontrol edelim/model performansını değerlendirme
# y_pred_gini_score=roc_auc_score(y_test,y_pred_gini)
# print('Modelin Gini Index ile ROC-AUC skoru: {0:0.4f}'.format(y_pred_gini_score))
#train ve test roc_auc değerlerini karşılaştıralım
# y_pred_train_gini=clf_gini.predict(X_train)
# y_pred_train_gini_score=roc_auc_score(y_train,y_pred_train_gini)
# print('Modelin gini index ile ROC-AUC Skoru: {0:0.4f}'.format(y_pred_train_gini_score))
#train ve test skorlarını karşılaştıralım
# print('Train set skoru:{:.4f}'.format(y_pred_train_gini_score))
# print('Test set skoru:{:.4f}'.format(y_pred_gini_score))
#SONUÇ:train=%60, test=%58 overfitting tehlikesi görünmüyor
#Decision Tree yi görselleştirelim
# plt.figure(figsize=(24,16))
# tree.plot_tree(clf_gini.fit(X_train,y_train))
# plt.show()
#Decision Tree Classifier Entropy ile
#Decision T.C. modelini criterion entropy  index olarak instantiate edelim
# clf_ent=DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=0)
#model i fit edelim
# clf_ent.fit(X_train,y_train)
# y_pred_ent=clf_ent.predict(X_test)
#ROC-AUC skorunu kontrol edelim
# y_pred_ent_score=roc_auc_score(y_test,y_pred_ent)
# print('Modelin Entropy ile ROC-AUC skoru:{0:0.4f}'.format(y_pred_ent_score))
# y_pred_train_ent=clf_ent.predict(X_train)
# y_pred_train_ent_score=roc_auc_score(y_train,y_pred_train_ent)
# print('Modelin Entropy ile ROC-AUC skoru:{0:0.4f}'.format(y_pred_train_ent_score))
#train ile test set skorları
# print('Train set skoru:{:.4f}'.format(y_pred_train_ent_score))
# print('Test set skoru:{:.4f}'.format(y_pred_ent_score))
#yorum:overfit tehlikesi görünmüyor
#Decision Tree yi görselleştirelim
# plt.figure(figsize=(24,16))
# tree.plot_tree(clf_ent.fit(X_train,y_train))
# plt.show()
#OVERFIT Ispatı(gini indexle):max_depth arttırılır.
clf_gini=DecisionTreeClassifier(criterion='gini',max_depth=9,random_state=0)
clf_gini.fit(X_train,y_train)#modeli fit edelim
y_pred_gini=clf_gini.predict(X_test)
y_pred_gini_score=roc_auc_score(y_test,y_pred_gini)
# print('Modelin gini index ile ROC-AUC Skoru: {0:0.4f}'.format(y_pred_gini_score))
y_pred_train_gini=clf_gini.predict(X_train)
y_pred_train_gini_score=roc_auc_score(y_train,y_pred_train_gini)
# print('Modelin gini index ile ROC-AUC Skoru: {0:0.4f}'.format(y_pred_train_gini_score))
#train ve test set skorlarını kıyaslayalım
print('Train set skoru:{:.4f}'.format(y_pred_train_gini_score))
print('Test set skoru:{:.4f}'.format(y_pred_gini_score))
#sonuç:0.7388, 0.5859 yani %74, %59 :OVERFIT
#görselleştirme
plt.figure(figsize=(24,16))
tree.plot_tree(clf_gini.fit(X_train,y_train))
plt.show()












