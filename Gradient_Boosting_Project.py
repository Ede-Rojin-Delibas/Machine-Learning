import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer #eksik değerleri doldurmak için
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')
df=pd.read_csv('bank-additional.csv',sep=';')
#duration sütununu çıkaralım(dataset açıklamasında söylüyor.)
df=df.drop('duration',axis=1)
#EDA
#verinin şekli
print(df.shape) #(4119,20)
#sütun adları
col_names=df.columns
#target variable ın dağılımı
df['y'].value_counts()
df.info()
df.isnull().sum()
#toplam kolon sayısı
len(df.columns)
#numerik sütun adedi
df.select_dtypes(include=['int64','float64']).shape
#numerik kolonlar
numeric_cols=df.select_dtypes(include=['int64','float64']).columns
#kategorik sütunlar(object)
cat_cols=df.select_dtypes(include='object').columns
"""veri özeti:
    Toplam 4119 adet veri var
    19 değişken,
    9 adet numerik değişken,
    10 adet kategorik değişken
    Hiç eksik veri yok
"""
#numerik değişkenlerin dağılımı
# df.hist(column=numeric_cols,figsize=(10,10))
# plt.subplots_adjust(wspace=0.5,hspace=0.5)
# plt.show()
df['poutcome'].value_counts()
df['poutcome']=df['poutcome'].map({'failure':-1,'nonexistent':0,'success':1})
df['poutcome'].value_counts()
df['default'].value_counts()#ordinal
#poutcome,default,housing ve loan a ordinal diyebiliriz bunların dışındakiler nominal
df['default']=df['default'].map({'yes':-1,'no':1,'unknown':0})
df['housing']=df['housing'].map({'yes':-1,'no':1,'unknown':0})
df['loan']=df['loan'].map({'yes':-1,'no':1,'unknown':0})
#nominal değişkenler için one hot encoding
nominal=['job','marital','education','contact','month','day_of_week']
#OHE yapmadan önce datanın şekli
df.shape #(4119,20)
df=pd.get_dummies(df,columns=nominal)
#OHE sonrası
df.columns
df.shape #(4119,55)
#y target değişkeni de encode etmeliyiz
df['y']=df['y'].map({'yes':1,'no':0})
df.head()
#Feature Vector ve Target variable
X=df.drop(['y'],axis=1) #feature v
y=df['y'] #target v
#train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#X_train ve X_test şekilleri
X_train.shape
X_test.shape
#feature scaling
#X_train sütunlarını tut
cols=X_train.columns
#numerik sütunlar için feature scaling yapmamız lazım:verinin aynı skalaya gelmesi için ölçekliyoruz
X_train[numeric_cols] #StandardScaler yapıcaz.Hem X_train hem de X_test üzerinde yapıcaz
#standart scaler yarat
scaler=StandardScaler()
#X_train üzerinde fit ve tranform yap
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
#StandardScaler transform sonrası veri yapıları değişir.
type(X_train) #array
#geri pandas dataframe e çevirmemiz lazım
X_train=pd.DataFrame(X_train,columns=cols)
X_test=pd.DataFrame(X_test,columns=cols)
# print(X_train[cols])
#eksik değerleri kontrol etme ve analiz etme
# print(df.isnull().sum()) #poutcome da 142 tane eksik değer var
#eksik değerleri doldur
# imputer=SimpleImputer(strategy='mean')
#AdaBoost
#eksik değerlerle çalişmaz fit kısmında hata alındı.
#create adaboost classifier object
# abc=AdaBoostClassifier(n_estimators=400,learning_rate=1,random_state=0)
# #train adaboost classifier
# model_abc=abc.fit(X_train,y_train)
# #predict the response for test dataset
# y_pred_abc=model_abc.predict(X_test)
#roc_auc
# print('AdaBoost ROC-AUC score: {0:0.2f}'.format(roc_auc_score(y_test,y_pred_abc)))#0.58
# print(xgb.__version__)
#xgboost classifier objesi oluşturma
xgb=XGBClassifier(n_estimators=400,max_depth=6,learning_rate=1,random_state=0)
#Train xgboost classifier
model_xgb=xgb.fit(X_train,y_train)
#predict the response for test dataset
y_pred_xgb=model_xgb.predict(X_test)
#roc_auc
print('XGBoost ile ROC-AUC score:{0:0.2f}'.format(roc_auc_score(y_test,y_pred_xgb)))
