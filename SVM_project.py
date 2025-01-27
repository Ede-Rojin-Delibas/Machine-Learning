import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("16_SVM/data/bank-additional.csv",sep=';')
df=df.drop('duration',axis=1)
#EDA
# print(df.head())
# print(df.shape) #4119,20
col_names=df.columns
#print(col_names)
#target variable ın dağılımı
#print(df['y'].value_counts())
'''no:3668
   yes:451'''
#sınıfların yes-no yüzde dağılımları
#no: %89 ; yes: %11 -> veri inbalanced
# print(df.info())
#eksik data kontrolü
# print(df.isnull().sum())
#toplam kolon sayısı
# print(len(df.columns)) # 20
#numerik kolonların adedi
#print(df.select_dtypes(include=['int64','float64']).shape) #(4119,9)
#numerik kolonlar
numeric_cols=df.select_dtypes(include=['int64','float64']).columns
#print(numeric_cols)
#kategorik kolonların (object) adedi
#print(df.select_dtypes(include=['object']).shape) #(4119,11)
cat_cols=df.select_dtypes(include='object').columns
#print(cat_cols)
#verinin özeti:4119 adet veri:19 adet değişken, 9 adet numerik değişken, 10 adet kategorik değişken
#numerik değişkenlerin dağılımı
# df.hist(column=numeric_cols,figsize=(10,10))
# plt.subplots_adjust(wspace=0.5,hspace=0.5)
# plt.show()
#ordinal değişkenler
df['poutcome'].value_counts()
df['poutcome']=df['poutcome'].map({'failure':-1,'nonexistent':0,'success':1})
#print(df['poutcome'].value_counts())
df['default'].value_counts()
df['default']=df['default'].map({'yes':-1,'unknown':0,'no':1})
# print(df['default'].value_counts())
df['housing']=df['housing'].map({'yes':-1,'unknown':0,'no':1})
df['loan']=df['loan'].map({'yes':-1,'unknown':0,'no':1})
#Nominal değişkenler:poutcome,default,housing,loan dışındakiler nominaldir:sıranın bir önemi yok ve one hot encoding yapılacaktır
nominal=['job','marital','education','contact','month','day_of_week']
df=pd.get_dummies(df,columns=nominal)
#print(df.shape) #ohe sonrası:(4119,55)
#y nin encode edilmesi
df['y']=df['y'].map({'yes':1,'no':0})
# print(df.head())
X=df.drop(['y'],axis=1)
y=df['y']
#print(X.shape) #(4119,54)
#print(y.shape) #(4119,)
#Train - Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
# print(X_train.shape) #(3295, 54)
# print(X_test.shape) #(824, 54)
#feature scaling
cols=X_train.columns
X_train[numeric_cols]=X_train[numeric_cols]
# print(X_train[numeric_cols])
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
# print(type(X_train)) #numpy.ndarray
X_train=pd.DataFrame(X_train,columns=cols)
X_test=pd.DataFrame(X_test,columns=cols)
# print(X_train[cols])
#SVC Classifier
from sklearn.svm import SVC
#accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
#ROC-AUC
from sklearn.metrics import roc_auc_score
#SVC yi default hyperparameter lar ile başlat
# svc=SVC()
#classifier ı fit et
# svc.fit(X_train,y_train)
# y_pred=svc.predict(X_test)
#accuracy yi yazdır
#print('Default hyperparameterlar ile accuracy score : {0:0.2f}'.format(accuracy_score(y_test,y_pred))) # 0.91
#classification report
#print('Classification report: {} '.format(classification_report(y_test,y_pred)))
#ROC-AUC:c nin default değeri 1
#print('Default Hyperparameter ile ROC-AUC Score: {0:0.2f}'.format(roc_auc_score(y_test,y_pred))) #0.58
#SVM with RBF Kernel & C=100
# svc=SVC(C=100.0)
# svc.fit(X_train,y_train)
# y_pred=svc.predict(X_test)
#print("RBF kernel ve C=100 ile ROC-AUC Score: {0:0.2f}".format(roc_auc_score(y_test,y_pred))) #0.65
#yine çok güvenilir değil grid search ile hyperparameter tuning yapılmalı
#en iyi hyperparameterları bulmanın yolu:GridSearch Cross Validation
from sklearn.model_selection import GridSearchCV
svc_grid=SVC()
parameters=[{'C':[0.1,1,10,100,1000],'gamma':['scale','auto',0.001,0.01,0.1,0.9],'kernel':['rbf']}]
#n_jobs=4 -> 4 işlemci core unda da kullanılsın
#scoring =>'balanced_accuracy', 'f1','precision','recall','roc_auc' şeklinde de denenebilir
#veri inbalanced olduğu için balanced_accuracy denendi.
grid_search=GridSearchCV(estimator=svc_grid,param_grid=parameters,
                         scoring='balanced_accuracy',
                         cv=5,n_jobs=4,verbose=1)
grid_search.fit(X_train,y_train)
#print("GridSearch CV best score:{:.4f}\n\n".format(grid_search.best_score_)) #0.6430
#print("GridSearch CV best params:{}\n\n".format(grid_search.best_params_))
#c=1000, gamma=0.001, kernel=rbf
#print("GridSearch CV best estimator:{}\n\n".format(grid_search.best_estimator_))
#c=1000, gamma=0.001

## en iyi sonucu veren parametreler ##
svc_best=SVC(C=1000.0,gamma=0.001,kernel='rbf')
svc_best.fit(X_train,y_train)
y_pred_best=svc_best.predict(X_test)
#ROC- AUC
#print('ROC - AUC score:{:.2f}'.format(roc_auc_score(y_test,y_pred_best))) #0.62
# print(svc_best.get_params())
#sonuçlar:{'C': 1000.0, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': 0.0,
#'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 0.001, 'kernel': 'rbf', 'max_iter': -1,
#'probability': False, 'random_state': None, 'shrinking': True, 'tol': 0.001, 'verbose': False}








