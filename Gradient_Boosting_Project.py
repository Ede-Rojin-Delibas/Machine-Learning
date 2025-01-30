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
#Remove the duration column (it 's on dataset explanation.)
df=df.drop('duration',axis=1)
#EDA
print(df.shape) #(4119,20)
col_names=df.columns
#Distribution of target variable
df['y'].value_counts()
df.info()
df.isnull().sum()
len(df.columns)
df.select_dtypes(include=['int64','float64']).shape
numeric_cols=df.select_dtypes(include=['int64','float64']).columns
cat_cols=df.select_dtypes(include='object').columns
"""summary of dataset:
    we have 4119 total data
    19 variables,
    9  numeric variables,
    10 categorical variables
    There is no null or NaN value on dataset
"""
#Distribution of numerical variables
df.hist(column=numeric_cols,figsize=(10,10))
plt.subplots_adjust(wspace=0.5,hspace=0.5)
# plt.show()
df['poutcome'].value_counts()
df['poutcome']=df['poutcome'].map({'failure':-1,'nonexistent':0,'success':1})
df['poutcome'].value_counts()
df['default'].value_counts()#ordinal
#we can say that poutcome,default,housing and loan are ordinal, other ones are nominal
df['default']=df['default'].map({'yes':-1,'no':1,'unknown':0})
df['housing']=df['housing'].map({'yes':-1,'no':1,'unknown':0})
df['loan']=df['loan'].map({'yes':-1,'no':1,'unknown':0})
#one hot encoding for nominal variables
nominal=['job','marital','education','contact','month','day_of_week']
#The shape of data before OHE
df.shape #(4119,20)
df=pd.get_dummies(df,columns=nominal)
#after OHE
df.columns
df.shape #(4119,55)
#We need to encode y the target variable
df['y']=df['y'].map({'yes':1,'no':0})
df.head()
#Feature Vector and Target variable
X=df.drop(['y'],axis=1) #feature v
y=df['y'] #target v
#train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#shapes of X_train and X_test
X_train.shape
X_test.shape
#feature scaling
#hold the X_train
cols=X_train.columns
#We need to do feature scaling for numerical colons : We scale the data to reach the same scale
X_train[numeric_cols] #StandardScaler for both X_train and X_test
scaler=StandardScaler()
#do fit and tranform on X_train
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
#After transform data types are change
type(X_train) #array
#Let's return to pandas dataframe
X_train=pd.DataFrame(X_train,columns=cols)
X_test=pd.DataFrame(X_test,columns=cols)
# print(X_train[cols])
#control the null values and analyz it
print(df.isnull().sum()) #we have 142 null values in poutcome
#fill the null values
imputer=SimpleImputer(strategy='mean')
#AdaBoost
#an error occured in fit part; it wont work with null variables
#create adaboost classifier object
abc=AdaBoostClassifier(n_estimators=400,learning_rate=1,random_state=0)
#train adaboost classifier
model_abc=abc.fit(X_train,y_train)
# predict the response for test dataset
y_pred_abc=model_abc.predict(X_test)
#roc_auc
print('AdaBoost ROC-AUC score: {0:0.2f}'.format(roc_auc_score(y_test,y_pred_abc)))#0.58
print(xgb.__version__)
#generate xgboost classifier object
xgb=XGBClassifier(n_estimators=400,max_depth=6,learning_rate=1,random_state=0)
#Train xgboost classifier
model_xgb=xgb.fit(X_train,y_train)
#predict the response for test dataset
y_pred_xgb=model_xgb.predict(X_test)
#roc_auc
print('XGBoost ile ROC-AUC score:{0:0.2f}'.format(roc_auc_score(y_test,y_pred_xgb)))
