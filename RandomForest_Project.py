import random
import time
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold,cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV

train_df=pd.read_csv('18_Random_Forests/data/train.csv')
# print(train_df.head())
# print(train_df.isnull().sum())
#null values for age
print(train_df['Age'].isnull().sum() / train_df.shape[0] *100) #19.86
ax=train_df['Age'].hist(bins=15,density=True,stacked=True,alpha=0.7)
train_df['Age'].plot(kind='density')
ax.set(xlabel='Age')
plt.xlim(0,90)
plt.grid()
# plt.show()
train_df['Age'].mean(skipna=True)
train_df['Age'].median(skipna=True)
#null values for cabin
print(train_df['Cabin'].isnull().sum() / train_df.shape[0] * 100) #77.10 : %77 of passenger's cabin numbers are null
#decision:%77 rate is a big rate so I will remove this column
# Embarked(gemiye binmek) null data
#print(train_df['Embarked'].isnull().sum() / train_df.shape[0] * 100) #0.22 so just 2 variable
print("Yolcuların hangi limandan bindikleri % olarak :(C=Cherbourg, Q=Queenstown, S=Southampton):")
print(train_df['Embarked'].value_counts() / train_df.shape[0] * 100)
"""Embarked
S    72.278339
C    18.855219
Q     8.641975 """
# sns.countplot(x='Embarked',data=train_df,palette='Set1')
# plt.show()
#Let's find the most boarded port-> idmax()
print("En fazla binilen liman:" ,train_df['Embarked'].value_counts().idxmax()) #S , eksik veriler S ile doldurulacak
#decision: Age ->We will fill in the missing values ​​with the median method(median=28)
#Embarked ->I will fill the null values with 'S'
train_data=train_df.copy()
train_data['Age'].fillna(train_df['Age'].median(skipna=True),inplace=True)
# train_data['Embarked'].fillna(train_df['Embarked'].value_counts().idxmax(),inplace=True)
train_data.drop('Cabin',axis=1, inplace=True)
# print(train_data.isnull().sum())
#redundant variable analysis:SibSp,Parch(It would be healthy to collect these two variables under a single variable.)
train_data['YalnizSeyahat']=np.where(train_data['SibSp']+train_data['Parch'] > 0, 0 , 1)
train_data.drop(['SibSp','Parch'],axis=1,inplace=True)
# print(train_data.head())
train_data=pd.get_dummies(train_data,columns=["Pclass","Embarked","Sex"],drop_first=True)
# print(train_data.head())
train_data.drop('Name',axis=1,inplace=True)
train_data.drop('Ticket',axis=1,inplace=True)
train_data.drop('PassengerId',axis=1,inplace=True)
#EDA
#print(train_data.shape) #(891,9)
print("Train data içindeki toplam veri adedi: ",train_data.shape[0]) #891
col_names=train_data.columns
# print(col_names)
#EDA for age
# plt.figure(figsize=(15,8))
#hayatta kalanlar -> Survived==1
ax=sns.kdeplot(train_data['Age'][train_data.Survived ==1],color="green",shade=True)
#ölenler -> Survived==0
sns.kdeplot(train_data['Age'][train_data.Survived ==0],color="red",shade=True)
plt.legend(['Survived','Died'])
plt.title('Yaş(Age) için hayatta kalma ve ölüm yoğunluk grafiği')
ax.set(xlabel='Age')
plt.xlim(-10,85)
# plt.show()
#EDA for fare (ücret)
plt.figure(figsize=(15,8))
ax=sns.kdeplot(train_data['Fare'][train_data.Survived ==1],color="green",shade=True)
sns.kdeplot(train_data['Fare'][train_data.Survived ==0],color="red",shade=True)
plt.legend(['Survived','Died'])
plt.title('Density plot of Fare for Surviving population and deceased population')
ax.set(xlabel='Fare')
plt.xlim(-20,200)
# plt.show()
#EDA for Passanger class (yolcu sınıfı)
sns.barplot(x='Pclass', y='Survived', data=train_df, color='green')
# plt.show()
#EDA for with family or alone travellers
# sns.barplot(x='YalnizSeyahat', y='Survived', data=train_data, color='green')
# plt.show()
#EDA-Sex(Cinsiyet)
# sns.barplot(x='Sex', y='Survived', data=train_df, color='green')
# plt.show()
#input - output ayrımı
y=train_data['Survived']
train_data.drop('Survived',axis=1,inplace=True)
#feature scaling (boyutlama)
# print(train_data.describe())
cols=train_data.columns
scaler=MinMaxScaler()
train_data=scaler.fit_transform(train_data)
# print(type(train_data)) #numpy.ndarray
train_data=pd.DataFrame(train_data,columns=[cols])
# print(train_data.head())
X_train,X_test,y_train,y_test=train_test_split(train_data,y,test_size=0.2,random_state=2)
# print(X_train.shape) #(712,8)
#print(X_test.shape) # (179,8)
fit_rf=RandomForestClassifier(random_state=42)
#for hyperparameter optimization I'll use GridSearchCV
#for random state make seed constant
np.random.seed(42)
#start timer
start=time.time()
#parametre grid
param_dist={
    'n_estimators':[100,200,400],
    'max_features':['auto','sqrt','log2',None],
    'max_depth':[2,3,4],
    'criterion':['gini','entropy']
}
#Run 4 parallel processors with n_jobs=4
cv_rf=GridSearchCV(fit_rf,param_grid=param_dist,cv=5,n_jobs=4)
cv_rf.fit(X_train,y_train)
print("GridSearch ile en iyi parametre bulma: \n",cv_rf.best_params_)
# #end up the timer
# end=time.time()
# print("Grid Search için geçen zaman:{0:.2f}".format(end-start))
"""GridSearch ile en iyi parametre bulma: 
 {'criterion': 'entropy', 'max_depth': 4, 'max_features': 'log2', 'n_estimators': 400}
Grid Search için geçen zaman:86.46"""
fit_rf.set_params(n_estimators=400,max_features='log2',max_depth=4,criterion='entropy',random_state=42)
fit_rf.fit(X_train,y_train)
#find the importance of variables
a=pd.concat((pd.DataFrame(X_train.columns,columns=['feature']),
           pd.DataFrame(fit_rf.feature_importances_,columns=['importance'])),
           axis=1).sort_values(by='importance',ascending=False)
#print(a) #sex is almost %50 important
y_pred=fit_rf.predict(X_test)
#accuracy
#print("Modelin accuracy skoru:{0:0.4f}".format(accuracy_score(y_test,y_pred))) #0.7877
#classification report
print(classification_report(y_test,y_pred))



