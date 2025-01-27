import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_df=pd.read_csv('12_Logistic_Regression/data/train.csv')
#train data içindeki eksik değerlere bakalım
# print(train_df.isnull().sum())
#Age için eksik değerler
# train_df['Age'].isnull().sum() / train_df.shape[0] * 100 #19.86 yani yolcuların yaklaşık %20 sinin age verisi eksik
# ax=train_df['Age'].hist(bins=15,density=True,stacked=True,alpha=0.7)
# train_df['Age'].plot(kind='density')
# ax.set_xlabel('Age')
# plt.xlim(0,90)
# plt.grid()
# plt.show()
#mean -> ortalama
#skipna -> eksik verileri es geç (skip)
#print(train_df['Age'].mean(skipna=True)) # 29.699
#median -> ortanca
#skipna -> eksik verileri es geç (skip)
#print(train_df['Age'].median(skipna=True)) #28.0
#kabin eksik değerler
# print(train_df['Cabin'].isnull().sum() / train_df.shape[0] * 100) # %77.10
"""yolcuların %77 sinin cabin numarası bilgisi eksik. Burada kritik bir karar verilmesi gerekiyor.Eldeki
% 23 ile %77 nin değerlerini mi doldurmalıyız yoksa bu sütunu tamamen çıkarmalı mıyız?
KARAR:%77 lik bir eksik veri oranı çok yüksek olduğu için en doğrusu bu sütunu çıkarmak olacaktır."""
#embarked : gemiye binmek için eksik veri oranı
# print(train_df['Embarked'].isnull().sum() / train_df.shape[0] * 100) # %0.2 çok düşük bir oran
#yolcuların bindikleri yerler
# print("Yolcuların hangi limandan bindikleri % olarak: (C=Cherbourg, Q=Queenstown, S=Southampton):")
# print(train_df['Embarked'].value_counts() / train_df.shape[0] * 100)
#grafik ile de görelim
# sns.countplot(x='Embarked',data=train_df,palette='Set1')
# plt.show()
#en fazla binilen limanı bulalım ->idmax()
# print("En fazla binilen liman: ",train_df['Embarked'].value_counts().idxmax()) :S
train_data=train_df.copy()
# train_data['Age'].fillna(train_df['Age'].median(skipna=True),inplace=True)
# train_data['Embarked'].fillna(train_df['Embarked'].value_counts().idxmax(),inplace=True)
# train_data.drop('Cabin',axis=1,inplace=True)
#bu ayarlamalardan sonra train datada eksik veri var mı diye kontrol edelim
# print(train_data.isnull().sum())
# print(train_data.head())
train_data['YalnizSeyahat']=np.where(train_data['SibSp']+train_data['Parch'] > 0, 0 , 1)
train_data.drop(['SibSp','Parch'],axis=1,inplace=True)
# print(train_data.head())
#kategorik sütunların encode edilmesi
train_data=pd.get_dummies(train_data,columns=['Pclass','Embarked','Sex'],drop_first=True)
# print(train_data.head())
train_data.drop('Name',axis=1,inplace=True)
train_data.drop('Ticket',axis=1,inplace=True)
train_data.drop('PassengerId',axis=1,inplace=True)
#EDA
# print(train_data.shape) #(891,10)
col_names=train_data.columns
# print(col_names)
# plt.figure(figsize=(15,8))
#hayatta kalanlar > survived==1
# ax=sns.kdeplot(train_data['Age'][train_data.Survived==1],color='green',shade=True)
#ölenler > Survived==0
# sns.kdeplot(train_data['Age'][train_data.Survived==0],color='red',shade=True)
# plt.legend(['Survived','Died'])
# plt.title('Yaş(Age) için Hayatta kalma ve ölüm yoğunluk grafiği')
# ax.set(xlabel='Age')
# plt.xlim(-10,85)
# plt.show()
#Fare(ücret) için EDA
# plt.figure(figsize=(15,8))
# ax=sns.kdeplot(train_data['Fare'][train_data.Survived==1],color='green',shade=True)
# sns.kdeplot(train_data['Fare'][train_data.Survived==0],color='red',shade=True)
# plt.legend(['Survived','Died'])
# plt.title('Density plot of Fare for Surviving population and deceased population')
# ax.set(xlabel='Fare')
# plt.xlim(-20,200)
# plt.show()
#Passenger Class içn EDA
# plt.figure(figsize=(10,6))
# sns.barplot('Pclass','Survived',data=train_df,color='green')
# plt.show()
#input-output ayrımı
y=train_data['Survived']
train_data.drop('Survived',axis=1,inplace=True)
#Feature Scaling
# print(train_data.describe())
cols=train_data.columns
# print(cols)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
#cabin varlığını binary göstermek
train_data['Has_Cabin'] = train_data['Cabin'].notnull().astype(int)  # Kabin varsa 1, yoksa 0
train_data.drop('Cabin', axis=1, inplace=True)  # Orijinal 'Cabin' sütununu kaldırıyoruz.
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)  # Ortanca ile doldur
train_data=scaler.fit_transform(train_data)
# print(type(train_data))
# print(train_data.dtypes)  # Tüm sütunların veri tiplerini görüntüleyin
# print(train_data.head())  # İlk birkaç satırı görüntüleyin
# print(train_data['Cabin'].unique())  # Tüm benzersiz değerleri alır
#train_datanın veri tipini yeniden data frame yapma
train_data=pd.DataFrame(train_data,columns=[cols])
# print(train_data.head())
#test train split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(train_data,y,test_size=0.2,random_state=2)
# print(X_train.shape) #712,9
#model oluşturma
#veri setini kontrol etme
# print(train_data.isnull().sum())  # Sütunlardaki eksik değerlerin sayısını görüntüleyin:177
# print(train_data.isnull().mean() * 100)  # Eksik değer oranlarını yüzdelik olarak görün
# train_data['Age'].fillna(train_data['Age'].median(), inplace=True)  # Ortanca ile doldur
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(solver='liblinear',random_state=0)
logreg.fit(X_train,y_train)
# #Tahmin
y_pred=logreg.predict(X_test)
#print(y_pred)
logreg.predict_proba(X_test)[:,0] #0 sınıfı ölüm
logreg.predict_proba(X_test)[:,1] #1 sınıfı hayatta kalma
#tahmin kalitesini ölçmek
from sklearn.metrics import accuracy_score
#print("Modelin Accuracy Score: {:0.4f}".format(accuracy_score(y_test,y_pred) )) #%79.89
#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
TN,FP,FN,TP=cm.ravel() #confusion matrixten dört temel değeri çıkarır.
# print(cm)
y_test_df=pd.DataFrame(y_test,columns=['Survived'])
# print(y_test_df.value_counts())
'''0:100
   1:79'''
#sınıf oranları
sinif_0_orani= 100/179
# print('Sınıf 0 (dead) oranı : %{:.0f}'.format(sinif_0_orani*100)) # %56
sinif_1_orani= 79/179
# print('Sınıf 1 (yaşayan) oranı: %{:.0f}'.format(sinif_1_orani*100)) # %44
#null accuracy hesaplanması
null_accuracy=100 / (100+79)
# print(null_accuracy) #0.5586
#accuracy yi elle hesaplama
#positive=1=>survived
#negative=0=>Dead
# manuel_accuracy=(TP+TN)/(TP+FP+FN+TN)
# print('Manuel Hesaplanan Accuracy Score: {:.4f}'.format(manuel_accuracy)) #0.7989
# #accuracy yi sklearn ile hesaplama
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
# print('Sklearn ile Hesaplanan Accuracy Score: {:.4f}'.format(accuracy)) #0.7989
#precision
precision_manuel=TP / (TP + FP)
# print("Precision Manuel: %{:.2f}".format(precision_manuel * 100)) #%84.13
from sklearn.metrics import precision_score
precision=precision_score(y_test,y_pred)
#print("Precision Score: {:.2f}".format(precision * 100)) #84.13
#Recall manuel
recall_manuel=TP / (TP + FN)
# print("Recall Manuel:%{:.2f}".format(recall_manuel * 100)) #%67.09
from sklearn.metrics import recall_score
recall=recall_score(y_test,y_pred)
# print("Recall Score: {:.2f}".format(recall * 100)) #%67.09
#F1 Score
f1_manuel=2 * (precision * recall) /(precision + recall)
# print("F1 Score Manuel: %{:.2f}".format(f1_manuel * 100)) #74.65
from sklearn.metrics import f1_score
f1=f1_score(y_test,y_pred)
# print("F1 Score: %{:.2f}".format(f1 * 100)) #%74.65
#ROC-AUC CURVE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
roc_auc=roc_auc_score(y_test,y_pred)
# print("ROC-AUC Score: %{:.2f}".format(roc_auc * 100)) #%78.54
# fpr,tpr,thresholds=roc_curve(y_test,y_pred)
# plt.plot(fpr,tpr,label='ROC-CURVE')
# plt.plot([0,1],[0,1],'k--',label='Random Guess')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC-AUC Curve')
# plt.xlim([0,1])
# plt.ylim([0,1])
# plt.legend(loc='lower right')
# plt.show()
#logloss
from sklearn.metrics import log_loss
log_loss=log_loss(y_test,y_pred)
print("Log-Loss Değeri:{:.2f}".format(log_loss))#7.25









