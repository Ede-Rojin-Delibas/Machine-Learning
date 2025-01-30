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
#Let's remove duration (it is not necessary)
df=df.drop('duration',axis=1)
#EDA(Exploratory Data Analysis)
# print(df.shape) #(4119,20)
# print(df.head())
#colon names
col_names=df.columns
# print(col_names)
#correlation of the target variable
print(df['y'].value_counts())
#percentage of yes-no classes
print(df['y'].value_counts() / float(len(df)))
"""y
    no:3668
    yes:451
    #percentages
    no:0.890
    yes:0.109 
    #comment: There is a big difference between classes(%89,%11)data is inbalanced.We need to be careful when we generate the results 
   """
# print(df.isnull().sum()) #check null datas
# print(len(df.columns))#total colon number
# print(df.select_dtypes(include=['int64','float64']).shape)    #number of numeric colons (4116,9)
numeric_cols=df.select_dtypes(include=['int64','float64']).columns
# print(numeric_cols)
"""Index(['age', 'campaign', 'pdays', 'previous', 'emp.var.rate',
       'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'],
      dtype='object')"""
#The number of categorical colons(object)
# print(df.select_dtypes(include='object').shape) #(4119,11)
cat_cols=df.select_dtypes(include='object').columns
# print(cat_cols)
'''Index(['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
       'month', 'day_of_week', 'poutcome', 'y'],
      dtype='object')'''
"""Summary of data:
    In total we have 4119 data
    19 variables
    9 numerical variables
    10 categorical variables
    1 y/target variables"""
#Distribution of numerical variables
df.hist(column=numeric_cols,figsize=(10,10))
plt.subplots_adjust(wspace=0.5,hspace=0.5)
# plt.show()
#The reason of doing that is to see the awkwardness of the data or to see outliers
# print(df['poutcome'].value_counts())
df['poutcome']=df['poutcome'].map({'failure':-1,'nonexistent':0,'success':1})
# print(df['poutcome'].value_counts())
# print(df['default'].value_counts())   #in default,Not having a current loan was interpreted as good.
df['housing']=df['housing'].map({'yes':-1,'unknown':0, 'no':1})
df['loan']=df['loan'].map({'yes':-1,'unknown':0, 'no':1})
# Returning the strings on 'default' column to numeric
df['default'] = df['default'].map({'yes': 1, 'no': 0, 'unknown': -1})
df['default'] = df['default'].fillna(-1)  # Let's fill the NaN values with -1 (it s up to you)
nominal=['job','marital','education','contact','month','day_of_week']
# print(df.shape)   #before one hot encoding the size/shape of df (4119,20)
df=pd.get_dummies(df,columns=nominal)
# print(df.shape)   #after OHE (4119,50)
#to encode the target variable y
df['y']=df['y'].map({'yes':1,'no':0})
# print(df.head())
numeric_cols_ohe=df.select_dtypes(include=['int64','float64']).columns
numeric_cols_ohe = numeric_cols_ohe.drop('y')
print(numeric_cols_ohe)
#FEATURE VECTOR AND TARGET VARIABLE
X=df.drop(['y'],axis=1)
y=df['y']
# print(X.shape) #(4119,54)
# print(y.shape)#(4119,)
#TRAIN - TEST SPLIT
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
# print(X_train.shape)#X_train şekli(3295, 54)
# print(X_test.shape )#X_test şekli(824, 54)
#Feature Scaling:bringing variables to the same size/scales
cols=X_train.columns #hold the X_train's
# print(cols)
# print(X_train[numeric_cols_ohe])
#We will train standardScaler on X_train. We will scale both X_train and X_test in the same way.
#create the Standart scale object
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)   #do the fit and transform on X_train
X_test=scaler.transform(X_test)     #do transform on X_test
#After StandardScaler transform data types will change
# print(type(X_train))
# print(X_train.dtypes) #to see the data types of the colons
# print(X_train.head()) #in default colon it has no values
#Scaling after error
#Determining the numeric colons
numeric_cols=X_train.select_dtypes(include=['int64','float64']).columns
#standardScaler
scaler=StandardScaler()
#fit the scale to numeric colons
X_train[numeric_cols]=scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols]=scaler.transform(X_test[numeric_cols])
#controlling
# print(X_train.head())
#Decision Tree Classifier(With Gini Index)
# Let's instantiate the DTC model as criterion gini index
clf_gini=DecisionTreeClassifier(criterion='gini',max_depth=3,random_state=0)
# fit the model
clf_gini.fit(X_train,y_train)
# Let's make a prediction with the model that we create with gini index
y_pred_gini=clf_gini.predict(X_test)
# Let's control the ROC_AUC score and evaluate the model performance
y_pred_gini_score=roc_auc_score(y_test,y_pred_gini)
print('Modelin Gini Index ile ROC-AUC skoru: {0:0.4f}'.format(y_pred_gini_score))
#Compare the train ve test roc_auc values
y_pred_train_gini=clf_gini.predict(X_train)
y_pred_train_gini_score=roc_auc_score(y_train,y_pred_train_gini)
print('Modelin gini index ile ROC-AUC Skoru: {0:0.4f}'.format(y_pred_train_gini_score))
#compare the train and test scores
print('Train set skoru:{:.4f}'.format(y_pred_train_gini_score))
print('Test set skoru:{:.4f}'.format(y_pred_gini_score))
#SONUÇ:train=%60, test=%58 There is no danger for overfit
#Visualize the Decision Tree
plt.figure(figsize=(24,16))
tree.plot_tree(clf_gini.fit(X_train,y_train))
# plt.show()
#Decision Tree Classifier with Entropy
#Decision T C Let's instantiate the model as criterion entropy index
clf_ent=DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=0)
#fit the model
clf_ent.fit(X_train,y_train)
y_pred_ent=clf_ent.predict(X_test)
#Let's control the ROC-AUC scores
y_pred_ent_score=roc_auc_score(y_test,y_pred_ent)
print('Modelin Entropy ile ROC-AUC skoru:{0:0.4f}'.format(y_pred_ent_score))
y_pred_train_ent=clf_ent.predict(X_train)
y_pred_train_ent_score=roc_auc_score(y_train,y_pred_train_ent)
print('Modelin Entropy ile ROC-AUC skoru:{0:0.4f}'.format(y_pred_train_ent_score))
#train and test scores
print('Train set skoru:{:.4f}'.format(y_pred_train_ent_score))
print('Test set skoru:{:.4f}'.format(y_pred_ent_score))
#comment :There is no danger for overfit
#Visualize the Decision Tree
plt.figure(figsize=(24,16))
tree.plot_tree(clf_ent.fit(X_train,y_train))
# plt.show()
#OVERFIT POOF(with gini index):incease the max_depth.
clf_gini=DecisionTreeClassifier(criterion='gini',max_depth=9,random_state=0)
clf_gini.fit(X_train,y_train)#modeli fit edelim
y_pred_gini=clf_gini.predict(X_test)
y_pred_gini_score=roc_auc_score(y_test,y_pred_gini)
# print('Modelin gini index ile ROC-AUC Skoru: {0:0.4f}'.format(y_pred_gini_score))
y_pred_train_gini=clf_gini.predict(X_train)
y_pred_train_gini_score=roc_auc_score(y_train,y_pred_train_gini)
# print('Modelin gini index ile ROC-AUC Skoru: {0:0.4f}'.format(y_pred_train_gini_score))
#Let's compare the train and test scores
print('Train set skoru:{:.4f}'.format(y_pred_train_gini_score))
print('Test set skoru:{:.4f}'.format(y_pred_gini_score))
#Results:0.7388, 0.5859 yani %74, %59 :OVERFIT
#Visualization
plt.figure(figsize=(24,16))
tree.plot_tree(clf_gini.fit(X_train,y_train))
plt.show()












