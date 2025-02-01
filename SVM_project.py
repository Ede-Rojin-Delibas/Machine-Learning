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
#print(df['y'].value_counts())
'''no:3668
   yes:451'''
#yes-no percentage distributions of classes
#no: %89 ; yes: %11 -> data is inbalanced
print(df.info())
print(df.isnull().sum())
# print(len(df.columns)) # 20
#print(df.select_dtypes(include=['int64','float64']).shape) #(4119,9)
numeric_cols=df.select_dtypes(include=['int64','float64']).columns
#print(numeric_cols)
print(df.select_dtypes(include=['object']).shape) #(4119,11)
cat_cols=df.select_dtypes(include='object').columns
#print(cat_cols)
#Summary of data: 4119 data: 19 variables, 9 numerical variables, 10 categorical variables
df.hist(column=numeric_cols,figsize=(10,10))
plt.subplots_adjust(wspace=0.5,hspace=0.5)
# plt.show()
#ordinal variables
df['poutcome'].value_counts()
df['poutcome']=df['poutcome'].map({'failure':-1,'nonexistent':0,'success':1})
#print(df['poutcome'].value_counts())
df['default'].value_counts()
df['default']=df['default'].map({'yes':-1,'unknown':0,'no':1})
# print(df['default'].value_counts())
df['housing']=df['housing'].map({'yes':-1,'unknown':0,'no':1})
df['loan']=df['loan'].map({'yes':-1,'unknown':0,'no':1})
#Nominal variables: Except for poutcome, default, housing, loan, they are nominal:
# the order does not matter and one hot encoding will be done
nominal=['job','marital','education','contact','month','day_of_week']
df=pd.get_dummies(df,columns=nominal)
#print(df.shape) #ohe sonrası:(4119,55)
#encoding y
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
#Initialize SVC with default hyperparameters
svc=SVC()
#fit the classifier
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Default hyperparameterlar ile accuracy score : {0:0.2f}'.format(accuracy_score(y_test,y_pred))) # 0.91
#classification report
print('Classification report: {} '.format(classification_report(y_test,y_pred)))
#ROC-AUC:default value of c is 1
print('Default Hyperparameter ile ROC-AUC Score: {0:0.2f}'.format(roc_auc_score(y_test,y_pred))) #0.58
#SVM with RBF Kernel & C=100
svc=SVC(C=100.0)
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print("RBF kernel ve C=100 ile ROC-AUC Score: {0:0.2f}".format(roc_auc_score(y_test,y_pred))) #0.65
#Again, it is not very reliable, hyperparameter tuning should be done with grid search.
#The best way of finding hyperparameters:GridSearch Cross Validation
from sklearn.model_selection import GridSearchCV
svc_grid=SVC()
parameters=[{'C':[0.1,1,10,100,1000],'gamma':['scale','auto',0.001,0.01,0.1,0.9],'kernel':['rbf']}]
#n_jobs=4 -> 4 CPU core will use
#scoring =>'balanced_accuracy', 'f1','precision','recall','roc_auc'
#Data is inbalanced so, balanced_accuracy.
grid_search=GridSearchCV(estimator=svc_grid,param_grid=parameters,
                         scoring='balanced_accuracy',
                         cv=5,n_jobs=4,verbose=1)
grid_search.fit(X_train,y_train)
print("GridSearch CV best score:{:.4f}\n\n".format(grid_search.best_score_)) #0.6430
print("GridSearch CV best params:{}\n\n".format(grid_search.best_params_))
#c=1000, gamma=0.001, kernel=rbf
print("GridSearch CV best estimator:{}\n\n".format(grid_search.best_estimator_))
#c=1000, gamma=0.001

## Parameters that give the best results ##
svc_best=SVC(C=1000.0,gamma=0.001,kernel='rbf')
svc_best.fit(X_train,y_train)
y_pred_best=svc_best.predict(X_test)
#ROC- AUC
print('ROC - AUC score:{:.2f}'.format(roc_auc_score(y_test,y_pred_best))) #0.62
print(svc_best.get_params())
#Results:{'C': 1000.0, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': 0.0,
#'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 0.001, 'kernel': 'rbf', 'max_iter': -1,
#'probability': False, 'random_state': None, 'shrinking': True, 'tol': 0.001, 'verbose': False}








