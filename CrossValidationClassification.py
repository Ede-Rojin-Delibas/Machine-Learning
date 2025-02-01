#K-Fold Cross Validation - Classification Instance
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold,cross_val_score
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
train_data=pd.read_csv('14_Model_Secimi/data/titanic/train.csv')
# print(train_data.head())
train_data.dropna(axis=0,subset=['Survived'],inplace=True)
y=train_data.Survived
# print(y)
#drop y from train data
train_data.drop(['Survived'],axis=1,inplace=True)
#drop age, it has null variables
train_data.drop(['Age'],axis=1,inplace=True)
#just choose numerical columns
numeric_cols=[cname for cname in train_data.columns if train_data[cname].dtype in ['int64','float64'] ]
X=train_data[numeric_cols].copy()
# print(X.head())
print("Train datanın şekli: {} ve sonuç değişkenin şekli :{}".format(X.shape,y.shape))
# print(pd.concat([X,y],axis=1).head())
#model with stratified kfold
kf=StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
cnt=1
for train_index,test_index in kf.split(X,y):
    print(f'Fold:{cnt},Train set:{len(train_index)},Test set:{len(test_index)}')
    cnt+=1
#run the cross_val_score logistic regresyon model
#take CV score
#skorlama yöntemi -> accuracy
score=cross_val_score(LogisticRegression(random_state=42),X,y,cv=kf,scoring="accuracy")
#print(f"Her bir fold'un skoru:{score}")
#print(f"Ortalama Skoru:{'{:.2f}'.format(score.mean())}") #0.68
#HYPERPARAMETER TUNING: Let's find out which solver (solution algorithm) is the best by trying different parameters.
solvers=['newton-cg','lbfgs','liblinear','sag','saga']
#mean score for each solver
#make max_iter 4000
for solver in solvers:
    score=cross_val_score(LogisticRegression(solver=solver,max_iter=4000,random_state=42),X,y,cv=kf,scoring="accuracy")
    print(f"Ortalama Skor:({solver}):{'{:.3f}'.format(score.mean())}")
#best result is on newton ,lbfgs and liblinear
## GRID SEARCH ##
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2 ,random_state=42)
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)
#Let's solve with knn
#take k=3 (randomly)
neigh=KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train)

y_pred=neigh.predict(X_test)
# print(y_pred)
#if probability of prediction is above 0.5 then that class will choose
y_predict_proba=neigh.predict_proba(X_test)
# print(y_predict_proba)
#Let's look prediction quality for F1 score
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score
conf_mat=confusion_matrix(y_test,y_pred)
# print(conf_mat)
#accuracy
accuracy=accuracy_score(y_test,y_pred)
#print('Sklearn Accuracy Score:{:.4f}'.format(accuracy)) #0.6145
#f1 score
f1=f1_score(y_test,y_pred)
#print('F1 score:%{:.2f}'.format(f1*100)) # %46.51
parameters={
    'n_neighbors':[1,3,5,7,9,11,13]
}
gsc=GridSearchCV(estimator=KNeighborsClassifier(),
                 param_grid=parameters,
                 cv=5,
                 verbose=1, #sonuçları bana yaz demek
                 scoring='accuracy')
gsc.fit(X_train,y_train)
#grid search results
# print(f"en iyi Hyperparameter:{gsc.best_params_}") #13
# print(f"en iyi score:{gsc.best_score_}") # 0.630
#results in detail
# print("Detaylı GridSearchCV sonucu:")
gsc_result=pd.DataFrame(gsc.cv_results_).sort_values('mean_test_score',ascending=False)
#to see it basically
# print(gsc_result[['param_n_neighbors','mean_test_score','rank_test_score']])
#Let's run the best hyperparameter values which is k=13 for once.
neigh_final=KNeighborsClassifier(n_neighbors=13)
neigh_final.fit(X_train,y_train)
y_pred_final=neigh_final.predict(X_test)
accuracy_final=accuracy_score(y_test,y_pred_final)
#print("Final Score:{:.4f}".format(accuracy_final)) #0.6425
f1_final=f1_score(y_test,y_pred_final)
#print('F1 score final:%{:.2f}'.format(f1_final*100)) # %42.86
from sklearn.model_selection import StratifiedKFold,cross_val_score
kf=StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
score=cross_val_score(KNeighborsClassifier(),X_train,y_train,cv=kf,scoring="accuracy")
# print(score.mean()) #0.6333
#Do the same treatment for random search
#n_iter -> how many parametre will be?
rsc=RandomizedSearchCV(estimator=KNeighborsClassifier(),
                       param_distributions=parameters,
                       n_iter=3,
                       cv=5,
                       verbose=1,
                       scoring='accuracy')
rsc.fit(X_train,y_train)
print("En iyi parametreler:{}".format(rsc.best_params_)) #11
print("En iyi score:{}".format(rsc.best_score_)) #0.6291












