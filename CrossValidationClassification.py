#K-Fold Cross Validation - Classification Örneği
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold,cross_val_score
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
train_data=pd.read_csv('14_Model_Secimi/data/titanic/train.csv')
# print(train_data.head())
#eksik verili satırları sil
train_data.dropna(axis=0,subset=['Survived'],inplace=True)
#sonuç değişkeni
y=train_data.Survived
# print(y)
#train datadan y yi çıkar
train_data.drop(['Survived'],axis=1,inplace=True)
#içinde null değerler olan age sütununu sil
train_data.drop(['Age'],axis=1,inplace=True)
#sadece numerik sütunları seç
numeric_cols=[cname for cname in train_data.columns if train_data[cname].dtype in ['int64','float64'] ]
X=train_data[numeric_cols].copy()
# print(X.head())
# print("Train datanın şekli: {} ve sonuç değişkenin şekli :{}".format(X.shape,y.shape))
#ilk 5 train datasını göster
# print(pd.concat([X,y],axis=1).head())
#stratified kfold ile model
kf=StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
cnt=1
#split ile indexleri
# for train_index,test_index in kf.split(X,y):
#     print(f'Fold:{cnt},Train set:{len(train_index)},Test set:{len(test_index)}')
#     cnt+=1
#cross_val_score logistic regresyon modeliyle çalışır
#CV skorunu alalım
#skorlama yöntemi -> accuracy
score=cross_val_score(LogisticRegression(random_state=42),X,y,cv=kf,scoring="accuracy")
#print(f"Her bir fold'un skoru:{score}")
#print(f"Ortalama Skoru:{'{:.2f}'.format(score.mean())}") #0.68
#HYPERPARAMETER TUNING:Değişik parametreler deneyerek hangi solver'in (çözüm algoritmasının)en iyi olduğunu bulalım
#logistic regresyonun bütün solver larını deneyelim
solvers=['newton-cg','lbfgs','liblinear','sag','saga']
#her bir solver için ortalama score hesaplayalım
#max_iter i 4000 verdik
# for solver in solvers:
#     score=cross_val_score(LogisticRegression(solver=solver,max_iter=4000,random_state=42),X,y,cv=kf,scoring="accuracy")
#     print(f"Ortalama Skor:({solver}):{'{:.3f}'.format(score.mean())}")
#en iyi sonuç veren newton ,lbfgs ve liblinear
## GRID SEARCH ##
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2 ,random_state=42)
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)
# knn ile çözelim
#rastgele olarak k=3 alalım
neigh=KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train)
#tahmin yapalım
y_pred=neigh.predict(X_test)
# print(y_pred)
#tahmin olasılıkları 0.5 in üstü ise o sınıf seçilir
y_predict_proba=neigh.predict_proba(X_test)
# print(y_predict_proba)
#tahmin kalitesi için F1 skoruna bakalım
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
#grid search sonuçları
# print(f"en iyi Hyperparameter:{gsc.best_params_}") #13
# print(f"en iyi score:{gsc.best_score_}") # 0.630
#detaylı sonuçlar
# print("Detaylı GridSearchCV sonucu:")
gsc_result=pd.DataFrame(gsc.cv_results_).sort_values('mean_test_score',ascending=False)
#daha sade yazalım
# print(gsc_result[['param_n_neighbors','mean_test_score','rank_test_score']])
#en iyi hyperparameter değeri olan k=13 için 1 kere çalıştıralım
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
#random search ile aynı işlemler
#n_iter -> kaç adet parametre olacak
rsc=RandomizedSearchCV(estimator=KNeighborsClassifier(),
                       param_distributions=parameters,
                       n_iter=3,
                       cv=5,
                       verbose=1,
                       scoring='accuracy')
rsc.fit(X_train,y_train)
print("En iyi parametreler:{}".format(rsc.best_params_)) #11
print("En iyi score:{}".format(rsc.best_score_)) #0.6291












