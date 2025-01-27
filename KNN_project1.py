import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

dataset=pd.read_csv('data/iris.csv')
#veri setindeki adet(n -> satır sayısı) ve feature( p-> değişken sayısı)
#shape -> (n,p)
dataset.shape
dataset.head()
dataset.describe()
dataset.info()
dataset.isnull().sum()
dataset.columns
dataset.index
dataset.dtypes
dataset.duplicated().sum()
dataset.nunique()
dataset.value_counts()
dataset.groupby('Species').size() #her türden kaç tane var
#değişken sütunları
feature_columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
#feature matrisi
X=dataset[feature_columns].values
#label vektörü
y=dataset['Species'].values
#LABEL ENCODING
#Scikit-learn den bir LabelEncoder nesnesi alalım
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
#encode etmeden önce y yi görelim
# print('Iris-setosa:\n',y[0:10])
# print('\n')
# print('Iris-virginica:\n',y[100:110])
#y yi encode edelim
y=le.fit_transform(y)
# print('Iris-setosa:\n',y[0:10])
# print('\n')
# print('Iris-virginica:\n',y[100:110])
#encode ettikten sonra y yi görelim
# print('Iris-setosa:\n',y[0:10])
# print('\n')
# print('Iris-versicolour:\n',y[50:60])
# print('\n')
# print('Iris-virginica:\n',y[100:110])
#train-test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=4)
#train data şekli
# print(X_train.shape)
# print(X_test.shape)
#veriyi görselleştirme:pair-plot
# plt.figure()
# sns.pairplot(dataset.drop('Id',axis=1),hue='Species',size=3,markers=['o','s','D'])
# plt.show()
#boxplot
# plt.figure()
# dataset.drop('Id',axis=1).boxplot(by='Species',figsize=(15,10))
# plt.show()
#model geliştirme
from sklearn.neighbors import KNeighborsClassifier
def sklearn_knn(train_data,label_data,test_data,k):
    #knn classifier yarat
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data,label_data)
    predict_label=knn.predict(test_data)
    return predict_label
#k=3
y_predict=sklearn_knn(X_train,y_train,X_test,3)
# print(y_predict)
#model doğruluğunu ölçme
def accuracy(test_labels,pred_labels):
    #doğruluk hesaplama fonksiyonu
    #doğru tahminlerin sayısını hesapla
    correct=np.sum(test_labels==pred_labels)
    #toplam test data adedi
    n=len(test_labels)
    #accuracy -> doğruluk oranı=doğru tahmin / toplam test verisi
    accur=correct/n
    return accur
m=accuracy(y_test,y_predict)
# print(m) #%96 lık bir doğruluk oranı elde ettik
n=len(dataset)
# print(n) #150
import math
k_max=math.sqrt(n)
# print(k_max) #12.247
#accuracy oranları için bir liste yarat
normal_accuracy=[]
#olabilecek k değerleri
k_value=range(1,13)
from sklearn.metrics import accuracy_score
accuracy_sklearn=accuracy_score(y_test,y_predict) *100
# print('Model Doğruluğumuz:'+ str(round(accuracy_sklearn,2))+'%')
#döngü ile tek tek k değerlerine bak
for k in k_value:
    y_predict=sklearn_knn(X_train,y_train,X_test,k)
    accur=accuracy_score(y_test,y_predict)
    normal_accuracy.append(accur)
#k değerlerine göre elde ettiğimiz accuracy leri çizelim
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.plot(k_value,normal_accuracy,c='g')
plt.grid(True)#ızgara ekle
plt.show()








