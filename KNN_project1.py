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
dataset.groupby('Species').size()
feature_columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
#feature matrix
X=dataset[feature_columns].values
#the vector label
y=dataset['Species'].values
#LABEL ENCODING
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
print('Iris-setosa:\n',y[0:10])
print('\n')
print('Iris-virginica:\n',y[100:110])
#encode the y
y=le.fit_transform(y)
print('Iris-setosa:\n',y[0:10])
print('\n')
print('Iris-virginica:\n',y[100:110])
#after encoding
print('Iris-setosa:\n',y[0:10])
print('\n')
print('Iris-versicolour:\n',y[50:60])
print('\n')
print('Iris-virginica:\n',y[100:110])
#train-test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=4)
print(X_train.shape)
print(X_test.shape)
plt.figure()
sns.pairplot(dataset.drop('Id',axis=1),hue='Species',size=3,markers=['o','s','D'])
# plt.show()
#boxplot
plt.figure()
dataset.drop('Id',axis=1).boxplot(by='Species',figsize=(15,10))
# plt.show()
#model improvement
from sklearn.neighbors import KNeighborsClassifier
def sklearn_knn(train_data,label_data,test_data,k):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data,label_data)
    predict_label=knn.predict(test_data)
    return predict_label
#k=3
y_predict=sklearn_knn(X_train,y_train,X_test,3)
# print(y_predict)
#measuring model accuracy
def accuracy(test_labels,pred_labels):
    #accuracy calculating function
    #calculate the number of correct guesses
    correct=np.sum(test_labels==pred_labels)
    n=len(test_labels)
    #accuracy -> accuracy rate=true prediction /total test data
    accur=correct/n
    return accur
m=accuracy(y_test,y_predict)
# print(m) #accuracy rate is %96 percentage
n=len(dataset)
# print(n) #150
import math
k_max=math.sqrt(n)
# print(k_max) #12.247
#create a list for accuracy rates
normal_accuracy=[]
#possible k values
k_value=range(1,13)
from sklearn.metrics import accuracy_score
accuracy_sklearn=accuracy_score(y_test,y_predict) *100
# print('Model Doğruluğumuz:'+ str(round(accuracy_sklearn,2))+'%')
for k in k_value:
    y_predict=sklearn_knn(X_train,y_train,X_test,k)
    accur=accuracy_score(y_test,y_predict)
    normal_accuracy.append(accur)
#plotting the accuracies
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.plot(k_value,normal_accuracy,c='g')
plt.grid(True)
plt.show()








