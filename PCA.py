import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import decomposition
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D

iris=datasets.load_iris()
X=iris.data
y=iris.target
fig=plt.figure(1,figsize=(8,6))
ax = fig.add_subplot(111, projection='3d', elev=48, azim=134)
for name, label in [('Setosa',0),('Versicolour',1),('Virginica',2)]:
    ax.text3D(X[y==label,0].mean(),
              X[y==label,1].mean(),
              X[y==label,2].mean(),name,
              horizontalalignment='center',
              bbox=dict(alpha=.5,edgecolor='w',facecolor='w'))
#Making the label order's same
y_clr=np.choose(y,[1,2,0]).astype(float)
ax.scatter(X[:,0],X[:,1],X[:,2],c=y_clr,cmap=plt.cm.nipy_spectral, edgecolor='k')
# Axis labels
ax.set_xlabel("Sepal Length")
ax.set_ylabel("Sepal Width")
ax.set_zlabel("Petal Length")
# plt.show()
#Prediction with a single model (decisionTreeClassifier) ​​without PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,stratify=y,random_state=42)
# print(X_train.shape) #105,4
#decision tree, max_depth=2
clf=DecisionTreeClassifier(max_depth=2,random_state=42)
clf.fit(X_train,y_train)
preds=clf.predict_proba(X_test)
# print(preds)
#predict without PCA
pca=decomposition.PCA(n_components=2)
y_pred=clf.predict(X_test)
print("PCA olmadan Accuracy:{:.2f}".format(accuracy_score(y_test,y_pred)))
#Model builting with 4 variables and accuracy is %89
#If we reduce the number of variables with PCA and try again
pca=decomposition.PCA(n_components=2)
#print(X.mean()) #3.464
X_centered=X - X.mean(axis=0)
print(X_centered.shape) #150,4
pca.fit(X_centered)
X_pca=pca.transform(X_centered)
# print(X_pca.shape) #150,2
#result of PCA
plt.plot(X_pca[y==0,0],X_pca[y==0,1],'bo',label='Setosa')
plt.plot(X_pca[y==1,0],X_pca[y==1,1],'go',label='Versicolour')
plt.plot(X_pca[y==2,0],X_pca[y==2,1],'ro',label='Virginica')
plt.legend(loc=0)
# plt.show()
X_train, X_test,y_train,y_test=train_test_split(X_pca,y,test_size=0.3,stratify=y,random_state=42)
clf_pca=DecisionTreeClassifier(max_depth=2,random_state=42)
clf_pca.fit(X_train,y_train)
#accuracy without PCA
y_pred_pca=clf_pca.predict(X_test)
print("PCA ile Accuracy:{:.2f}".format(accuracy_score(y_test,y_pred_pca))) # 0.91
print(pca.components_)
print(pca.explained_variance_ratio_)
for i,component in enumerate(pca.components_):
print("{}.component:{}% variance".format(i+1,round(100*pca.explained_variance_ratio_[i],2)))
"""component-1:92.46% variance 
    component-2:5.31% variance"""
### PCA second example: MNIST- hand writing numbers ###
#In this example will predict hand writing numbers with PCA approach.
digits=datasets.load_digits()
X=digits.data
y=digits.target
plt.figure(figsize=(16,6))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X[i,:].reshape([8,8]),cmap='gray');
print(X.shape) #1797,64
#I will reduce the data with PCA which shapes is 62 to 2
pca=decomposition.PCA(n_components=2)
X_reduced=pca.fit_transform(X)
print("%d boyutlu data 2 boyutlu uzaya yansıtıldı(projecting)." % X.shape[1])
# print(X_reduced)
#plot 2D data
plt.figure(figsize=(12,10))
plt.scatter(X_reduced[:,0],X_reduced[:,1],c=y,
            edgecolor='none',alpha=0.7,s=40,
            cmap = plt.get_cmap('nipy_spectral', 10))
plt.colorbar()
plt.title("MNIST - PCA Projeksiyonu");
# plt.show()
#total explained variable
pca=decomposition.PCA().fit(X)
plt.figure(figsize=(10,7))
plt.plot(np.cumsum(pca.explained_variance_ratio_),color='k',lw=2)
plt.xlabel('Principal Component Sayısı')
plt.ylabel('Toplam Açıklanan Variance')
plt.xlim(0,63)
plt.yticks(np.arange(0,1.1,0.1))
plt.axvline(21,c='b')
plt.axhline(0.9,c='r')
plt.show()
#result:with 21 variables, pca can explain total variance





















