import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import math
df=pd.read_csv('Smarket.csv')
print(df.describe())
#Today - Direction
df4=df[['Today','Direction']].copy()
#Manage the negatives
df4['Today']=df4['Today'].abs()
X=df4['Today']
y=df4['Direction']
#boxplot
y_grouped=df4.groupby('Direction')
print(y_grouped.count())
y_grouped.boxplot(column='Today',by='Direction', rot=90,figsize=(6,6))
#Clean the image titles
plt.title("Boxplot for Today by Direction")
plt.xlabel("Direction")
plt.ylabel("Today")
plt.show()
"""OUTPUT:Direction       
Down         602
Up           648 
Comment for result: (TURKISH)Görüldüğü gibi hisse senetlerinin bugünkü fiyatının aşağı mı yoksa yukarı 
mı olacağına bir önceki günün fiyatı üzerinden gidemiyoruz.Sağlıklı bir sonuç vermiyor. 1250 kayıttan 602
bir önceki güne göre düşmüş, 648 ise yükselmiş. Neredeyse yarı yarıya (BASE ORAN:DEFAULT ORAN %50)

ENGLISH(As can be seen, whether the current price of stocks is up or down We cannot go by the previous 
day's price. This is not giving us a good result. 602 out of 1250 records down compared to the 
previous day, 648 up. Almost in half )"""

#UNSUPERVISED LEARNING
#Working on NCI60 dataset
df=pd.read_csv('NCI60.csv')
print(df.head())
X=df.iloc[:,1:6831]
print(X)
# Scaling the data
sc=StandardScaler()
X_scaled=sc.fit_transform(X)
print(X_scaled)
#PCA : principal Component Analysis
pca=PCA(n_components=2)
pca_result=pca.fit_transform(X_scaled)
print('Eigenvalues')
print(pca.explained_variance_)
#Variances Percentage
print('Variances Percentage')
print(pca.explained_variance_ratio_ * 100)
principalDf=pd.DataFrame(data=pca_result,columns=['PC1','PC2'])
print(principalDf.head(10))
finalDf=pd.concat([principalDf,df[['labs']]],axis=1)
print(finalDf.head(10))
#Plot PCA
fig=plt.figure(figsize=(5,5))
ax=fig.add_subplot(1,1,1)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('En Etkili Genler - Hastalık') #most effective gens and disease
targets=['COlON','MELANOMA','LEUKEMIA']
for target in targets:
    indexler = finalDf['labs'] == target
    ax.scatter(finalDf.loc[indexler,'PC1'],
               finalDf.loc[indexler,'PC2'],s=50)
ax.legend(targets,loc='upper right')
ax.grid()
plt.show()

#Working on Advertising data
df=pd.read_csv('4_Ogrenme_Learning/data/Advertising.csv')
print(df.head())
print(df.describe())
X_1=df['TV']
X_2=df['radio']
X_3=df['newspaper']
print(type(X_1)) #series
print(X_1.shape) #It will be turn into list in the form of series
y=df['sales']
#Preparing for Regresyon:Changing the shapes to vectors(for regression)
linear_Regressor=LinearRegression()
#Edit the shape of dataframe's -> output
y_r=y.values.reshape(-1,1)
print(y.shape)
print(y_r.shape)
#Edit the shape of dataframe's -> input
X_1_r=X_1.values.reshape(-1,1)
X_2_r=X_2.values.reshape(-1,1)
X_3_r=X_3.values.reshape(-1,1)
print(X_1_r.shape)
#Relationship between Sales -TV :data -> X_1
plt.scatter(X_1,y,c='orange')
plt.xlabel('TV')
plt.ylabel('Satış')
plt.title('Satış - TV')
plt.show()
#Regression
linear_Regressor.fit(X_1_r,y_r)
y_pred_1=linear_Regressor.predict(X_1_r)
plt.plot(X_1,y_pred_1,color='red')
plt.xlabel('TV')
plt.ylabel('Satış')
plt.title('Satış - TV')
plt.show()
#Sales-Radio :data -> X_2
plt.scatter(X_2,y,c='green')
linear_Regressor.fit(X_2_r,y_r)
y_pred_2=linear_Regressor.predict(X_2_r)
plt.plot(X_2,y_pred_2,color='red')
plt.xlabel('Radio')
plt.ylabel('Satış')
plt.title('Satış-Radio')
plt.show()
#Sales-Newspaper :data -> X_3
plt.scatter(X_3,y,c='pink')
linear_Regressor.fit(X_3_r,y_r)
y_pred_3=linear_Regressor.predict(X_3_r)
plt.plot(X_3,y_pred_3,color='red')
plt.xlabel('Newspapaer')
plt.ylabel('Satış')
plt.title('Satış-Newpaper')
plt.show()

#Income.csv:I'll look the relationship between Education - Income
income=pd.read_csv('4_Ogrenme_Learning/data/Income.csv')
print(income.head(5))
X=income['Education']
y=income['Income']
#Preparation for regression :reshaping
X_r=X.values.reshape(-1,1)
y_r=y.values.reshape(-1,1)
# graph for Linear Regression
plt.figure(figsize=(10,6))
plt.scatter(X,y)
lr=LinearRegression() #regresyon
lr.fit(X_r,y_r)
y_pred=lr.predict(X_r)
plt.plot(X,y_pred,color='red')
plt.xlabel('Education')
plt.ylabel('Income')
plt.title('Education-Income')
plt.show() #The result so the last graph is linear.
#Simple/Single Linear Regression Project
df=pd.read_csv('4_Ogrenme_Learning/data/Advertising.csv')
print(df.head(10))
print(df.tail(2)) #2 Last lines
#To see the general informations about data
print(df.info())
#Basic statistics
print(df.describe())
#Visualization the data
data=df[['TV','sales']]
X=data['TV'] #input ->feature
y=data['sales'] #output
print(type(X))
#Plot the graph
plt.figure(figsize=(6,6))
sns.scatterplot(data=data,x='TV',y='sales',color='orange')
plt.title('SALES-TV')
plt.show()
#Creation of the model
#Creating the linear regresyon object
lr=LinearRegression()
#Lookin' up the shapes of input ve output
print('X in boyutu:', X.shape)
print('y nin boyutu:', y.shape)
#the result is not what sklearn's LinearRegression class wants
#So the result is not 2 Dimentional.It returns (200,0). And they both need to be (200,1)
#We need to reshape
X=X.values.reshape(-1,1) #It means accept the first one and add an column aside.
y=y.values.reshape(-1,1)
print(y.shape)
#train-test split
X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=100)
print('X train boyutu:',X_train.shape) #type is numpy.ndarray
print('y test boyutu:',y_test.shape)
#Run the linear regression model(fit)
lr.fit(X_train,y_train) #It will give the betas so coefficients
#Calculate the coefficients
print(lr.intercept_)
print(lr.coef_)
#Make a prediction
y_pred=lr.predict(X_test)
print(y_pred) #(60,1)
#Plot the truth and the prediction data
#The truth
fig,ax=plt.subplots(figsize=(6,6))
ax.scatter(X_test,y_test,label='Grand Truth',color='red')
#The prediction
ax.scatter(X_test,y_pred,label='Prediction',color='green')
plt.title('SALES - TV - PREDICTION')
plt.xlabel('TV')
plt.ylabel('SALES')
plt.legend(loc='upper left')
# plt.show()
#The first 10 real data and let's see the y(target) values
print(y_test[0:10])
#The first 10 prediction values
print(y_pred[0:10])
#Any change of prediction values
indexler=range(1,61)
fig,ax=plt.subplots(figsize=(6,6))
ax.plot(indexler,y_test,label='Grand Truth',color='red',linewidth=2)
#Prediction
ax.plot(indexler,y_pred,label='Grand Truth',color='green',linewidth=2)
plt.title('SALES - TV - PREDICTION')
plt.xlabel('TV')
plt.ylabel('SALES')
plt.legend(loc='upper left')
# plt.show()
#Plot the Errors
indexler=range(1,61)
#Residuals-Hatalar
fig,ax=plt.subplots(figsize=(6,6))
ax.plot(indexler,y_test-y_pred,label='Residuals',color='green',linewidth=2)
#Plot the Zero line
ax.plot(indexler,np.zeros(60),color='black')
#Control the model correction (RMSE,R^2)
r_2=r2_score(y_test,y_pred) #r^2
print(r_2)
mse=mean_squared_error(y_test,y_pred) #mse
print(mse)
rmse=math.sqrt(mse) #rmse
print(rmse)

