import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('20_Unsupervised_Learning/data/Mall_Customers.csv',index_col='CustomerID')
# print(df.info())
# print(df.head())
# print(df.describe())
# print(df.isnull().sum())
#dublike(mükerrer) data control
df.drop_duplicates(inplace=True)
#for inputs annual_income and spending_score
X=df.iloc[:,[2,3]].values
#find the optimum k with elbow method
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(X)
    #inertia_ returns wcss for this model
    wcss.append((kmeans.inertia_))
#wcss
plt.figure(figsize=(10,5))
sns.lineplot(x=range(1,11),y=wcss,marker='o',color='red')
plt.title('Elbow Method')
plt.xlabel('Cluster Sayısı')
plt.ylabel('WCSS')
plt.show() #5 yani K=5
#k-means
kmeans=KMeans(n_clusters=5,init='k-means++',random_state=42)
y_kmeans=kmeans.fit_predict(X)
plt.figure(figsize=(15,7))
sns.scatterplot(x=X[y_kmeans==0,0],y=X[y_kmeans==0,1],color='yellow',label='Cluster 1',s=50)
sns.scatterplot(x=X[y_kmeans==1,0],y=X[y_kmeans==1,1],color='blue',label='Cluster 2',s=50)
sns.scatterplot(x=X[y_kmeans==2,0],y=X[y_kmeans==2,1],color='green',label='Cluster 3',s=50)
sns.scatterplot(x=X[y_kmeans==3,0],y=X[y_kmeans==3,1],color='grey',label='Cluster 4',s=50)
sns.scatterplot(x=X[y_kmeans==4,0],y=X[y_kmeans==4,1],color='orange',label='Cluster 5',s=50)
sns.scatterplot(x=kmeans.cluster_centers_[:,0],
                y=kmeans.cluster_centers_[:,1],
                color='red',label='Centroids',s=300,markers='x')
plt.grid(False)
plt.title('Müşteri Clusterları')
plt.xlabel('Aylık Gelir (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


