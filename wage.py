import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Wage.csv')
print(df.describe())
print(df.head(10))
#The relationship between Age-Wage
df1=df[['age','wage']]
#girdi -> input, feature -> X
X=df1['age']
#çıktı -> output, label -> y
y=df1['wage']
#Prepare the mean
y_mean=df1.groupby('age').mean()
print(y_mean)
#Prepare the graph
fig, ax= plt.subplots(figsize=(8,8))
#Plot the data
ax.scatter(X,y,color='orange')
#Plot the mean
ax.plot(y_mean.index,y_mean)
#Format the graph
plt.title('MAAŞ-YAŞ')
plt.xlabel('Yaş')
plt.ylabel('Maaş')
plt.show()
#WAGE-YEAR
df2=df[['year','wage']]
X=df2['year']   #input
y=df2['wage']   #target/output
#Calculate the mean
y_mean=df2.groupby('year').mean()
fig,ax=plt.subplots(figsize=(6,6))
#Plot the data
ax.scatter(X,y,color='orange')
#Plot the mean
ax.plot(y_mean.index,y_mean,color='red')
#Format the graph
plt.title('MAAŞ-YIL')
plt.xlabel('Yıl')
plt.ylabel('Maaş')
plt.show()
#WAGE-EDUCATION level (boxplot)
df3=df[['education','wage']]
X=df3['education'] #input
y=df3['wage'] #output
#Group by education
y_grouped=df3.groupby('education')
y_grouped.boxplot(subplots=False,figsize=(6,6),rot=90)
#NOTATION Example
df=pd.read_csv('Wage.csv',index_col=0)
print(df.describe()) n=3000
#EDA
print(df.head())
print(df.columns)
print(df.columns.size)
degiskenler=['year', 'age', 'sex', 'maritl', 'race', 'education', 'region','jobclass',
             'health', 'health_ins', 'logwage']
print(len(degiskenler)) #p=11
X=df[degiskenler]
y=df['wage']
print(y.head())
print(X.iloc[0,:])
print(y.iloc[3])

