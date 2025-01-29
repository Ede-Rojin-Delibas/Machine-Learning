import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import math
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder #for label encoding
from sklearn.preprocessing import OneHotEncoder

#OLS - as a one variable for table reading
df=pd.read_csv('4_Ogrenme_Learning/data/Advertising.csv',index_col=0)
data=df[['TV','sales']]
#input ->output
X=data['TV']
y=data['sales'] #output
#Preparation(Shape controls) -> (200,1)
X=X.values.reshape(-1,1)
y=y.values.reshape(-1,1)
X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=100)
#Statsmodels OLS
X_train_ols=sm.add_constant(X_train)
#Generate statsmodels ols
sm_model=sm.OLS(y_train,X_train_ols)
#Take the results of OLS model
sonuc=sm_model.fit()
#Generate the OLS summary table
# print(sonuc.summary())
#Multiple Linear Regression Project - Advertising.csv :Purpose is modeling the relationship between TV, Radio and Newspaper budget and sales using linear regression
#EDA
sns.pairplot(df)
# plt.show() #We are trying to find the relationship between sales and other variables
X=df[['TV','radio','newspaper']]
y=df['sales']
# print(df.columns)
sns.pairplot(df,x_vars=df.columns[:3],y_vars=df.columns[3],height=5)
#Generate the model
lr= LinearRegression() #Create the linear Regression object.
#Preparation; first we need to check the size/shape of input and output
# print(X.shape) (200,3)
# print(y.shape) #y is wrong:(200,)
y=y.values.reshape(-1,1)
# print(y.shape)
#train-test split
X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=100)
print(X_train.shape) #(140,3)
print(y_train.shape) #(140,1)
print(X_test.shape)  #(60,3)
print(y_test.shape)  #(60,1)
#Run the Lineer Regression model(fit)
lr.fit(X_train,y_train)
print(lr.intercept_)
print(lr.coef_) #slope will be 3 cause we have 3 coefficient.
katsayilar=pd.DataFrame(lr.coef_,columns=['beta_1(TV)','beta_2(Radio)','beta_3(newspaper)'])
print(katsayilar) #newspaper's coefficient is super low;It means it won't be effective on the data and results
#Prediction
y_pred=lr.predict(X_test)
# print(y_pred)
# print(y_pred.shape) #(60,1)
#Plot the real data and prediction result :Truth label-y_test,prediction label - y_pred, input-X_test
# print(y_pred[0:10])
# print(y_test[0:10])
#Let's see the changes between predictions
indexler=range(1,61)
fig,ax=plt.subplots(figsize=(6,6))
ax.plot(indexler,y_test,label='Grand Truth',color='red',linewidth=2) #Data(Truth)
ax.plot(indexler,y_pred,label='Prediction',color='green',linewidth=2) #Prediction
plt.title('Gerçek - Prediction')
plt.xlabel('Data Index')
plt.ylabel('Sales')
plt.legend(loc='upper left')
# plt.show()
#Plot the residuals ,Error: Residuals (y-y^) ,y_test-y_pred
#Let's see the errors in the prediction points
indexler=range(1,61)
ax.plot(indexler,y_test-y_pred,label='Residuals',color='red',linewidth=2)
#Plot the zero line
ax.plot(indexler,np.zeros(60),color='black')
plt.title('Hatalar')
plt.xlabel('Data Index')
plt.ylabel('Sales')
plt.legend(loc='upper left')
# plt.show()
#Control the model correction
r_2=r2_score(y_test,y_pred) #Calculate the r_2
# print(r_2) # %90 is true result,
mse=mean_squared_error(y_test,y_pred)#MSE->RMSE
# print(mse)
rmse=math.sqrt(mse)
# print(rmse) #On average we are wrong by 1.36
#OLS
X_train_ols=sm.add_constant(X_train)
sm_model=sm.OLS(y_train,X_train_ols)
sonuc=sm_model.fit
print(sonuc.summary())
#Correlation
sns.heatmap(df.corr(),annot=True)
# plt.show()
#Built a new model according to results:Newspaper will remove for new model.
X_train_yeni=X_train[['TV','radio']] #new feature matrix:X_train_yeni
X_train_yeni.head()
X_test_yeni=X_test[['TV','radio']] #new test matrix:X_test_yeni
X_test_yeni.head()
lr.fit(X_train_yeni,y_train) # fictionalize the model
y_pred_yeni=lr.predict(X_test_yeni) #Take the new predictions
X_train_yeni_ols=sm.add_constant(X_train_yeni)
sm_model=sm.OLS(y_train,X_train_yeni_ols) #Generate the new ols model
sonuc=sm_model.fit()  #Take the result of OLS model
print(sonuc.summary())    #Generate the OLS summary table
#Now we have 2 variables (tv and radio) and they both are important(we have that assumption because of p values - 0.000
#Proof:Is newspaper important?: Just look at the newpaper
X=df['newspaper']
y=df['sales']
X=X.values.reshape(-1,1)    #reshaping
y=y.values.reshape(-1,1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=100)
X_train_ols=sm.add_constant(X_train)
model=sm.OLS(y_train,X_train_ols)
result=model.fit()
print(result.summary())
#converting categorical data to numerical data
evlilik_durumu=('Evli','Bekar','Belirtilmemiş')   #raw data
evlilik_df=pd.DataFrame(evlilik_durumu,columns=['Evlilik_Durumu'])# create a df
print(evlilik_df.info())    #type is object
evlilik_df['Evlilik_Durumu']=evlilik_df['Evlilik_Durumu'].astype('category')
print(evlilik_df.info()) #category
evlilik_df['Evlilik_Kategorileri'] = evlilik_df['Evlilik_Durumu'].cat.codes
print(evlilik_df)
evlilik_durumu=('Evli','Bekar','Belirtilmemiş')
evlilik_df=pd.DataFrame(evlilik_durumu,columns=['Evlilik_Durumu'])
#Generate the label encoder object
label_encoder=LabelEncoder()
evlilik_df['Evlilik_Kategorilerii_Sklearn']=label_encoder.fit_transform(evlilik_df['Evlilik_Durumu'])
#One-Hot Encoding
enc=OneHotEncoder(handle_unknown='ignore')
enc_result=enc.fit_transform(evlilik_df[['Evlilik_Durumu']])
enc_df=pd.DataFrame(enc_result.toarray())
evlilik_df=evlilik_df.join(enc_df)    #Add the enc_df to evlilik_df
#For any colon->Generate binary(0,1) colon
dummy_df=pd.get_dummies(evlilik_df,columns=['Evlilik_Durumu'])
# print(dummy_df)
evlilik_df=evlilik_df.join(dummy_df)






