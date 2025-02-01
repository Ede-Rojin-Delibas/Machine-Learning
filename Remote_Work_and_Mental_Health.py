######### REMOTE WORK AND MENTAL HEALTH #########
import numpy as np
import pandas as pd
import math
from mpl_toolkits.mplot3d import Axes3D #3D grafikler oluşturmak için(scatter plot)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn import tree
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
# generate a DataFrame
# df=pd.read_csv('Impact_of_Remote_Work_on_Mental_Health1.csv')
df=pd.read_csv('rw_new_data.csv')
#EDA
# print(df.head())
# print(df.describe())  #it returned numerical colons and statistical assumptions
# print(df.info()) #How many record do we have?,do we have empty places on colons? and return data types.
# print(df.shape)   #colons and attributes on dataset.(5000,20)
# print(len(df)) #it returned total number of observations (satır sayısı).
# print(df.columns) #number of columns
# print(len(df.describe().columns)) #total numerical colons.
'''Summary of data:
    20 columns:1 output, 19 input(6 are numerical, 13 are categorical)
'''
#Error occured: correction of categorical values :Gender the valu 4
df['Gender'] = df['Gender'].replace({'4' : 'Female'})
# print(df.Gender.unique())
#To see unique(tekil) data number
for col in df.columns:
    print(col,df[col].nunique())
#Let's see the values for categorical columns
for col in df.columns:
    values=[]
    if col not in df.describe().columns:
        for val in df[col].unique():
            values.append(val)
            print("{0} -> {1}".format(col, values))
data=df.copy()
#unvariate analysis
# Job_Role:
print(data['Job_Role'].value_counts())
custom_palette = ['#FF5733', '#33FF57', '#3357FF', '#F4F134', '#34F4F1']  # special colour list
sns.countplot(data.Job_Role,palette=custom_palette)
#palette='viridis'
# plt.show()
#colour assignment to referances
sns.countplot(x='Job_Role', data=data, hue='Gender', palette='coolwarm')  # Female/Male colouring
plt.title("Job Role Dağılımı (Cinsiyete Göre)")
plt.xticks(rotation=45)
# plt.show()
#The relationship between years of Experience and stress level
sns.countplot(x='Years_of_Experience', data=data, hue='Stress_Level', palette='coolwarm')
plt.title("Deneyim ve Stres Seviyesi")
plt.xticks(rotation=45)
# plt.show()
#The relationship between Mental health condition and social isolation rating
sns.countplot(x='Mental_Health_Condition', data=data, hue='Social_Isolation_Rating', palette='coolwarm')
plt.title("Sosyal İzolasyon ve Ruhsal Sağlık Durumu Arasındaki İlişki")
plt.xticks(rotation=45)
# plt.show()
#shape and subplot settings
fig=plt.figure(figsize=(20,12))
plt.subplot(2,3,1) #first subplot(2 attributes, 3 columns , 1th graph)
# Plotting the histogram
plt.hist(['Work_Life_Balance_Rating'],color='skyblue',bins=10)  #It works for dividing the histograms values to 10K.
plt.title('Work Life Balance')
plt.xlabel('Rating')
plt.ylabel('Frequency')
# plt.show()
plt.subplot(2,3,5)
plt.title('Work Life Balance Score')
plt.hist(data.Work_Life_Balance_Rating,bins=10,color='skyblue')
plt.xlabel('Rating')
plt.ylabel('Frequency')
# plt.show()
plt.title('Physical Activity')
sns.countplot(data.Physical_Activity)
plt.subplot(2,3,2)
# plt.show()
#bivariate analysis
#The relationship between satisfaction with Remote Work and WLB
plt.figure(figsize=(16,8))
plt.title('Satisfaction with Remote Work ve WLB',fontsize=16)
sns.barplot(x=data.Satisfaction_with_Remote_Work, y=data.Work_Life_Balance_Rating, palette='Set2')
plt.xticks(rotation=90)
plt.tight_layout()
# plt.show()
plt.figure(figsize=(16, 8))
plt.title('Work Location ve Work Life Balance (Boxplot)', fontsize=16)
sns.boxplot(x=data.Work_Location, y=data.Work_Life_Balance_Rating, palette='Set1')
plt.xticks(rotation=90)
plt.tight_layout()
# plt.show()
# Basic statistical summary of Work_Life_Balance_Rating
# print(data.describe())
print(data['Work_Life_Balance_Rating'].nunique())
# Distribution of columns(frequency)
print(data['Work_Life_Balance_Rating'].value_counts())
#Control management (Distribution and Variance)
print(data.groupby('Work_Location')['Work_Life_Balance_Rating'].describe())
# print(data.columns)
# print(df[['Job_Role']].head(5))  # Tablonun ilk 5 satırını kontrol et
# Boxplot for Years of Experience vs. Satisfaction with Remote Work
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Years_of_Experience', y='Satisfaction_with_Remote_Work', palette='coolwarm')
plt.title('Satisfaction with Remote Work by Years of Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Satisfaction with Remote Work')
# plt.show()
# Bar plot for Work Location vs. Productivity Change
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Work_Location', hue='Productivity_Change', palette='Set2')
plt.title('Work Location vs. Productivity Change')
plt.xlabel('Work Location')
plt.ylabel('Count')
plt.legend(title='Productivity Change')
# plt.show()
# Countplot for Work-Life Balance Rating
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Work_Life_Balance_Rating', palette='pastel')
plt.title('Work-Life Balance Rating')
plt.xlabel('Work-Life Balance Rating')
plt.ylabel('Count')
# plt.show()

############## ÇOKLU LINEER REGRESYON ################
X=df[['Age','Depression','Unsatisfied']]
y=df['Work_Life_Balance_Rating']
#Generating a model
lr=LinearRegression()
# print(X.shape) #input size
# # print(y.shape) #output size
# y=y.values.reshape(-1,1)
# print(y.shape)
#train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=100)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
#run the linear regression model
lr.fit(X_train,y_train)
lr.intercept_#[2.79558753]
lr.coef_#[[0.00295403 0.01549519 0.13456676]]
katsayilar=pd.DataFrame(lr.coef_,columns=['beta_1(Age)','beta_2(Depression)','beta_3(Unsatisfied)'])
# print(katsayilar)
'''   beta_1(Age)  beta_2(Depression)  beta_3(Unsatisfied)
0     0.002954            0.015495             0.134567'''
#prediction
y_pred=lr.predict(X_test)
print(y_pred)
print(y_pred[0:10])
print(y_test[0:10])
print(y_pred.shape) #(1000, 1)
#variation between each prediction
indexler=range(1,1001)
fig,ax=plt.subplots(figsize=(6,6))
ax.plot(indexler,y_test,label='Grand Truth',color='red',linewidth=2) #truth
ax.plot(indexler,y_pred,label='Prediction',color='green',linewidth=2) #predict
plt.title('Gerçek - Prediction')
plt.xlabel('Data Index')
plt.ylabel('Work_Life_Balance_Rating')
plt.legend(loc='upper left')
# plt.show()
#plot the residuals
indexler=range(1,1001)
ax.plot(indexler,y_test-y_pred,label='Residuals',color='red',linewidth=2)
#ploting the zero line
ax.plot(indexler,np.zeros(1000),color='black')
plt.title('Hatalar')
plt.xlabel('Data Index')
plt.ylabel('Sales')
plt.legend(loc='upper left')
# plt.show()
#correcting the model accuracy
r_2=r2_score(y_test,y_pred)
# print(r_2) #-0.010794210126301218
mse=mean_squared_error(y_test,y_pred)
# print(mse) #1.9514756906614021
rmse=math.sqrt(mse)
# print(rmse) #1.3969522864655766
#OLS
X_train_ols=sm.add_constant(X_train)
print(X_train_ols.dtypes)
print(y_train.dtype)
X_train_ols = X_train_ols.astype(float)  # Convert the all columns to type float
y_train = y_train.astype(float)
# Shape as a one-dimensional vector
y_train = y_train.ravel()  # or y_train = np.squeeze(y_train)
X_train_ols=np.asarray(X_train_ols)
y_train=np.asarray(y_train)
sm_model=sm.OLS(y_train,X_train_ols)
sonuc=sm_model.fit()
# print(sonuc.summary())

#FEATURE SELECTION AND MODEL IMPROVEMENT:FORWARD SELECTION AND FEATURE IMPORTANCE
#new variable selection
X=df.drop(columns=['Work_Life_Balance_Rating'])
y=df['Work_Life_Balance_Rating']
#important new variable selection
selector=SelectKBest(score_func=f_regression,k=5) #selection of the best 5 variables
X_selected=selector.fit_transform(X,y)
selected_features=X.columns[selector.get_support()]
print("Seçilen değişkenler : ",selected_features)
#chosed variables :  Index(['Manufacturing', 'Retail', 'Medium', 'No Change', 'Unsatisfied'], dtype='object')

# # Mutual Information test
selector_mi = SelectKBest(score_func=mutual_info_regression, k=5)
X_selected_mi = selector_mi.fit_transform(X, y)
selected_features_mi = X.columns[selector_mi.get_support()]
print("Mutual Information Skorları:", selected_features_mi)
#Mutual Information Scores: Index(['Age', 'Male', 'IT', 'Medium', 'Poor'], dtype='object')
X_train, X_test, y_train, y_test = train_test_split(X_selected_mi, y, test_size=0.2, random_state=42)
scaler = StandardScaler()  # also you can use 'MinMaxScaler' in here
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#Fitting the model with logistic regression
model=LogisticRegression()
#CROSS VALIDATION
# 5 k-fold CV
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')

# print("Cross-validation skorları:", cv_scores)  # It shows all scores in fold
# print("CV Ortalama Skor:", cv_scores.mean())  # It shows average success
'''Cross-validation skorları: [0.22    0.205   0.19875 0.21625 0.195  ]
    CV Ortalama Skor: 0.20700000000000002
'''
model.fit(X_train_scaled,y_train)
y_pred=model.predict(X_test_scaled)
accuracy=accuracy_score(y_test,y_pred)
# print("Accuracy:",accuracy) #Accuracy: 0.206 #Low accuracy
#print(y_train.value_counts(normalize=True))
"""Work_Life_Balance_Rating
1    0.21250
3    0.20800
4    0.19625
5    0.19350
2    0.18975
#yorumu: (Turkish)sınıfların veri setindeki oranı nispeten dengeli,Bu, sınıf dengesizliği nedeniyle düşük model performansı
beklenmediğini gösterir. Ancak modelin accuracy'sinin düşük olması, bu oranlara rağmen feature'ların hedef 
değişkenle yeterince iyi ilişkili olmadığını gösterebilir.
(Eng)The proportion of classes in the dataset is relatively balanced, which results in poor model performance due to class imbalance
indicates that it is not expected. However, the low accuracy of the model means that despite these rates, the features are not on target. 
It may indicate that it is not well enough related to the variable.
"""
### Alternative performance metrics ###
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Number_of_Virtual_Meetings', y='Satisfaction_with_Remote_Work', hue='Gender', palette='deep')
plt.title('Number of Virtual Meetings vs. Satisfaction with Remote Work')
plt.xlabel('Number of Virtual Meetings')
plt.ylabel('Satisfaction with Remote Work')
# plt.show()
# Correlation matrix heatmap with only numeric columns
plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include=[np.number])  # Just choose the numerical colons
correlation_matrix = numeric_df.corr() #calculation of correlation matrix
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', mask=np.triu(correlation_matrix))
plt.title('Correlation Matrix')
# plt.show()
#Visualization using 3D scatter plot
# an example from dataset
data = {
    'Age': [25, 30, 35, 40, 45],
    'Hours_Worked_Per_Week': [40, 45, 38, 50, 55],
    'Satisfaction_with_Remote_Work': [3, 4, 2, 5, 1]
}
df = pd.DataFrame(data)
# generate a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Age'], df['Hours_Worked_Per_Week'], df['Satisfaction_with_Remote_Work'], c='b', marker='o')
ax.set_xlabel('Age')
ax.set_ylabel('Hours Worked per Week')
ax.set_zlabel('Satisfaction with Remote Work')
plt.title('3D Scatter Plot')
# plt.show()
#The distribution of Work_Life_Balance_Rating by its own
sns.displot(data.Work_Life_Balance_Rating)#,bins=10,kde=True,color='skyblue'
# plt.show()
# print(data.Work_Life_Balance_Rating.describe())
cols=['Age',
       'Years_of_Experience', 'Hours_Worked_Per_Week',
       'Number_of_Virtual_Meetings', 'Work_Life_Balance_Rating',
       'Social_Isolation_Rating','Company_Support_for_Remote_Work'
]
#Relate with Regression lines
plt.figure(figsize=(25,35)) #arrange the general graph shape
for i in range(len(cols)):
    plt.subplot(4,3,i+1)
    sns.regplot(x=eval('data' + '.'+ cols[i]),y=data['Work_Life_Balance_Rating'],
                ci=None,  scatter_kws={'alpha': 0.6},#For transparency of dots
                line_kws={'color': 'red'}, #the colour of regression plot
                color='blue') #the colour of dots
    plt.title(cols[i] + ' vs WLB ', fontsize=14)
    plt.xlabel(cols[i], fontsize=12)
    plt.ylabel('WLB', fontsize=12)
    plt.xticks(fontsize=10,rotation=45)
    plt.yticks(fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout(pad=5.0)
plt.subplots_adjust(hspace=0.6, wspace=0.5)  # spaces :hspace and wspace
# plt.show()
data.drop(columns=['Employee_ID'],axis=1,inplace=True)
#One-Hot Encoding
categorical_columns=['Gender','Job_Role','Industry','Work_Location','Stress_Level','Mental_Health_Condition',
                     'Access_to_Mental_Health_Resources','Productivity_Change','Satisfaction_with_Remote_Work',
                     'Physical_Activity','Sleep_Quality','Region']
encoded_data=pd.get_dummies(data,columns=categorical_columns)
#Investigate the OHE data
# print(encoded_data.head())
# print(encoded_data.columns)
# print(encoded_data.info())    #the new datas are bool
# print(encoded_data.describe())
gender_data = encoded_data[['Gender_Male', 'Gender_Female']].sum() #Gender distribution is numeric
gender_data.plot(kind='bar')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
# plt.show()
#the relationship between Job_Role_Project Manager and Stress_Level_High
sns.barplot(x='Stress_Level_High', y='Job_Role_Project Manager', data=encoded_data)
plt.title('High Stress Level vs Job Role (Manager)')
# plt.show()
#bivariate analysis:Job_Role vs Work_Life_Balance
plt.figure(figsize=(8,6))
sns.boxplot(x=encoded_data['Job_Role_Data Scientist'], y=data.Work_Life_Balance_Rating)
plt.title("Job Role: Data Scientist vs Work-Life Balance Rating")
plt.xlabel("Data Scientist (1=Yes, 0=No)")
plt.ylabel("Work-Life Balance Rating")
# plt.show()
plt.figure(figsize=(20,30))
for idx,column_name in enumerate(encoded_data.columns):
    if idx % 12 ==0 and idx!=0:
        # Styling
        plt.style.use('seaborn')
        plt.tight_layout()
        #plt.show()
        plt.figure(figsize=(20, 30)) # generate a new figure
    plt.subplot(4,3,idx % 12 + 1) # 4x3 grid,12 figure in graphs
    sns.regplot(x=encoded_data[column_name],y=encoded_data['Stress_Level_High'])
    plt.title(f"{column_name} vs Stress Level")
# # Styling
plt.style.use('seaborn')
plt.tight_layout()
# plt.show()
#Analysis with OHE encoded_data results(bivariate)
plt.style.use('tableau-colorblind10')  # entegrate the Seaborn styles
# figsize has been expanded and style adjustments have been made to avoid confusing titles
plt.figure(figsize=(20, 30))
for idx, column_name in enumerate(encoded_data.columns):
    if idx % 12 == 0 and idx != 0:
        #Create a new figure after every 12 plots
        plt.tight_layout()  # edit previous figure
        plt.show()
        plt.figure(figsize=(20, 30))

        # Creating a subplot
    plt.subplot(4, 3, idx % 12 + 1)  # 4x3 grid
    sns.regplot(x=encoded_data[column_name], y=encoded_data['Stress_Level_High'])

    #Make headlines more readable
    plt.title(f"{column_name} vs Stress Level", fontsize=12)


    plt.xlabel(column_name, fontsize=10)
    plt.ylabel('Stress Level (High)', fontsize=10)

# Edit the last figure and show it
# plt.tight_layout()  # Reduces congestion
# plt.show()

# print(encoded_data.shape)
# print(data.shape)
#save the OHE data
encoded_data.to_csv('C:/Users/Ede Rojin DELİBAŞ/OneDrive/Masaüstü/RemoteWork&MentalHealth/encoded_data.csv',index=False)
# Examine data associated with the correlation matrix
correlation_matrix = encoded_data.corr() #Calculate the correlations
#WLB:
wlb_correlation=correlation_matrix['Work_Life_Balance_Rating'].sort_values(ascending=False)#focus on WLB relationships
# print(wlb_correlation)
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
# plt.show()
rw_new_data=data.copy()
for i in categorical_columns:
    temp_df=pd.get_dummies(eval('rw_new_data.'+i),drop_first=True)
    rw_new_data=pd.concat([rw_new_data,temp_df],axis=1)
    rw_new_data.drop(columns=[i],axis=1,inplace=True)
# print(rw_new_data.head())
rw_new_data.to_csv('C:/Users/Ede Rojin DELİBAŞ/OneDrive/Masaüstü/RemoteWork&MentalHealth/rw_new_data.csv',index=False)
print(rw_new_data.columns)
correlation_matrix = rw_new_data.corr()
correlation_matrix.columns = correlation_matrix.columns.str[:3]  # Take the first 3 letters
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
#WLB:
wlb_correlation=['Work_Life_Balance_Rating','Low','Remote','North America']
correlation_matrix = rw_new_data[wlb_correlation].corr()
# # print(wlb_correlation)
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
# plt.show()
#relationship:low
#WLB,'Hours_Worked_Per_Week','Number_of_Virtual_Meetings','Medium','Medium'
wlb_correlation=['Work_Life_Balance_Rating','Low','Hours_Worked_Per_Week','Number_of_Virtual_Meetings','Medium']
correlation_matrix = rw_new_data[wlb_correlation].corr()
# # print(wlb_correlation)
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
# plt.show()
#correlation related remote
wlb_correlation=['Remote','Social_Isolation_Rating','Company_Support_for_Remote_Work','Satisfied']
correlation_matrix = rw_new_data[wlb_correlation].corr()
# # print(wlb_correlation)
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
# plt.show()#low relation
#Mental_Health
wlb_correlation=['Depression','Medium','Work_Life_Balance_Rating','Yes']
correlation_matrix = rw_new_data[wlb_correlation].corr()
# # print(wlb_correlation)
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
# plt.show()#low relation
wlb_correlation=['Increase','Burnout','Work_Life_Balance_Rating','']
correlation_matrix = rw_new_data[wlb_correlation].corr()
# # print(wlb_correlation)
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
# plt.show()#low relation
sns.pairplot(rw_new_data,hue='Work_Life_Balance_Rating')
# plt.show()
# Let's choose dependent and independent variables
X = rw_new_data[['Hours_Worked_Per_Week', 'Number_of_Virtual_Meetings']]
y = rw_new_data['Work_Life_Balance_Rating']  # target
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# Creating polynomial properties (more complex relationships can be captured by increasing the degree)
poly = PolynomialFeatures(degree=2, include_bias=False)  # 2. derece polinom
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
#Defining a linear reg model for polynomial regression
model=LinearRegression()
# fitting the modeli
model.fit(X_train_poly,y_train)
# Examining the coefficients of the model
print("Katsayılar:",model.coef_) #5 slopes,coefficients: [-0.01522629  0.06112391  0.00023802 -0.00062636 -0.00228006]
print("Bias(Intercept):",model.intercept_) #Bias(Intercept): 3.069268970304387
# #model evaluation :predictions
y_train_pred=model.predict(X_train_poly)
y_test_pred=model.predict(X_test_poly)
# #Model performance evaluation(MSE and R^2 scores)
mse_train=mean_squared_error(y_train,y_train_pred)
mse_test=mean_squared_error(y_test,y_test_pred)
r2_train=r2_score(y_train,y_train_pred)
r2_test=r2_score(y_test,y_test_pred)
# print("Eğitim Verisi - MSE:",mse_train,"R2:",r2_train) #Eğitim Verisi - MSE: 2.0048163422224805 R2: 0.002086307420924194
# print("Test Verisi - MSE:",mse_test,"R2:",r2_test) #Test Verisi - MSE: 1.9156997597169967 R2: -0.005727011371324586

#### **train dataset -residual and R² score:**
#####MSE's comment: - It shows that the model's errors in the predictions are high.
#- **Negative R²:** Model, It cannot explain the variance of the target variable in the test data in any way. Even model predictions,
# is worse off than a simple average estimate!**(overfitting)
#visualization
plt.scatter(y_test,y_test_pred,alpha=0.7)
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahminler")
plt.title("Polinomsal Regresyon: Gerçek vs Tahmin")
# plt.show()
#Regression line on features and predictions
plt.figure(figsize=(10,6))
plt.scatter(X_test['Hours_Worked_Per_Week'],y_test_pred,color='red',label="Tahminler")
plt.xlabel('Hours Worked Per Week')
plt.ylabel('Predicted WLB')
plt.title('Hours Worked Per Week vs Predicted WLB(Polynomial Regression)')
plt.legend()
# plt.show()
#Overfitting: Your model has probably also learned the noise in the data, improving its ability to generalize to new data.
#lost. This may be due to the use of a high degree polynomial or insufficient amount of data may occur.
#lost model is weak and overfit.For this reason,I' ll do some regulations with lasso and ridge regularization methods
#Describe these models
ridge=Ridge(alpha=1.0) #alpha determines regulation level(0 is normal regression)
lasso=Lasso(alpha=0.1)
# fitting the ridge model
ridge.fit(X_train_poly,y_train)
y_pred_ridge=ridge.predict(X_test_poly)
#fit the lasso model
lasso.fit(X_train_poly,y_train)
y_pred_lasso=lasso.predict(X_test_poly)
#evaluating performance
# print("Ridge:")
# print("MSE:",mean_squared_error(y_test,y_pred_ridge))
# print("R2:",r2_score(y_test,y_pred_ridge))
"""Ridge:
MSE: 1.9156976297083876
R2: -0.00572589313388594"""
# print("\nLasso:")
# print("MSE:",mean_squared_error(y_test,y_pred_lasso))
# print("R2:",r2_score(y_test,y_pred_lasso))
"""Lasso:
MSE: 1.9122484262923471
R2: -0.003915089000497751"""
# print("Ridge Katsayılar:",ridge.coef_)
# print("Lasso Katsayılar:",lasso.coef_)
"""Ridge Katsayılar: [-0.01522601  0.06110617  0.00023799 -0.00062617 -0.00227944]
Lasso Katsayılar: [-0.00000000e+00  0.00000000e+00 -8.88032422e-06 -9.61284096e-06
 -0.00000000e+00]
 """
# coefficient of ridge and lasso and visualization of their working
alphas=[0.01,0.1,1,10,100]
ridge_coefs=[]
lasso_coefs=[]
for alpha in alphas:
    ridge=Ridge(alpha=alpha).fit(X_train_poly,y_train)
    lasso=Lasso(alpha=alpha).fit(X_train_poly,y_train)
    ridge_coefs.append(ridge.coef_)
    lasso_coefs.append(lasso.coef_)
# Plot the Ridge coefficients
plt.figure(figsize=(10,6))
plt.plot(alphas,ridge_coefs,marker='o')
plt.xscale('log')
plt.title("Ridge katsayılarının Düzenleme ile Değişimi")
plt.xlabel("Alpha (log scale)")
plt.ylabel("Katsayılar")
# plt.show()
#  Plot the lasso coefficients
plt.figure(figsize=(10,6))
plt.plot(alphas,lasso_coefs,marker='x')
plt.xscale('log')
plt.title("Lasso katsayılarının düzenleme ile Değişimi")
plt.xlabel("Alpha (log scale)")
plt.ylabel("Katsayılar")
# plt.show()
#output comment: Despite the ridge, the model's performance could not improve. (MSE) The ridge model for R2 is a simple mean prediction model.
#It's not even as descriptive as #. That is, the relationship between the independent variables (inputs) and the target variable (output) is linear.
#not captured by regression model or too weak
#Lasso:Lasso gives a slightly better result than the ridge model in terms of MSE (with a smaller error).
#r2:lasso's r2 score is slightly higher than Ridge but still negative
#Coefficient of Ridge and LAsso:It prevented ridge overfitting but could not learn a more meaningful relationship.

########## DECISION TREES CLASSIFICATION ############
#Distribution of target variable
# print(df['Work_Life_Balance_Rating'].value_counts())
# print(df.isnull().sum())
# print(len(df.columns))#(41)
# print(df.select_dtypes(include=['int64','float64']).shape) #(5000,7)
numeric_cols=df.select_dtypes(include=['int64','float64']).columns
# print(df.select_dtypes(include='object').shape)
cat_cols=df.select_dtypes(include='object').columns
# print(cat_cols)
#Distribution of numeric variables
df.hist(column=numeric_cols,figsize=(10,10))
plt.subplots_adjust(hspace=0.5,wspace=0.5)
# plt.show()
df_encoded=pd.get_dummies(df,columns=cat_cols)
#feature vector and target variable
X=df.drop(['Work_Life_Balance_Rating'],axis=1)
y=df['Work_Life_Balance_Rating']
# print(X.shape) #(5000,40)
# print(y.shape) #(5000,)
#Train - Test Split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
# print(X_train.shape) #(4000,40)
# print(X_test.shape)  #(1000,40)
#FEATURE SCALING
cols=X_train.columns
# print(cols)
#fit the StandardScaler on X_train
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train) #X_train is a Numpy array now
X_test=scaler.transform(X_test)
# print(type(X_train))
# print(X_train.shape)
# print(X_train.head())
#Decision Tree Classification with gini index
clf_gini=DecisionTreeClassifier(criterion='gini',max_depth=3,random_state=0)
clf_gini.fit(X_train,y_train) #fit the model
#Making a prediction with our model which is generated with gini index
y_pred_gini_proba=clf_gini.predict_proba(X_test)
#ROC-AUC SCORE
y_pred_gini_score=roc_auc_score(y_test,y_pred_gini_proba,multi_class="ovr")
# print('Modelin Gini Index ile ROC-AUC skoru: {0:0.4f}'.format(y_pred_gini_score))
#Comparing train and test ROC-AUC values
y_pred_train_gini_proba=clf_gini.predict_proba(X_train)
y_pred_train_gini_score=roc_auc_score(y_train,y_pred_train_gini_proba,multi_class="ovr")
# print('Modelin gini index ile ROC-AUC Skoru: {0:0.4f}'.format(y_pred_train_gini_score))
# print('Train set skoru:{:.4f}'.format(y_pred_train_gini_score))
# print('Test set skoru:{:.4f}'.format(y_pred_gini_score))
""" Train set skoru:0.5476
    Test set skoru:0.4939
    train=%55
    test=%50
"""
#Visualization of decision trees
plt.figure(figsize=(24,16))
tree.plot_tree(clf_gini.fit(X_train,y_train))
# plt.show()
#Decision Tree Classifier with Entropy
clf_ent=DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=0)
#fit the model
clf_ent.fit(X_train,y_train)
y_pred_ent_proba=clf_ent.predict_proba(X_test)
#ROC-AUC score
y_pred_ent_score=roc_auc_score(y_test,y_pred_ent_proba,multi_class="ovr")
# print('Modelin Entropy ile ROC-AUC skoru:{0:0.4f}'.format(y_pred_ent_score))
y_pred_train_ent_proba=clf_ent.predict_proba(X_train)
y_pred_train_ent_score=roc_auc_score(y_train,y_pred_train_ent_proba,multi_class="ovr")
# print('Modelin Entropy ile ROC-AUC skoru:{0:0.4f}'.format(y_pred_train_ent_score))
""" Modelin Entropy ile ROC-AUC skoru:0.4941 ; %49
    Modelin Entropy ile ROC-AUC skoru:0.5472 ;%50 
    """
# train and test scores
# print('Train set skoru:{:.4f}'.format(y_pred_train_ent_score))
# print('Test set skoru:{:.4f}'.format(y_pred_ent_score)) #same results above
plt.figure(figsize=(24,16))
tree.plot_tree(clf_ent.fit(X_train,y_train))
# plt.show()
#proof for overfit :Increased max_depth with gini index
clf_gini=DecisionTreeClassifier(criterion='gini',max_depth=6,random_state=0)
clf_gini.fit(X_train,y_train)
y_pred_gini_proba=clf_gini.predict_proba(X_test)
y_pred_gini_score=roc_auc_score(y_test,y_pred_gini_proba,multi_class="ovr")
# print('Modelin gini index ile ROC-AUC Skoru: {0:0.4f}'.format(y_pred_gini_score)) #0.4972
y_pred_train_gini_proba=clf_gini.predict_proba(X_train)
y_pred_train_gini_score=roc_auc_score(y_train,y_pred_train_gini_proba,multi_class="ovr")
# print('Modelin gini index ile ROC-AUC Skoru: {0:0.4f}'.format(y_pred_train_gini_score))#0.6310
######## examination of results ########
# print('Train set skoru:{:.4f}'.format(y_pred_train_gini_score))
# print('Test set skoru:{:.4f}'.format(y_pred_gini_score))
# max_depth=9
clf_gini=DecisionTreeClassifier(criterion='gini',max_depth=9,random_state=0)
clf_gini.fit(X_train,y_train)
y_pred_gini_proba=clf_gini.predict_proba(X_test)
y_pred_gini_score=roc_auc_score(y_test,y_pred_gini_proba,multi_class="ovr")
# print('Modelin gini index ile ROC-AUC Skoru: {0:0.4f}'.format(y_pred_gini_score))
y_pred_train_gini_proba=clf_gini.predict_proba(X_train)
y_pred_train_gini_score=roc_auc_score(y_train,y_pred_train_gini_proba,multi_class="ovr")
# print('Modelin gini index ile ROC-AUC Skoru: {0:0.4f}'.format(y_pred_train_gini_score))
##### examination of results ##### overfit risk
# print('Train set skoru:{:.4f}'.format(y_pred_train_gini_score))#0.7802
# print('Test set skoru:{:.4f}'.format(y_pred_gini_score))#0.5077
#visualizing
plt.figure(figsize=(24,16))
tree.plot_tree(clf_gini.fit(X_train,y_train))
# plt.show()
############### GRADIENT BOOSTING ##############
df=pd.read_csv('rw_new_data.csv')
# print(df.shape) #(5000,41)
col_names=df.columns
# print(col_names)
# print(df['Work_Life_Balance_Rating'].value_counts())
# print(df.info())
# print(df.isnull().sum()) #Mental_Health_Condition : 1196,Physical_Activity:1629 null data
df['Mental_Health_Condition'] = df['Mental_Health_Condition'].fillna(df['Mental_Health_Condition'].mode()[0])
df['Physical_Activity'] = df['Physical_Activity'].fillna(df['Physical_Activity'].mode()[0])
# print(df.isnull().sum()) #I Filled the null variables with mode
# print(len(df.columns)) #41
# print(df.select_dtypes(include=['int64','float64']).shape) #(5000,7)
numeric_cols=df.select_dtypes(include=['int64','float64']).columns
# print(numeric_cols)
cat_cols=df.select_dtypes(include=['object']).columns
# print(cat_cols)
df.hist(column=numeric_cols,figsize=(10,10))
plt.subplots_adjust(wspace=0.5,hspace=0.5)
# plt.show()
#feature vector and target variable
X=df.drop(['Work_Life_Balance_Rating'],axis=1)
y=df['Work_Life_Balance_Rating']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
# print(X_train.shape) #4000,40
# print(X_test.shape) #1000,40
#feature scaling
# make the values of the target variable y zero-based (xgboost expects 0-based class values)
y_train=y_train-1
y_test=y_test-1
cols=X_train.columns
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
# print(type(X_train))  #array
X_train=pd.DataFrame(X_train,columns=cols)
X_test=pd.DataFrame(X_test,columns=cols)
# print(type(X_train)) #dataframe
########### adaboost ###########
abc=AdaBoostClassifier(n_estimators=100,learning_rate=1,random_state=0)
model_abc=abc.fit(X_train,y_train)
y_pred_abc_proba=model_abc.predict_proba(X_test)
# print('AdaBoost ROC-AUC score: {0:0.2f}'.format(roc_auc_score(y_test,y_pred_abc_proba,multi_class="ovr"))) #0,47
# prediction of AdaBoost ROC liness
fpr_abc, tpr_abc, _ = roc_curve(y_test, y_pred_abc_proba[:, 1], pos_label=1)
roc_auc_abc = auc(fpr_abc, tpr_abc)  # calculation of AUC
# plotting the AdaBoost and XGBoost ROC lines
plt.figure(figsize=(10, 6))
plt.plot(fpr_abc, tpr_abc, color='blue', lw=2, label=f'AdaBoost (AUC = {roc_auc_abc:.2f})')
# for Referance 45 degrees line
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Guess')
########### xgboost ###########
xgb=XGBClassifier(n_estimators=100,learning_rate=1,max_depth=3,random_state=0)
model_xgb=xgb.fit(X_train,y_train)
y_pred_xgb_proba=model_xgb.predict_proba(X_test)
# print('XGBoost ile ROC-AUC score:{0:0.2f}'.format(roc_auc_score(y_test,y_pred_xgb_proba,multi_class="ovr"))) # 0.48
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_xgb_proba[:, 1], pos_label=1)  # ROC eğrisi için
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)  # AUC hesabı
plt.plot(fpr_xgb, tpr_xgb, color='green', lw=2, label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Guess')
plt.title('AdaBoost ve XGBoost - ROC Eğrisi Karşılaştırması')
plt.xlabel('False Positive Rate (Yanlış Pozitif Oranı)')
plt.ylabel('True Positive Rate (Doğru Pozitif Oranı)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()



















