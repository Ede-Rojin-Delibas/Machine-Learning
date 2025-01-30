#CAR PRICE PREDICTION : archive.ics.uci.edu(So many datasets in this site)
# import inline
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import warnings
warnings.simplefilter(action='ignore',category=Warning)
df=pd.read_csv('8_Lineer_Regreson_Gercek_Proje/data/Automobile.csv')
print(df.head())
print(df.info())
print(df.describe()) #For see the statistical datas
print(df.shape) #(205,26)
print(len(df))#total observation number=205
print(df.columns) #total colon number=26
print(len(df.describe().columns)) #How many total numerical columns do we have?
# In total 26 colon and the 25 of it is input, 1 is output
#The 15 of input colons are numerical ,10 is categorical
#Let's see the number of unique data in each column.
for col in df.columns:
    print(col,df[col].nunique())
#The values for categorical colons
for col in df.columns:
    values=[]
#     #kategorik
#     if col not in df.describe().columns:
#         for val in df[col].unique():
#             values.append(val)
#         print("{0} -> {1}".format(col,values))
#Let's see the model names
# print(df.CarName)
manufacturer = df['CarName'].apply(lambda x: x.split(' '))
# print(manufacturer)
manufacturer = df['CarName'].apply(lambda x: x.split(' ')[0])
data=df.copy()
# print(data.head())
#Remove the carname column
data.drop(columns=['CarName'],axis=1,inplace=True)
#Add manufacturer columns
data.insert(3,'manufacturer',manufacturer)
#Let'S see how many car the producers have
data.groupby(by='manufacturer').count() #It might be wrong, so examine it anyway.
#Let's see all values
data.manufacturer.unique()
#Correcting the lower and upper letters
data.manufacturer=data.manufacturer.str.lower()
#Correcting brand names
data.replace({
    'maxda':'mazda',
    'porcshce':'porsche',
    'toyouta':'toyota',
    'voksvagen':'vm',
    'volkswagen':'vw'
},inplace=True)
# print(data.manufacturer.unique())
#univariate(tekli) analysis: Let's look at the variables one by one and see how they look.
#symboling -> sigorta riski
sns.countplot(data.symboling)
# plt.show()
fig=plt.figure(figsize=(20,12))
plt.subplot(2,3,1)
plt.title('Fueltype')
sns.countplot(data.fueltype) #with oil(gas) cars are mostly
plt.subplot(2,3,2)
plt.title('Fuelsystem')
sns.countplot(data.fuelsystem) #mpfi (multi point fuel injection)mostly chosed new tech
plt.subplot(2,3,3)
plt.title('Aspiration')
sns.countplot(data.aspiration) #mpfi (multi point fuel injection)mostly chosed new tech is standard
plt.subplot(2,3,4)
plt.title('Door Number')
sns.countplot(data.doornu1)
#mostly with 4 doors
plt.subplot(2,3,5)
plt.title('Car Body')
sns.countplot(data.carbody) #mpfi (multi point fuel injection)
# mostly sedan
# plt.subplot(2,31
sns.countplot(data.drivewheel) #mpfi (multi point fuel injection)
# plt.show() #The traction system is mostly standard traction.
#bivariate(ikili)analysis :Let's see the effect of price to the variables
#producer based average prices
plt.figure(figsize=(16,8))
plt.title('Üretici fiyatları',fontsize=16)
sns.barplot(x=data.manufacturer, y=data.price,hue=data.fueltype,palette='Set2')
plt.xticks(rotation=90)
plt.tight_layout()
# plt.show()
#symboling
plt.figure(figsize=(8,6))
sns.boxplot(x=data.symboling, y=data.price)
# plt.show()
#fueltype
plt.figure(figsize=(8,6))
sns.boxplot(x=data.fueltype, y=data.price)
# plt.show()
#enginelocation
plt.title('Engine Location',fontsize=16)
sns.countplot(data.enginelocation)
sns.boxplot(x=data.enginelocation, y=data.price)
# plt.show() #There are mostly cars with front engines and their prices are very low.
#cylindernumber
plt.title('Cylinder Number')
sns.countplot(data.cylindernumber)
sns.boxplot(x=data.cylindernumber, y=data.price)
# plt.show()#We can say that the price increases relatively as the number of cylinders increases.
#We will look at the distribution of the price itself; let's just look at how the price is clustered
sns.displot(data.price)
# plt.show() #The price is generally seen to be between 5000 and 20000 USD.
print(data.price.describe())
#Bivariate graphs(pair-plot)
# print(data.columns)
cols=['wheelbase','carlength','carwidth', 'carheight','curbweight','enginesize','boreratio','stroke',
      'compressionratio','horsepower','peakrpm','citympg','highwaympg']
#regression lines with relations
plt.figure(figsize=(20,25))
for i in range(len(cols)):
    plt.subplot(5,3,i+1)
    plt.title(cols[i] + ' - Fiyat ')
    sns.regplot(x=eval('data' + '.'+ cols[i]),y=data.price)
plt.tight_layout()
# plt.show() #In here almost all variables are important on price
# uneffective ones:carheight,stroke, compressionratio,peakrpm,highwaympg,citympg let's remove them.
data_yeni=data[['car_ID', 'symboling', 'fueltype', 'manufacturer', 'aspiration', #new data with new colons
       'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'wheelbase',
       'carlength', 'carwidth', 'curbweight', 'enginetype',
       'cylindernumber', 'enginesize', 'fuelsystem', 'boreratio', 'horsepower','price']]
# print(data_yeni.head())
#pair-plot
sns.pairplot(data_yeni)
# plt.show()
#feature engineering(Calculating Torque,It is related to the traction power of the vehicle.)
torque=data.horsepower * 5252 / data.peakrpm
data.insert(10,'torque',pd.Series(data.horsepower * 5252 / data.peakrpm,index=data.index))
#torque - price (Their relationship)
plt.title('Torque - Fiyat' ,fontsize=16)
sns.regplot(x=data.torque,y=data.price)
# plt.show()
#Tork correlation
plt.title('Torque Dağılımı' ,fontsize=18)
sns.distplot(data.torque)
# plt.show()
#Let's add fueleconomy : It is the verage fuel consumption of the car inside and outside the city.
data['fueleconomy']=(0.55*data.citympg) + (0.45 * data.highwaympg)
# print(data.fueleconomy)
#Let's delete the important varibles for model and look at the changes
data.drop(columns=['car_ID','manufacturer','doornumber','carheight','compressionratio',
                   'symboling','stroke','citympg','highwaympg','fuelsystem','peakrpm'],
          axis=1,inplace=True)
# print(data.head())
#Introduction of Model
cars=data.copy()
#Let's take dummy variables for categorical variables
dummies_list=['fueltype','aspiration','carbody','drivewheel',
              'enginelocation','enginetype','cylindernumber']   #categorical colons
for i in dummies_list:
    temp_df=pd.get_dummies(eval('cars' + '.'+i),drop_first=True)
    cars=pd.concat([cars,temp_df],axis=1)
    cars.drop([i],axis=1,inplace=True)
train_data,test_data=train_test_split(cars,train_size=0.7,random_state=42)
# print(train_data.head())
scaler=MinMaxScaler()#creating the scaler object
#except price cause the y variable is not scaled.
scale_cols=[ 'wheelbase','torque','carlength', 'carwidth', 'curbweight','enginesize',
             'fueleconomy', 'boreratio', 'horsepower']
train_data[scale_cols]=scaler.fit_transform(train_data[scale_cols])
# print(train_data.head())
y_train=train_data.pop('price')
# print(y_train.head())
X_train=train_data
lr=LinearRegression()
lr.fit(X_train,y_train) #fit:It meana training the Linear regression with data
#prepare RFE:RFE(estimator(tahminci/değerlendirici),n_features_to_select), Let's define RFE to leave 10 variables remaining.
rfe=RFE(estimator=lr,n_features_to_select=10)
rfe=rfe.fit(X_train,y_train)#Let's train rfe
print(rfe.support_) #10 True
print(rfe.ranking_)
print(list(zip(X_train.columns,rfe.support_,rfe.ranking_)))
#Just choosed colons(değişkenler)
# print(X_train.columns[rfe.support_])
#Now we know the important colons
X_train_rfe=X_train[X_train.columns[rfe.support_]]
# print(X_train_rfe)
#OLS ANALYSIS(We will choose our own eliminations.)
X_train_rfemodel=X_train_rfe.copy()
X_train_rfemodel=sm.add_constant(X_train_rfemodel)#add_constant for statsmodels  -> a column that consist of 1s for beta_0
print(X_train_rfemodel.isnull().sum())
print(X_train_rfemodel.dtypes)
X_train_rfemodel = X_train_rfemodel.astype(float)
# Just turn the type bool colons to numeric
bool_columns = X_train_rfemodel.select_dtypes(include=['bool']).columns
X_train_rfemodel[bool_columns] = X_train_rfemodel[bool_columns].astype(int)
print(X_train_rfemodel.shape)  # feature matrix' shape
print(y_train.shape)  # target vector's shape
print(X_train_rfemodel.head())  # check the data structure
print(X_train_rfemodel.shape)  # chech the shape
lr = sm.OLS(y_train,X_train_rfemodel).fit()
# print(lr.summary())
#The variable two is much larger than p, so it is unimportant, we can remove it.
X_train_rfemodel=X_train_rfemodel.drop(['two'],axis=1) #then run OLS again
def train_ols(X,y):
    X=sm.add_constant(X)
    lr=sm.OLS(y,X).fit()
    print(lr.summary())
train_ols(X_train_rfemodel,y_train)
# R-squared is %88 (it is a really good accuracy
X_train_rfemodel=X_train_rfemodel.drop(['dohcv'],axis=1)# It s bigger than p so we can remove it
train_ols(X_train_rfemodel,y_train)
X_train_rfemodel=X_train_rfemodel.drop(['five'],axis=1)
train_ols(X_train_rfemodel,y_train)
#All the remaining variables are important.
#coefficient importance order
# print(X_train.columns)
X_train_final=X_train[['curbweight','enginesize','horsepower','rear','four','six','twelve']]
lr_final=LinearRegression()
lr_final.fit(X_train_final,y_train)
# print(lr_final.coef_)
katsayilar=pd.DataFrame(lr_final.coef_,X_train_final.columns,columns=['katsayi'])
print(katsayilar.sort_values(by='katsayi',ascending=False))
#LAST ANALYSIS
#The variable that increases the price the most is enginesize
plt.title('Engine Size')
sns.regplot(x=data.enginesize,y=data.price)
# plt.show()
#rear variable comes from engine location:enginelocation['front','rear] ; Having the engine at the back increases the price.
plt.title('Engine Location')
sns.countplot(x=data.enginelocation)
sns.boxplot(x=data.enginelocation,y=data.price)
# plt.show()
plt.title('Curbweight') #The weight of the car increases the price
sns.regplot(x=data.curbweight,y=data.price)
# plt.show()
plt.title('Horsepower')
sns.regplot(x=data.horsepower,y=data.price)
# plt.show()
#four-six-twelve (They come from cylindirsayısı).base variable -> eight(The highest price and evaluated it from that)
plt.title('Cylinder Number')
sns.countplot(x=data.cylindernumber)
sns.boxplot(x=data.cylindernumber,y=data.price)
plt.show()

