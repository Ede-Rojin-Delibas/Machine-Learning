import pandas as pd
import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold,cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression

train_data=pd.read_csv('14_Model_Secimi/data/housing/train.csv')
print(train_data.head())
#delete the missing lines
train_data.dropna(axis=0,subset=['SalePrice'],inplace=True)
print(train_data.head())
#Delete the null columns
train_data.drop(['LotFrontage','GarageYrBlt','MasVnrArea','Alley','PoolQC','MiscFeature'],axis=1,inplace=True)
print(train_data.head())
#take the target variable
y=train_data['SalePrice']
#remove the target
train_data.drop(['SalePrice'],axis=1,inplace=True)
#Just take the numerical columns
numeric_cols=[cname for cname in train_data.columns if train_data[cname].dtype in ['int64','float64']]
X=train_data[numeric_cols].copy()
print(X)
print("The Input data's shape :{} and the target variable's shape:{}".format(X.shape,y.shape)) #input:(1460,34); target/y:(1460,)
#for k=5 , k->n_split
kf=KFold(n_splits=5,random_state=42,shuffle=True)
cnt=1
#split metodu  Train-Test olarak ayırmak için bize indeksleri döner.
# for train_index,test_index in kf.split(X,y):
#     print(f'Fold:{cnt},Train set:{len(train_index)},Test set:{len(test_index)}')
#     cnt+=1
#RMSE yi hesaplamak için  -(eksi) ile çarpacağız.
#cross_val_score dan bize negatif gelecek
def rmse(score):
    rmse=np.sqrt(-score)
    print(f'RMSE:{rmse}')
#lineer regresyon modeli
score=cross_val_score(LinearRegression(),X,y,cv=kf,scoring="neg_mean_squared_error")
# print(f'her bir fold için skor:{score}')
# print(rmse(score.mean()))
# print(np.sqrt(-sum(score) / len(score)))





