#ARABA FİYAT TAHMİNİ : archive.ics.uci.edu(bir sürü veri seti var)
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
# from SMarket import y_train
import warnings
warnings.simplefilter(action='ignore',category=Warning)
df=pd.read_csv('8_Lineer_Regreson_Gercek_Proje/data/Automobile.csv')
# print(df.head())
# print(df.info())
#print(df.describe()) #istatistiksel verileri görme
# print(df.shape) #(205,26)
# print(len(df))#toplam gözlem sayısı =205
# print(df.columns) #toplam sütun sayısı=26
# print(len(df.describe().columns))#toplam kaç adet numeric sütun var: 26 sütun var,bunun 25 i girdi, 1 i çıktı
#girdi sütunlarının 15 i sayısal , 10 u kategorik
#veri önişleme sürecin %80 ini oluşturur
#her sütun içindeki tekil(unique) veri adedini görelim
# for col in df.columns:
#     print(col,df[col].nunique())
#kategorik olan sütunlar için değerleri görelim
# for col in df.columns:
#     values=[]
#     #kategorik
#     if col not in df.describe().columns:
#         for val in df[col].unique():
#             values.append(val)
#         print("{0} -> {1}".format(col,values))
#model adlarını görelim
# print(df.CarName)
manufacturer = df['CarName'].apply(lambda x: x.split(' '))
# print(manufacturer)
manufacturer = df['CarName'].apply(lambda x: x.split(' ')[0])
data=df.copy()
# print(data.head())
#carname sütununu datadan çıkaralım
data.drop(columns=['CarName'],axis=1,inplace=True)
#manufacturer sütununu ekle
data.insert(3,'manufacturer',manufacturer)
#hangi üreticinin kaç adet aracı var onu görelim
data.groupby(by='manufacturer').count() #gelen sütunlar yanlış olabiliyor onları inceleyip düzeltelim
#önce bütün değerleri görelim
data.manufacturer.unique()
#büyük küçük harfleri düzeltelim
data.manufacturer=data.manufacturer.str.lower()
#hatalı marka adlarını düzeltelim
data.replace({
    'maxda':'mazda',
    'porcshce':'porsche',
    'toyouta':'toyota',
    'voksvagen':'vm',
    'volkswagen':'vw'
},inplace=True)
# print(data.manufacturer.unique())
#tekli(univariate) analiz:değişkenlere tek tek kendi içlerinde bakıp nasıl göründüklerini inceleyelim
#symboling -> sigorta riski
# sns.countplot(data.symboling)
# plt.show()
# fig=plt.figure(figsize=(20,12))
# plt.subplot(2,3,1)
# plt.title('Fueltype')
# sns.countplot(data.fueltype) #benzinli(gas) arabalar çoğunlukta
# plt.subplot(2,3,2)
# plt.title('Fuelsystem')
# sns.countplot(data.fuelsystem) #mpfi (multi point fuel injection)en çok tercih edilen yeni teknoloji
# plt.subplot(2,3,3)
# plt.title('Aspiration')
# sns.countplot(data.aspiration) #mpfi (multi point fuel injection)en çok tercih edilen yeni teknoloji #çoğunluk standart beslemeli
# plt.subplot(2,3,4)
# plt.title('Door Number')
# sns.countplot(data.doornumber)
# #çoğunluk 4 kapılı
# plt.subplot(2,3,5)
# plt.title('Car Body')
# sns.countplot(data.carbody) #mpfi (multi point fuel injection)en çok tercih edilen yeni teknoloji
# #çoğunluk sedan
# plt.subplot(2,3,6)
# plt.title('Drive Wheel')
# sns.countplot(data.drivewheel) #mpfi (multi point fuel injection)en çok tercih edilen yeni teknoloji
# plt.show() #çekiş sistemi standart çeker çoğunlukta
#ikili(bivariate)analiz:değişkenlerin fiyatı nasıl etkilediğini görelim
#üretici bazlı ortalama fiyatlar
# plt.figure(figsize=(16,8))
# plt.title('Üretici fiyatları',fontsize=16)
# sns.barplot(x=data.manufacturer, y=data.price,hue=data.fueltype,palette='Set2')
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.show()
#symboling
# plt.figure(figsize=(8,6))
# sns.boxplot(x=data.symboling, y=data.price)
# plt.show()
#fueltype
# plt.figure(figsize=(8,6))
# sns.boxplot(x=data.fueltype, y=data.price)
# plt.show()
#enginelocation
# plt.title('Engine Location',fontsize=16)
# sns.countplot(data.enginelocation)
# sns.boxplot(x=data.enginelocation, y=data.price)
# plt.show() #çoğunlukla motoru önde olan arabalar var ve fiyatları çok düşük
#cylindernumber
# plt.title('Cylinder Number')
# sns.countplot(data.cylindernumber)
# sns.boxplot(x=data.cylindernumber, y=data.price)
# plt.show()#silindir sayısı arttıkça fiyatın görece arttığını söyleyebiliriz
#Fiyatın kendi içinde dağılımına bakacağız :sadece  fiyatın nasıl kümelendiğine bakalım
# sns.displot(data.price)
# plt.show() #fiyatın genelde 5000 ile 20000 USD arasında dağıldığı görülür.
# print(data.price.describe())
#ikili grafikler(pair-plot)
# print(data.columns)
cols=['wheelbase','carlength','carwidth', 'carheight','curbweight','enginesize','boreratio','stroke',
      'compressionratio','horsepower','peakrpm','citympg','highwaympg']
#regresyon doğruları ile ilişkiyi görelim
# plt.figure(figsize=(20,25))
# for i in range(len(cols)):
#     plt.subplot(5,3,i+1)
#     plt.title(cols[i] + ' - Fiyat ')
#     sns.regplot(x=eval('data' + '.'+ cols[i]),y=data.price) #regresyon doğruları
# plt.tight_layout()
# plt.show() #burada neredeyse tüm değişkenler fiyat üzerinde önemli
# etkisiz olanlar:carheight,stroke, compressionratio,peakrpm,highwaympg,citympg bunları çıkarıyoruz.
data_yeni=data[['car_ID', 'symboling', 'fueltype', 'manufacturer', 'aspiration', #yeni sütunlarla yeni data
       'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'wheelbase',
       'carlength', 'carwidth', 'curbweight', 'enginetype',
       'cylindernumber', 'enginesize', 'fuelsystem', 'boreratio', 'horsepower','price']]
# print(data_yeni.head())
#şimdi data_new içindeki tüm sütunlara ikili olarak (pair-plot)bakalım.
# sns.pairplot(data_yeni)
# plt.show()
#feature engineering(Tork hesaplama, aracın çekiş gücüyle ilgilidir.)
torque=data.horsepower * 5252 / data.peakrpm
data.insert(10,'torque',pd.Series(data.horsepower * 5252 / data.peakrpm,index=data.index))
#bu yeni sütun(torque) ile fiyat arasındaki ilişkiyi bakalım
# plt.title('Torque - Fiyat' ,fontsize=16)
# sns.regplot(x=data.torque,y=data.price)
# plt.show()
#Tork dağılımını görelim
# plt.title('Torque Dağılımı' ,fontsize=18)
# sns.distplot(data.torque)
# plt.show()
#yakıt ekonomisi sütunu yok onu ekleyelim (fueleconomy):arabanın şehir içinde ve dışında ortalama yakıt tüketimi.
data['fueleconomy']=(0.55*data.citympg) + (0.45 * data.highwaympg)
# print(data.fueleconomy)
#model için önemli değişkenleri silip diğerlerini bırakalım
data.drop(columns=['car_ID','manufacturer','doornumber','carheight','compressionratio',
                   'symboling','stroke','citympg','highwaympg','fuelsystem','peakrpm'],
          axis=1,inplace=True)
# print(data.head())
#Model Tanımlama
cars=data.copy()
#kategorik değişkenler için dummy variable ları alalım:
dummies_list=['fueltype','aspiration','carbody','drivewheel',
              'enginelocation','enginetype','cylindernumber']#kategorik sütunlar
for i in dummies_list:
    temp_df=pd.get_dummies(eval('cars' + '.'+i),drop_first=True)
    cars=pd.concat([cars,temp_df],axis=1)
    cars.drop([i],axis=1,inplace=True)
train_data,test_data=train_test_split(cars,train_size=0.7,random_state=42)
# print(train_data.head())
scaler=MinMaxScaler()#scaler nesnesi oluşturma
#price hariç, y değişkeni scale edilmez. #numeric kolonları scale edelim
scale_cols=[ 'wheelbase','torque','carlength', 'carwidth', 'curbweight','enginesize',
             'fueleconomy', 'boreratio', 'horsepower']
train_data[scale_cols]=scaler.fit_transform(train_data[scale_cols])
# print(train_data.head())
y_train=train_data.pop('price')
# print(y_train.head())
X_train=train_data
#çoklu lineer regresyon için kütüphaneleri import edelim
lr=LinearRegression()
lr.fit(X_train,y_train) #fit:Lineer regresyonu veri ile train etmek demektir
#RFE'yi hazırla:RFE(estimator(tahminci/değerlendirici),n_features_to_select),geriye 10 adet değişken bırakacak şekilde RFE tanımlayalım
rfe=RFE(estimator=lr,n_features_to_select=10)
rfe=rfe.fit(X_train,y_train)#rfe yi train edelim
# print(rfe.support_) #10 tane True
# print(rfe.ranking_)
# print(list(zip(X_train.columns,rfe.support_,rfe.ranking_)))
#sadece seçilmiş sütunlar (değişkenler)
# print(X_train.columns[rfe.support_])
#önemli sütunları artık biliyoruz
X_train_rfe=X_train[X_train.columns[rfe.support_]] #en önemli olan 10 değişkenin içerisinden yeni bir data oluşturduk
# print(X_train_rfe)
#OLS ANALİZİ(kendi eleyeceklerimizi seçicez)
X_train_rfemodel=X_train_rfe.copy()
X_train_rfemodel=sm.add_constant(X_train_rfemodel)#statsmodels için add_constant -> beta_0 için 1'lerden oluşan sütun
# print(X_train_rfemodel.isnull().sum())
print(X_train_rfemodel.dtypes)
# X_train_rfemodel = X_train_rfemodel.astype(float)
# Sadece bool türündeki sütunları sayısala çevir
bool_columns = X_train_rfemodel.select_dtypes(include=['bool']).columns
X_train_rfemodel[bool_columns] = X_train_rfemodel[bool_columns].astype(int)
# print(X_train_rfemodel.shape)  # Özellik matrisi boyutu
# print(y_train.shape)  # Hedef vektör boyutu
# print(X_train_rfemodel.head())  # Veri yapısını kontrol edin
# print(X_train_rfemodel.shape)  # Şekli kontrol edin
lr = sm.OLS(y_train,X_train_rfemodel).fit()
# print(lr.summary())
#two değişkeni p den çok büyük, dolayısıyla önemsiz,çıkarabiliriz
# X_train_rfemodel=X_train_rfemodel.drop(['two'],axis=1) #ardından tekrar OLS çalıştıralım
def train_ols(X,y):
    X=sm.add_constant(X)
    lr=sm.OLS(y,X).fit()
    print(lr.summary())
# train_ols(X_train_rfemodel,y_train)
#R-squared %88 çok iyi bir doğruluk oranı
# X_train_rfemodel=X_train_rfemodel.drop(['dohcv'],axis=1)# p den büyük çıkardık
# train_ols(X_train_rfemodel,y_train)
# X_train_rfemodel=X_train_rfemodel.drop(['five'],axis=1)# p den büyük çıkardık
# train_ols(X_train_rfemodel,y_train) #değişkenlerin hepsi p den küçük o yüzden durabiliriz.
#Geri kalan değişkenlerin hepsi önemli
#KATSAYI ÖNEM SIRASI
# print(X_train.columns)
X_train_final=X_train[['curbweight','enginesize','horsepower','rear','four','six','twelve']]
lr_final=LinearRegression()
lr_final.fit(X_train_final,y_train)
# print(lr_final.coef_)
katsayilar=pd.DataFrame(lr_final.coef_,X_train_final.columns,columns=['katsayi'])
# print(katsayilar.sort_values(by='katsayi',ascending=False))
#SON ANALİZLER
#fiyatı en çok arttıran değişken enginesize
# plt.title('Engine Size')
# sns.regplot(x=data.enginesize,y=data.price)
# plt.show()
#rear değişkeni engine locationdan geliyor:enginelocation['front','rear] ; motorun arkada olması fiyatı arttırıyor.
# plt.title('Engine Location')
# sns.countplot(x=data.enginelocation)
# sns.boxplot(x=data.enginelocation,y=data.price)
# plt.show()
# plt.title('Curbweight') #aracın ağırlığı fiyatı arttırıyor.
# sns.regplot(x=data.curbweight,y=data.price)
# plt.show()
# plt.title('Horsepower')
# sns.regplot(x=data.horsepower,y=data.price)
# plt.show()
#four-six-twelve değişkenleri nerden geliyor:cylindirsayısı ndan geliyor.baz değişken -> eight(en yüksek fiyat üzerinden değerlendirdi)
plt.title('Cylinder Number')
sns.countplot(x=data.cylindernumber)
sns.boxplot(x=data.cylindernumber,y=data.price)
plt.show()

