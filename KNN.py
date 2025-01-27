import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#veri seti oluşturma
def veri_olustur():
    features = np.array(
        [[2.88,3.05],[3.1,2.45],[3.05,2.8],[2.9,2.7],[2.75,3.4],[3.23,2.9],
         [3.2,3.75],[3.5,2.9],[3.65,3.6],[3.35,3.3]]
    )
    labels=['A','A','A','A','A','B','B','B','B','B']
    return features,labels
#fonksiyonu çağırıp veriyi oluşturma
features,labels=veri_olustur()
# print('FEATURES:\n',features)
# print('LABELS:\n',labels)
features[0]
x_df=pd.DataFrame(features)
# for point in x_df[0]:
#     print(point)
#classları renkli olarak grafikte gösterme
# plt.figure(figsize=(5,5))
# plt.xlim(2.4,3.8)
# plt.ylim(2.4,3.8)
#mevcut data
# plt.scatter(x_df.illoc[:5,0],x_df.iloc[:5,1],c='b') #A sınıfı mavi
# plt.scatter(x_df.illoc[5:,0],x_df.iloc[5:,1],c='g')# B sınıfı yeşil
# #tahmin etmek istediğimiz nokta:[3.18,3.15]
# plt.scatter([3.18,3.15],c='r',marker='x')
# plt.show()
def Manhattan(x,y):
    d=np.sum(np.abs(x-y))
    return d
# x=np.array([3,5])
# print("x: ",x)
# y=np.array([6,9])
# print("y: ",y)
# d_man=Manhattan(x,y)
# print(d_man)
def euclidean(x,y):
    d=np.sqrt(np.sum(np.square(x-y)))
    return d
x=np.array([3,5])
# print("x: ",x)
y=np.array([6,9])
# print("y: ",y)
d_euc=euclidean(x,y)
# print(d_euc)
#adetlere göre çoğunluk sıralaması yapan fonksiyon
import operator
def cogunluk_yontemi(class_count):
    sorted_class_count=sorted(class_count.items(),key=operator.itemgetter(1),reverse=True)
    return sorted_class_count
#dict
arr={'A':3,'B':2,'C':6,'D':5}
cogunluk_yontemi(arr)#sınıfların sayısını büyükten küçüğe yazalım
def knn(test_data,train_data,labels,k):
    """knn uygulama fonksiyonu"""
    distances=np.array([]) #mesafeler için boş bir liste oluşturma
    for each_data in train_data: #euclidean distance ile mesafeleri hesapla
        d=euclidean(test_data,each_data)
        distances=np.append(distances,d)
        #mesafeleri sırala ve sıralı indexleri al
    sorted_distance_index=distances.argsort()
    sorted_distance=np.sort(distances)
    #yarıçapı hesapla(k-1 indexli eleman son elemandır)
    r=(sorted_distance[k] + sorted_distance[k-1])/2
    class_count={}
    #çoğunluk yöntemi
    for i in range(k):
        vote_label=labels[sorted_distance_index[i]]
        class_count[vote_label]=class_count.get(vote_label,0)+1
    #sınıf değeri yani seçilen class (label)
    final_label=cogunluk_yontemi(class_count)
    return final_label,r
#tahmin noktamız bizim test datamız olacak
test_data=np.array([3.18,3.15])
#knn fonksiyonunun çağırılması k=5 için
final_label,r=knn(test_data,features,labels,5)
print(final_label)
print(r)
#sonuçların görselleştirilmesi(k=5 için elde ettiğim yarıçap(r)için bir çember çizelim)
#Polar coodinates: x=r*costeta ,y=r*sinteta
def cember(r,a,b):
    theta=np.arange(0,2*np.pi,0.01)
    x=a+r*np.cos(theta)
    y=b+r*np.sin(theta)
    return x,y
#çember fonksiyonunu çağırıp, çemberin her bir noktası için x ve y değerlerini alalım
k_circle_x,k_circle_y=cember(r,3.18,3.15)
#class'ları renkli olarak grafikte görme
plt.figure(figsize=(5,5))
plt.xlim(2.4,3.8)
plt.ylim(2.4,3.8)
#mevcut data
plt.scatter(x_df.iloc[:5,0],x_df.iloc[:5,1],c="b")
plt.scatter(x_df.iloc[5:,0],x_df.iloc[5:,1],c="g")
#tahmin etmek istediğimiz nokta [3.18,3.15]
plt.scatter([3.18],[3.15],c='r',marker='x')
#çemberi çizelim
plt.plot(k_circle_x,k_circle_y)
plt.show()



