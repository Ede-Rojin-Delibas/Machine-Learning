import numpy as np
import matplotlib.pyplot as plt
X=2 * np.random.rand(100,1)
# print(X)
y=4+3*X+np.random.rand(100,1)
# print(y)
# plt.scatter(X,y)
# plt.xlabel('X')
# plt.ylabel('y')
# plt.show() #x ve y arasında lineer bir ilişki vardır.
def cal_cost(W,X,y):
    n=len(y)
    #önce tahmini hesapla y^=XW
    #X=(n,p)
    #W=(p,1)
    #y^=(n,1)
    tahmin=X.dot(W)
    #cost function değerlerini hesapla->j
    cost=(1/2*n) * np.sum(np.square(tahmin-y))
    #maliyet değerini al(cost)
    return cost
#GRADIENT DESCENT FONKSİYONU
def gradient_descent(X,y,W,learning_rate=0.01,iterations=100):
    '''Parametreler:
    X=X matrisi(bias unit eklenmiş hali, yani 1'lerden oluşan ilk sütun eklemiş
    y=y vektörü,
    W=katsayı vektörü (w'lerden oluşmuş)
    learning_rate=alpha (öğrenme katsayısı)
    iterations=toplam döngü sayısı
    Dönüş:
    *W vektörünün son hali
    *maliyet listesinin (cost history)
    *W vektörünün listesi(weight history)
    '''
    n = len(y)
    cost_history=np.zeros(iterations)
    w_history=np.zeros((iterations,2))
    for it in range(iterations):
        tahmin=np.dot(X,W)
        W=W-(1/n) * learning_rate * X.T.dot(tahmin-y)
        w_history[it, :] = W.T
        cost_history[it]=cal_cost(W,X,y)
    return W,cost_history,w_history
W=np.random.randn(2,1)
# print(W)
lr=0.01
n_iter=1000
#w0 a 1 lerden(bias) oluşan bir sütun eklememiz lazım
n=len(X)
X_b=np.c_[np.ones((n,1)),X] #concaneta methodu (numpy)
# print(X_b)
W_final,cost_history,w_history=gradient_descent(X_b,y,W,lr,n_iter)
# print(W_final)
# print(cost_history)
# print(cost_history[0])
# print(cost_history[1])
W_0=W[0]
W_0_final=W_final[0]
W_1=W[1]
W_1_final=W_final[1]
#gerçek katsayılar 4 ve 3 tür
#LEARNING RATE
#MALİYET DEĞİŞİMİNİ ÇİZEN FONKSİYON
def cost_vs_iterations(cost_history,n_iter):
    fig,ax=plt.subplots(figsize=(8,6))
    plt.plot(range(n_iter),cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('J(W)')
    plt.title('Cost - Iterations')
    plt.grid()
    plt.show()
# cost_vs_iterations(cost_history,n_iter)
#alpha değerini değiştirmek için bir fonksiyon
def call_gradient(learning_rate,n_iter):
    W_final,cost_history,w_history=gradient_descent(X_b,y,W,learning_rate,n_iter)
    cost_vs_iterations(cost_history,n_iter)
    #deney1: lr sabit kalsın iterasyon sayısını 2000 yapalım

# learning_rate=0.01
# n_iter=2000
# call_gradient(learning_rate, n_iter) #hiç bir şey değişmedi;iterasyon sayısını 500 yapmak da bir şeyi değiştirmedi
#deney2:n_iter=1000 ve learning_rate=0.05
# learning_rate=0.05
# n_iter=1000
# call_gradient(learning_rate, n_iter) #sonuca çok hızlı yaklaştı, 40. iterasyonda
#0.1 yapınca lr yi daha da iyi bir sonuç aldık
#alpha 1 olursa ne olur
#maliyet arttı : OVERSHOOT
#Stochastic Gradient Descent Fonksiyonu
def stochastic_gradient_descent(X,y,W,learning_rate=0.01,iterations=100):
    '''Parametreler:
        X=X matrisi(bias unit eklenmiş hali, yani 1'lerden oluşan ilk sütun eklemiş
        y=y vektörü,
        W=katsayı vektörü (w'lerden oluşmuş)
        learning_rate=alpha (öğrenme katsayısı)
        iterations=toplam döngü sayısı
        Dönüş:
        *W vektörünün son hali
        *maliyet listesinin (cost history)
        *W vektörünün listesi(weight history)
        '''
    n = len(y)
    cost_history = np.zeros(iterations)
    for it in range(iterations):
        cost=0
        #her seferinde rastgele bir X_i değeri seçip maliyet hesaplayacağız.
        for i in range(n):
            #rastgele bir değer al
            rand_ind=np.random.randint(0,n)
            X_i=X[rand_ind,:].reshape(1,X.shape[1])
            y_i=y[rand_ind].reshape(1,1)
            #tek X_i için tahmin
            tahmin=np.dot(X_i,W)
            #tek X_i için katsayı değişimi
            W=W - (1/n) * learning_rate * (X_i.T.dot((tahmin-y_i)))
            #tek X_i için hesaplanan cost u cost değişkenine ekle
            cost+=cal_cost(W,X_i,y_i)
        #bu iterasyon için cost değerini cost_history'ye ekle
        cost_history[it]=cost
    return W,cost_history

#Mini-Batch Stochastic Gradient Descent
def minibatch_stochastic_gradient_descent(X,y,W,learning_rate=0.01,n_iter=100,batch_size=20):
    '''Parametreler:
            batch_size: 20 şer olarak X ler üzerinde dön
            X=X matrisi(bias unit eklenmiş hali, yani 1'lerden oluşan ilk sütun eklemiş
            y=y vektörü,
            W=katsayı vektörü (w'lerden oluşmuş)
            learning_rate=alpha (öğrenme katsayısı)
            iterations=toplam döngü sayısı
            Dönüş:
            *W vektörünün son hali
            *maliyet listesinin (cost history)
            *W vektörünün listesi(weight history)
    '''
    n = len(y)
    cost_history = np.zeros(iterations)
    n_batches=int(n/batch_size)
    for it in range(iterations):
        cost=0
        #X ve y leri karıştır(veri sıraları rastgele olsun)aynı batch gelmesin diye karıştırdık. bu ihtimal var çünkü
        indices=np.random.permutation(n)
        X=X[indices]
        y=y[indices]
        #her seferinde rastgele bir X_i partisi alıp maliyet hesaplayacağız
        for i in range(0,n,batch_size):
            #rastgele X_i ve y_i partileri
            X_i=X[i:i+batch_size]
            y_i = y[i:i + batch_size]
            #bias column ekle 1 ler sütunu
            x_i=np.c_[np.ones(len(X_i)),X_i]
            #X_i partileri için tahmin
            tahmin=np.dot(x_i,W)
            #X_i partisi için katsayı değişimi
            W=W - (1/n) * learning_rate * (X_i.T.dot((tahmin-y_i)))
            #X_i partisi için hesaplanan cost u cost değişkenine ekle
            cost += cal_cost(W, X_i, y_i)
        # bu iterasyon için cost değerini cost_history'ye ekle
        cost_history[it] = cost
        return W, cost_history


























