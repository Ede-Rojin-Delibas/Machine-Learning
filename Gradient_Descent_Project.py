import Gradient_Descent
import numpy as np
import matplotlib.pyplot as plt

from Gradient_Descent import stochastic_gradient_descent

X=2 * np.random.rand(100,1)
# print(X)
error=np.random.randn(100,1)
# print(error)
y= 4 + 3 * X + error
# print(y)
plt.plot(X,y,'b.')
plt.xlabel("$x$",fontsize=15)
plt.ylabel("$y$",rotation=0, fontsize=15)
_=plt.axis([0,2,0,15])#axis ile grafik eksen sınırlarını ayarladı . grafiğin x ve y eksenindeki sınırları belirledi.
#X e bias sütunu ekle
X_b=np.c_[np.ones((100,1)),X]
#Katsayı tahminlerini hesapla -> numpy ile çöz
w_best=np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) #W yu hesaplıyor
# print(w_best)
#yeni data geldiğinde tahmin yap:
X_new=np.array([[0],[2]])
# print(X_new)
#bias sütunu ekle
X_new_b=np.c_[np.ones((2,1)),X_new]
# print(X_new_b)
#tahmin yap, elimizde optimum w değerleri var ->w_best
y_pred=X_new_b.dot(w_best)
# print(y_pred)
#tahmin doğrusu
# plt.plot(X_new,y_pred,'r-')
# plt.plot(X,y,'b.') #bütün data
# plt.xlabel("$x_1$", fontsize=15)
# plt.ylabel("$y$",rotation=0, fontsize=15)
# plt.axis([0,2,0,15])
# plt.show()
# cal_cost=Gradient_Descent.cal_cost
lr=0.01
n_iter=1000
W=np.random.randn(2,1) #başlangıç için W vektörü
#X vektörüne bias column ekle
X_b=np.c_[np.ones((len(X),1)),X]
#gradient descent çalıştır
W,cost_history,w_history=Gradient_Descent.gradient_descent(X_b,y,W,lr,n_iter)
print('W0: {:0.3f}'.format(W[0][0]))
print('W1: {:0.3f}'.format(W[1][0]))
print('Final Cost/MSE: {:0.3f}'.format(cost_history[-1]))
#iterasyonları çizelim
fig,ax=plt.subplots(figsize=(8,6))
ax.set_ylabel('J(W)')
ax.set_xlabel('Iterations')
_=ax.plot(range((n_iter)),cost_history,'b.')
# plt.show()
fig,ax=plt.subplots(figsize=(8,6))
_=ax.plot(range((200)),cost_history[:200],'b.')
# plt.show()
#gradient descent i çizen bir fonksiyon
def plot_GD(n_iter,lr,ax,ax1=None):
    """

    :param n_iter:iterasyon sayısı
    :param lr: Learning Rate
    :param ax:Gradient Descent çizmek için axis
    :param ax1:Cost history vs. Iterations grafiği için axis
    :return:
    """
    _ = ax.plot(X,y, 'b.')
    W=np.random.randn(2,1)
    tr=0.1
    cost_history=np.zeros(n_iter)
    for i in range(n_iter):
        pred_prev=X_b.dot(W)
        W,h,_=Gradient_Descent.gradient_descent(X_b,y,W,lr,1)
        pred=X_b.dot(W)
        cost_history[i]=h[0]
        if i%25==0:
            _ = ax.plot(X,pred,'r-',alpha=tr)
            if tr<0.8:
                tr+=0.2
    if not ax1 == None:
        _ = ax1.plot(range(n_iter),cost_history,'b.')

# fig=plt.figure(figsize=(30,25))
# fig.subplots_adjust(hspace=0.4,wspace=0.4)
#iterasyon ve learning rate listesi
it_lr=[(2000,0.001),(500,0.01),(200,0.05),(100,0.1)]
count=0
for n_iter,lr in it_lr:
    count+=1
    ax=fig.add_subplot(4,2,count)
    count+=1
#     ax1=fig.add_subplot(4,2,count)
#     ax.set_title('lr:{}'.format(lr))
#     ax1.set_title('Iterations:{}'.format(n_iter))
#     plot_GD(n_iter,lr,ax,ax1)
# _,ax = plt.subplots(figsize=(14,10))
# plot_GD(100,0.1,ax)
# plt.show()
#stochastic gradient descent
# lr=0.5
# n_iter=50
# W = np.random.randn(2, 1)
# X_b=np.c_[np.ones((len(X),1)),X]
# gradient descent çalıştır
# W,cost_history=Gradient_Descent.stochastic_gradient_descent(X_b,y,W,lr,n_iter)
# print('W0: {:0.3f}'.format(W[0][0]))
# print('W1: {:0.3f}'.format(W[1][0]))
# print('Final Cost/MSE: {:0.3f}'.format(cost_history[-1]))
#şimdi SGD Yİ grafik üzerinde görelim
# fig,ax=plt.subplots(figsize=(8,6))
# ax.set_ylabel('{J(W)}',rotation=0)
# ax.set_xlabel('Iterations')
# _=ax.plot(range((n_iter)),cost_history,'b.')
# plt.show()
#mini -batch gradient descent
lr=0.1
n_iter=200
W = np.random.randn(2, 1)
X_b=np.c_[np.ones((len(X),1)),X]
W,cost_history=Gradient_Descent.minibatch_stochastic_gradient_descent(X_b,y,W,lr,n_iter)
print('W0: {:0.3f}'.format(W[0][0]))
print('W1: {:0.3f}'.format(W[1][0]))
print('Final Cost/MSE: {:0.3f}'.format(cost_history[-1]))
#şimdi mini_stoc_gr_des grafiğini çizelim
fig,ax=plt.subplots(figsize=(10,8))
ax.set_ylabel('{J(W)}',rotation=0)
ax.set_xlabel('Iterations')
_=ax.plot(range((n_iter)),cost_history,'b.')
plt.show()


