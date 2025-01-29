import numpy as np
import matplotlib.pyplot as plt
X=2 * np.random.rand(100,1)
print(X)
y=4+3*X+np.random.rand(100,1)
print(y)
plt.scatter(X,y)
plt.xlabel('X')
plt.ylabel('y')
plt.show() #It has an linear relationship between x ve y
def cal_cost(W,X,y):
    n=len(y)
    #First calculate the prediction y^=XW
    # X=(n,p)
    # W=(p,1)
    # y^=(n,1)
    tahmin=X.dot(W)
    #Calculate the cost function values ->j
    cost=(1/2*n) * np.sum(np.square(tahmin-y))
    #Take the cost value
    return cost
#GRADIENT DESCENT FUNCTION
def gradient_descent(X,y,W,learning_rate=0.01,iterations=100):
    '''Parameters:
    X=X matrix(the bias unit added form,so added a column to the first kolon and the
    values consist of 1s.
    y=y vector,
    W=coefficient vector(consist of w's)
    learning_rate=alpha (learning coefficient)
    iterations=total loop number
    Return:
    *The final state of W
    *cost list(cost history)
    *The list of W(weight history)
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
#We need to add a column to the first column of w0 and the values are 1s(bias)
n=len(X)
X_b=np.c_[np.ones((n,1)),X] #concaneta method (numpy)
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
#The real coefficients are 4 and 3.
#LEARNING RATE
#The function that plots the cost change
def cost_vs_iterations(cost_history,n_iter):
    fig,ax=plt.subplots(figsize=(8,6))
    plt.plot(range(n_iter),cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('J(W)')
    plt.title('Cost - Iterations')
    plt.grid()
    plt.show()
# cost_vs_iterations(cost_history,n_iter)
#It's a function for changing the alpha value
def call_gradient(learning_rate,n_iter):
    W_final,cost_history,w_history=gradient_descent(X_b,y,W,learning_rate,n_iter)
    cost_vs_iterations(cost_history,n_iter)
    #experiment1: lr is constant and make the iteration number 2000

# learning_rate=0.01
# n_iter=2000
# call_gradient(learning_rate, n_iter) #Nothing has changed;Making the iteration number 500 has also changed nothing
#experiment2:n_iter=1000 and learning_rate=0.05
# learning_rate=0.05
# n_iter=1000
# call_gradient(learning_rate, n_iter) #It closed to the result super fast, in 40. iteration
#Making the lr 0.1 gives the better result.
#What happened when alpha is 1?
#Cost's increased : OVERSHOOT
#Stochastic Gradient Descent Function
def stochastic_gradient_descent(X,y,W,learning_rate=0.01,iterations=100):
    '''Parameters:
            X=X matrix(The bias unit has been added, that is the first column consisting of 1's has been added
            y=y vector,
            W=Coefficient vector(consist of ws)
             learning_rate=alpha (learning coefficient)
            iterations=total loop number
            Return:
            *The final state of W
            *cost list(cost history)
            *The list of W(weight history)
        '''
    n = len(y)
    cost_history = np.zeros(iterations)
    for it in range(iterations):
        cost=0
        #I'll chose a X_i value in every level and calculate the cost.
        for i in range(n):
            #Take a random value
            rand_ind=np.random.randint(0,n)
            X_i=X[rand_ind,:].reshape(1,X.shape[1])
            y_i=y[rand_ind].reshape(1,1)
            #Just a prediction for X_i
            tahmin=np.dot(X_i,W)
            #One coefficient change for X_i
            W=W - (1/n) * learning_rate * (X_i.T.dot((tahmin-y_i)))
            #Add cost to cost for one calculation just for X_i
            cost+=cal_cost(W,X_i,y_i)
        #For this iteration add the cost value to the cost_history
        cost_history[it]=cost
    return W,cost_history

#Mini-Batch Stochastic Gradient Descent
def minibatch_stochastic_gradient_descent(X,y,W,learning_rate=0.01,n_iter=100,batch_size=20):
    '''Parameters:
            batch_size: Turn over the X's in 20 turns
            X=X matrix(The bias unit has been added, that is the first column consisting of 1's has been added
            y=y vector,
            W=katsayı vektörü (w'lerden oluşmuş)
            learning_rate=alpha (learning coefficient)
            iterations=total loop number
            Return:
            *The final state of W
            *cost list(cost history)
            *The list of W(weight history)
    '''
    n = len(y)
    cost_history = np.zeros(iterations)
    n_batches=int(n/batch_size)
    for it in range(iterations):
        cost=0
        #Mix the X and y's (Let the data orders be random) We mixed it because do not came in the same batch.Cause that's a possibility.
        indices=np.random.permutation(n)
        X=X[indices]
        y=y[indices]
        #each time we will take a random batch X_i and calculate the cost
        for i in range(0,n,batch_size):
            #Random X_i and y_i
            X_i=X[i:i+batch_size]
            y_i = y[i:i + batch_size]
            #Add a bias column(1s column)
            x_i=np.c_[np.ones(len(X_i)),X_i]
            #Prediction for X_i
            tahmin=np.dot(x_i,W)
            #Coefficient change for X_i s
            W=W - (1/n) * learning_rate * (X_i.T.dot((tahmin-y_i)))
            #Add the calculation X_i for cost to the cost value
            cost += cal_cost(W, X_i, y_i)
        #For this iteration add the cost value to the cost_history
        cost_history[it] = cost
        return W, cost_history


























