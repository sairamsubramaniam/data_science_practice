

import numpy as np



# INPUTS
iters = 10
data = [[-1,-1, 1],[1,0,-1],[-1,1.5,1]]
# data = [[-1,-1, 1],[1,0,-1],[-1,10,1]]
# data = [[-4,2, 1],[-2,1,1],[-1,-1,-1],[2,2,-1],[1,-2,-1]]


# SEPARATE DEPENDENT & INDEPENDENT
X = np.array( [ d[:-1] for d in data ] )
Y = np.array( [ d[-1] for d in data ] )


# INITIALIZE WEIGHTS & BIASES
theta = np.zeros(  ( len(data[0])-1 )  )
theta_0 = 0


# PERCEPTRON ALGORITHM
def perceptron(X, Y, theta, string, theta_0=0, iters=5):
    print("--------------------- "+string+" ---------------------")
    print(X[0], X[1], X[2], Y)
    cntr = 0
    for iter_num in range(iters):
        print("ITER ", cntr)
        for i in range(len(Y)):
            print("THETA_0 and THETA BEFORE: ", theta_0, "  |  ", theta)
            pred = (  theta.T @ X[i] + theta_0)
            loss = ( Y[i] * pred )
            print("Y: ", Y[i], " ; PRED: ", pred, " ; LOSS: ", loss)
            if loss <= 0:
                theta = theta + Y[i] * X[i]
                theta_0 += Y[i]
            print("THETA_0 and THETA AFTER: ", theta_0, "  |  ", theta)
        cntr += 1
    print("======================================================")
    print()
    print()
    return theta, theta_0


a1, a2 = perceptron(X=X, Y=Y, theta=np.zeros(  ( len(data[0])-1 )  ), string="CASE 1")
b1, b2 = perceptron(X=[X[1],X[2],X[0]], Y=[Y[1],Y[2],Y[0]], theta=np.zeros(  ( len(data[0])-1 )  ), string="CASE 2")
c1, c2 = perceptron(X=[X[2],X[0],X[1]], Y=[Y[2],Y[0],Y[1]], theta=np.zeros(  ( len(data[0])-1 )  ), string="CASE 3")

print(a1)
print(b1)
print(c1)
print(a2,b2,c2)
# print(a2)

