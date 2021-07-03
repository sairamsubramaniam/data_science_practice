
import copy
import numpy as np



# INPUTS
iters = 10
# data = [[-1,-1, 1],[1,0,-1],[-1,1.5,1]]
# data = [[-1,-1, 1],[1,0,-1],[-1,10,1]]
# data = [[-4,2, 1],[-2,1,1],[-1,-1,-1],[2,2,-1],[1,-2,-1]]
data = [(0,0,-1), (2,0,-1), (3,0,-1), (0,2,-1), (2,2,-1), (5,1,1), (5,2,1), (2,4,1), (4,4,1), (5,5,1)]


# SEPARATE DEPENDENT & INDEPENDENT
X = np.array( [ d[:-1] for d in data ] )
Y = np.array( [ d[-1] for d in data ] )


# INITIALIZE WEIGHTS & BIASES
theta = np.zeros(  ( len(data[0])-1 )  )
theta_0 = 0


# PERCEPTRON ALGORITHM
def perceptron(X, Y, theta, string, theta_0=0, iters=20):
    cntr = 0
    for iter_num in range(iters):
        print("ITER ", cntr)
        for i in range(len(Y)):
            theta0_before = copy.deepcopy(theta_0)
            theta_before = copy.deepcopy(theta)
            pred = (  theta.T @ X[i] + theta_0)
            loss = ( Y[i] * pred )
            if loss <= 0:
                theta = theta + Y[i] * X[i]
                theta_0 += Y[i]
            if theta0_before != theta_0:
                print(i, " :  MISTAKE!!")
            print("********* EPOCH COMPLETE ************")
        cntr += 1
    print("======================================================")
    print()
    print()
    return theta, theta_0


def d_hinge_loss(x, y, pred, theta):
    loss = max(1 - ( y * pred ), 0)
    if loss <= 0:
        d = theta
    else:
        d = -y*x - theta

    return d



def reg_perceptron(X, Y, theta, string, theta_0=0, iters=20):
    cntr = 0
    for iter_num in range(iters):
        print("ITER ", cntr)
        for i in range(len(Y)):
            pred = (  theta.T @ X[i] + theta_0)
            d = d_hinge_loss(x=X[i], y=Y[i], pred=pred, theta=theta)
            theta = theta - d
            theta_0 += Y[i]
            print("Updated: ", theta, "  |  ", theta_0)

            print("********* EPOCH COMPLETE ************")
        cntr += 1
    print("======================================================")
    print()
    print()
    return theta, theta_0



# a1, a2 = reg_perceptron(X=X, Y=Y, theta=np.zeros(  ( len(data[0])-1 )  ), string="CASE 1")
# print(a1, a2)


# a1, a2 = perceptron(X=X, Y=Y, theta=np.zeros(  ( len(data[0])-1 )  ), string="CASE 1")
# b1, b2 = perceptron(X=[X[1],X[2],X[0]], Y=[Y[1],Y[2],Y[0]], theta=np.zeros(  ( len(data[0])-1 )  ), string="CASE 2")
# c1, c2 = perceptron(X=[X[2],X[0],X[1]], Y=[Y[2],Y[0],Y[1]], theta=np.zeros(  ( len(data[0])-1 )  ), string="CASE 3")

# print(a1, a2)
# print(b1)
# print(c1)
# print(a2,b2,c2)
# print(a2)

