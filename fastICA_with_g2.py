#import scipy as sp
import numpy as np
#import matplotlib.pyplot as plt


# g2(X) = X * exp(-1/2 * X**2)               # g1(X) = tanh(X)
def g(X):
    return X * np.exp((-1/2) * X**2)   

# g2'(X) = (1 - X**2) * exp(-1/2 * X**2)     # g1'(X) = 1 - tanh^2 (X)
def g_prime(X):
    return (1 - X**2) * np.exp((-1/2) * X**2)  

# some preprocessing techniques that make the problem of ICA estimation simpler 
# and better conditioned.

def preprocessing(X):
    # 1/ Centering (X): subtract its mean m = E{X} so as to make X a zero-mean variable
    
    mean = np.mean(X, axis=1, keepdims=True, dtype=np.uint8)
    X -= mean
    
    # 2/ whitening (X): e transform the observed variable X linearly so that we obtain 
    # a new  X˜ which is white, its components are uncorrelated and their variances equal unity
    
    cov_mtrx = np.cov(X)
    d, E = np.linalg.eigh(cov_mtrx)
    D = np.linalg.inv(np.sqrt(np.diag(d)))
    return np.dot(E, np.dot(D, np.dot(E.T, X)))

def ICA(X, max_iter=80, e=1e-60):
    X = preprocessing(X)
    m, n = X.shape
    W = np.random.rand(m, m)
    for i in range(m):
        w = W[i, :]
        for j in range(max_iter):
            # calculate next_w
            next_w = (X * g(np.dot(w.T, X))).mean(axis=1)\
                    - g_prime(np.dot(w.T, X)).mean() * w
            next_w /= np.linalg.norm(next_w)

            if i > 0:
                next_w -= np.dot(np.dot(next_w, W[:i].T), W[:i])

            # check distance between w & next_w
            dist = np.abs(np.abs((w * next_w).sum()) - 1)
            if dist < e:
                w = next_w
                print("Number of iterations is", j)
                break

            w = next_w
        W[i, :] = w
    print('The dimixing matrixe W =', W)

    return  W
    