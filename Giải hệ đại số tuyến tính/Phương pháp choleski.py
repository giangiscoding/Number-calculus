import numpy as np

def find_Q(A):
    n = A.shape[0]
    Q = np.zeros((n, n))
    for k in range(n):
        Q[k,k] = np.sqrt(A[k,k] - sum((Q[i,k])**2 for i in range(k)))

        Q[k] = (A[k] - sum(Q[i,k] * Q[i] for i in range(k))) / Q[k,k]
    return Q

def find_Y(Qt,B):
    n = Qt.shape[0]
    m = B.shape[1]
    Y = np.zeros((n, m))
    for i in range(n):
        Y[i] = (B[i] - sum(Qt[i,j]*Y[j] for j in range(i))) / Qt[i,i]
    return Y

def find_X(Q,Y):
    n = Q.shape[0]
    m = Y.shape[1]
    X = np.zeros((n, m))
    for i in range(n-1, -1, -1):
        X[i] = (Y[i] - sum(Q[i,j]*X[j] for j in range(i+1, n) ) ) / Q[i,i]
    return X


A = np.array([[2, -2, -3], 
              [-2, 5, 4], 
              [-3, 4, 5]])
B = np.array([[7,2],
              [-12,3],
              [-12,5]])

Q = find_Q(A)
Qt = Q.T
Y = find_Y(Qt,B)
X = find_X(Q, Y)

print(Q)
print(Y)
print(X)