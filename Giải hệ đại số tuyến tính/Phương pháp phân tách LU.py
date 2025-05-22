import numpy as np
def find_LU(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    L[:,0] = A[:,0]
    U[0] = A[0]/L[0,0]
    L_temp = np.zeros((1,n))
    U_temp = np.zeros((n))
    np.fill_diagonal(U, 1)
    for i in range(1,n):
        L[:,i] = A[:,i] - sum(U[k,i] * L[:,k] for k in range(i))

        if L[i,i] == 0:
            print("Matrix is singular, cannot perform LU decomposition.")
            return None

        U[i] = (A[i] - sum(L[i,k] * U[k] for k in range(i))) / L[i,i]

    return L, U

def find_Y(L,B):
    n = L.shape[0]
    m = B.shape[1]
    Y = np.zeros((n, m))
    for k in range(n):
        Y[k] = (B[k] - sum(L[k,i]*Y[i] for i in range(k))) / L[k,k]
    return Y

def find_X(U,Y):
    n = U.shape[0]
    m = Y.shape[1]
    X = np.zeros((n, m))
    for k in range(n-1, -1, -1):
        X[k] = (Y[k] - sum(U[k,i]*X[i] for i in range(k+1, n) ) ) / U[k,k]
    return X

A = np.array([[2, 2, -3], 
              [-4, -3, 4], 
              [2, 1, 2]])
B = np.array([[9, 3],
              [-15, 2],
              [3, 1]])
L, U = find_LU(A)
Y = find_Y(L,B)
X = find_X(U,Y)
print(L)
print(U)
print(Y)
print(X)