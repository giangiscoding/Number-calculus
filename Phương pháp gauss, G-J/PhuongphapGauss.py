import numpy as np

def gauss(A, b):
    augmented_A = np.hstack((A, b.reshape(-1, 1)))
    m = A.shape[0]
    n = m

    for i in range(n):
        if augmented_A[i][i] == 0:
            continue
        for j in range(i+1,m):
            factor = augmented_A[j, i] / augmented_A[i, i]
            augmented_A[j] = augmented_A[j] - factor * augmented_A[i]
        print(augmented_A)
    return augmented_A

A = np.array([
    [2, -1, 3, 1],
    [1, 2, 1, -2],
    [-1, 8, -3, -8],
    [3, 1, 4, -1]
    ])
b = np.array([-2, 1, 7, -1], dtype=float)
print(gauss(A, b))