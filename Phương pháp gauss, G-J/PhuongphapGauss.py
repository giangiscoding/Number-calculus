import numpy as np

def gauss(A, b):
    augmented_A = np.hstack((A, b.reshape(-1, 2)))
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
    [3, -2, 5, -7, 4],
    [2, 9, 14, -30, 0],
    [5, -4, 18, -26, 14],
    [-4, 2, 3, -5, 2],
    [1, 3, 2, -6, -2]
    ])
b = np.array([[3, 5],
              [-5, 10],
              [7, 11],
              [-2, -4],
              [-2, 3]])
print(gauss(A, b))