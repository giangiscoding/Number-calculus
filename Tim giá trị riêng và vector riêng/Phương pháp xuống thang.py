import numpy as np
from power_method_packs import power_method

def xuongthang1(A,v):
    n = A.shape[1]
    theta = np.identity(n)
    max_value = max(v)
    max_index = v.index(max_value)
    v = v/max_value
    theta = theta[max_index] - v
    B = theta @ A
    return B

def xuongthang2(A,K,tol):
    n = A.shape[0]

    for i in range(n):
        lambda_1, v_1 = power_method(A, K, tol)
        _, w_1 = power_method(A.T, K, tol)  # A.T là ma trận chuyển vị
        # Cập nhật ma trận A
        denominator = np.dot(w_1.T, v_1)
        A = A - (lambda_1 * np.outer(v_1, w_1.T)) / denominator
        print(f'Giá trị riêng thứ {i+1}', lambda_1)
        print(f'Vector riêng thứ {i+1} là')
        print('\n', v_1)
        print()

A = np.array([[4,5,2],
              [2,5,1],
              [2,1,6]])
xuongthang2(A,50,1e-5)