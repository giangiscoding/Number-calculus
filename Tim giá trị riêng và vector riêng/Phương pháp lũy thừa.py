import numpy as np
from numpy.linalg import norm, matrix_power
from sympy import symbols, det, Matrix, solve

def luythua(A, K, tol):
    """
    Phương pháp này dùng để tìm giá trị riêng trội
    A là ma trận cần tìm giá trị riêng trội
    K là số lần lặp tối đa
    tol là sai số cho phép
    """
    n = A.shape[1]
    x = np.random.rand(n, 1)
    check = 0
    lambdas = []
    v = np.zeros((n, 0))
    k = 1
    
    # Trường hợp có 1 giá trị riêng trội
    while check == 0 and k < K:
        y1 = A @ x
        y2 = A @ y1
        k += 1
        check = kiemtrasongsong(y1, y2, tol)

    if check == 1:
        mask = (y2 != 0) & (y1 != 0)
        ratio = np.divide(y2[mask], y1[mask])
        mean_value = ratio[1]
        lambdas.append(mean_value)
        v = y1 / norm(y1, 2)
        print("1 gia tri rieng")
        return lambdas, v

    # Trường hợp có 2 giá trị riêng đối nhau
    check = 0
    k = 1
    while check == 0 and k <= K:
        y1 = matrix_power(A, 2*k) @ x
        y2 = A @ y1
        y3 = A @ y2
        k += 1
        check = kiemtrasongsong(y1, y3, tol)  # Kiểm tra y1 và y3

    if check == 1:
        mask = (y3 != 0) & (y1 != 0)
        ratio = np.divide(y3[mask], y1[mask])
        mean_value = ratio[0]
        lambdas.append(np.sqrt(mean_value))
        lambdas.append(-lambdas[0])
        v1 = lambdas[0] * y1 + y2
        v1 = v1 / norm(v1, 2)
        v2 = lambdas[0] * y1 - y2
        v2 = v2 / norm(v2, 2)
        print("2 gia tri rieng doi nhau")
        return lambdas, np.column_stack((v1, v2))

    # Trường hợp 2 nghiệm phức liên hợp
    if check == 0:
        m = K  # Sử dụng K như m trong thuật toán gốc
        y1 = A @ x
        y2 = A @ y1
        y3 = A @ y2
        
        # Tìm các chỉ số khác 0
        nonzero_indices = np.nonzero(y1)[0]
        if len(nonzero_indices) < 2:
            return "Không tìm thấy đủ chỉ số khác 0"
        r, s = nonzero_indices[0], nonzero_indices[1]

        z = symbols('z')
        mat = Matrix([
            [1, y2[r, 0], y2[s, 0]],
            [z, y3[r, 0], y3[s, 0]],
            [z**2, y1[r, 0], y1[s, 0]]
        ])
        p = det(mat)
        lambda_solutions = solve(p, z)
        lambda_values = np.array([complex(sol.evalf()) for sol in lambda_solutions], dtype=np.complex128)

        v1 = y3 - lambda_values[0] * y2
        v1 = v1 / norm(v1, 2)
        v2 = y3 - lambda_values[1] * y2
        v2 = v2 / norm(v2, 2)
        v = np.column_stack((v1, v2))
        print("2 gia tri rieng phuc lien hop")
        return lambda_values, v
    
    return "Không tìm thấy giá trị riêng trong giới hạn lặp"

import numpy as np

def kiemtrasongsong(u, v, tol):
    u_norm = u / norm(u,2)
    v_norm = v / norm(v,2)
    
    check = (norm(u_norm - v_norm,2) <= tol) or (norm(u_norm + v_norm,2) <= tol)
    
    return check

np.set_printoptions(precision=15, suppress=True)

# The matrix from your image (converted commas to decimal points)
A = np.array([
    [ 0.308028,    4.29374056, -1.38633287,  0.15108192],
    [ 4.29374056,  0.31992,     1.21184431,  1.69639799],
    [-1.38633287,  1.21184431, -0.25076267, -0.47251727],
    [ 0.15108192,  1.69639799, -0.47251727,  0.42281467]
])

print(luythua(A, 1150, 1e-10))