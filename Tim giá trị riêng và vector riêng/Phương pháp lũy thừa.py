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
    x = np.ones((n, 1))
    check = 0
    lambdas = []
    v = np.zeros((n, 0))
    k = 1
    
    # Trường hợp có 1 giá trị riêng trội
    y1 = x.copy()
    while check == 0 and k < K:
        y1 = A @ y1
        y2 = A @ y1
        k += 1
        check = kiemtrasongsong(y1, y2, tol)

    if check == 1:
        mask = (y2 != 0) & (y1 != 0)
        ratio = np.divide(y2[mask], y1[mask])
        mean_value = np.mean(ratio) if ratio.size > 0 else 0
        lambdas.append(mean_value)
        v = y1 / norm(y1, 2)
        print("1 gia tri rieng")
        return lambdas, v

    # Trường hợp có 2 giá trị riêng đối nhau
    check = 0
    k = 1
    y1 = x.copy()
    while check == 0 and k <= K:
        y1 = A @ y1
        y2 = A @ y1
        y3 = A @ y2
        k += 1
        check = kiemtrasongsong(y1, y3, tol)  # Kiểm tra y1 và y3

    if check == 1:
        mask = (y3 != 0) & (y1 != 0)
        ratio = np.divide(y3[mask], y1[mask])
        mean_value = np.mean(ratio) if ratio.size > 0 else 0
        lambdas.append(np.sqrt(mean_value))
        lambdas.append(-lambdas[0])
        v1 = y2 + lambdas[0] * y1
        v1 = v1 / norm(v1, 2)
        v2 = y2 - lambdas[0] * y1
        v2 = v2 / norm(v2, 2)
        print("2 gia tri rieng doi nhau")
        return lambdas, np.column_stack((v1, v2))

    # Trường hợp 2 nghiệm phức liên hợp
    if check == 0:
        m = K  # Sử dụng K như m trong thuật toán gốc
        y1 = matrix_power(A, 2*m + 2) @ x
        y2 = matrix_power(A, 2*m) @ x
        y3 = matrix_power(A, 2*m + 1) @ x
        
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

        v1 = lambda_values[0] * y1
        v1 = v1 / norm(v1, 2)
        v2 = lambda_values[1] * y1
        v2 = v2 / norm(v2, 2)
        v = np.column_stack((v1, v2))
        print("2 gia tri rieng phuc lien hop")
        return lambda_values, v

    return "Không tìm thấy giá trị riêng trong giới hạn lặp"

def kiemtrasongsong(y1, y2, tol):
    ratios = []
    for a, b in zip(y1, y2):
        if abs(b) > tol:  # Tránh chia cho 0
            ratios.append(a / b)
        elif abs(a) > tol:  # Nếu b ≈ 0 nhưng a ≠ 0 → Không song song
            return False
    return all(np.isclose(ratio, ratios[0], atol=tol) for ratio in ratios)

A = np.array([[-2,  1,  1,  1],
    [-7, -5, -2, -1],
    [ 0, -1, -3, -2],
    [-1,  0, -1,  0]])
print(luythua(A, 10, 1e-20))