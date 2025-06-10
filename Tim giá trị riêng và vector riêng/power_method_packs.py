import numpy as np
from numpy.linalg import norm, matrix_power
from sympy import symbols, det, Matrix, solve

def power_method(A, K, tol):
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
    y1 = x
    while check == 0 and k < K:
        y1 = A @ y1
        y2 = A @ y1
        k += 1
        check = kiemtrasongsong(y1, y2, tol)

    if check == 1:
        mask = (y2 != 0) & (y1 != 0)
        ratio = np.divide(y2[mask], y1[mask])
        mean_value = ratio[1]
        lambdas.append(mean_value)
        v = y1 / norm(y1, 2)
        return lambdas, v

    # Trường hợp có 2 giá trị riêng đối nhau
    if check == 0:
        y1 = x
        k = 1
        while check == 0 and k <= K:
            y1 = matrix_power(A,2*k) @ y1
            y2 = A @ y1
            y3 = A @ y2
            k += 1
            check = kiemtrasongsong(y1, y2, tol)  # Kiểm tra y1 và y3

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
        return lambdas, np.column_stack((v1, v2))

    # Trường hợp 2 nghiệm phức liên hợp
    if check == 0:
        y1 = matrix_power(A,2*k)@x
        y2 = A @ y1
        y3 = A @ y2
        
        # Tìm các chỉ số khác 0
        nonzero_indices = np.nonzero(y1)[0]
        if len(nonzero_indices) < 2:
            return "Không tìm thấy đủ chỉ số khác 0"
        r, s = nonzero_indices[0], nonzero_indices[1]

        z = symbols('z')
        mat = Matrix([
            [1, y1[r, 0], y1[s, 0]],
            [z, y2[r, 0], y2[s, 0]],
            [z**2, y3[r, 0], y3[s, 0]]
        ])
        p = det(mat)
        lambda_solutions = solve(p, z)
        lambda_values = np.array([complex(sol.evalf()) for sol in lambda_solutions], dtype=np.complex128)

        v1 = y2 - lambda_values[0] * y1
        v1 = v1 / norm(v1, 2)
        v2 = y2 - lambda_values[1] * y1
        v2 = v2 / norm(v2, 2)
        v = np.column_stack((v1, v2))
        print("gg")
        return lambda_values, v
    return "Không tìm thấy giá trị riêng trong giới hạn lặp"

# Ham kiem tra song song
def kiemtrasongsong(u, v, tol):
    u_norm = u / norm(u,2)
    v_norm = v / norm(v,2)
    
    check = (norm(u_norm - v_norm,2) <= tol) or (norm(u_norm + v_norm,2) <= tol)
    
    return check