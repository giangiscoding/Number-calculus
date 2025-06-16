import numpy as np

def find_Q_cholesky(A): # Trả về Q là ma trận tam giác dưới L, sao cho A = Q @ Q.T
    n = A.shape[0]
    # Kiểm tra ma trận có đối xứng không (điều kiện cần cho Cholesky)
    if not np.allclose(A, A.T):
        print("Cảnh báo: Ma trận A không đối xứng. Phân tích Cholesky có thể không hợp lệ.")
        # return None # Hoặc vẫn tiếp tục nếu người dùng muốn thử

    Q = np.zeros((n, n), dtype=float)

    for i in range(n): # Duyệt qua các hàng của Q
        for j in range(i + 1): # Duyệt qua các cột của Q (j <= i để Q là tam giác dưới)
            sum_val = 0.0
            # Tính tổng Q[i, k] * Q[j, k] cho k từ 0 đến j-1
            # Đây là tích vô hướng của hàng thứ i (đến cột j-1) và hàng thứ j (đến cột j-1) của Q
            # hoặc các phần tử Q[i,k]*Q[j,k]
            for k in range(j): # k < j
                sum_val += Q[i, k] * Q[j, k]
            
            if i == j: # Phần tử trên đường chéo chính Q[i,i]
                val_under_sqrt = A[i, i] - sum_val
                if val_under_sqrt <= 0: # Phải > 0 cho xác định dương, ==0 cho bán xác định dương
                    print(f"Giá trị dưới căn bậc hai không dương ({val_under_sqrt:.4f}) tại Q[{i},{i}]. "
                          "Ma trận có thể không xác định dương.")
                    return None
                Q[i, i] = np.sqrt(val_under_sqrt)
            else: # Phần tử dưới đường chéo chính Q[i,j] với i > j
                if Q[j, j] == 0:
                    # Điều này không nên xảy ra nếu ma trận xác định dương và không có lỗi làm tròn nghiêm trọng
                    print(f"Phần tử đường chéo Q[{j},{j}] bằng 0. Phân tích Cholesky thất bại.")
                    return None
                Q[i, j] = (A[i, j] - sum_val) / Q[j, j]
    return Q

# Giải LY = B (L là tam giác dưới) bằng Forward Substitution
def solve_forward_substitution(L_matrix, B_vector_or_matrix):
    n = L_matrix.shape[0]
    
    if B_vector_or_matrix.ndim == 1: # Nếu B là vector 1D
        B_mat = B_vector_or_matrix.reshape(-1, 1)
        is_B_vector = True
    else:
        B_mat = B_vector_or_matrix
        is_B_vector = False
        
    m_cols_B = B_mat.shape[1]
    Y = np.zeros((n, m_cols_B), dtype=float)

    for j_col_B in range(m_cols_B): # Với mỗi cột của B (vế phải)
        for i in range(n): # Duyệt qua các hàng để tính Y[i]
            sum_val = 0.0
            for k in range(i): # k < i
                sum_val += L_matrix[i, k] * Y[k, j_col_B]
            
            if L_matrix[i, i] == 0:
                print(f"L[{i},{i}] bằng 0 trong quá trình thế xuôi. Ma trận L suy biến.")
                return np.full_like(Y, np.nan) # Trả về NaN
            Y[i, j_col_B] = (B_mat[i, j_col_B] - sum_val) / L_matrix[i, i]
    
    if is_B_vector:
        return Y.flatten() # Trả về Y dạng vector nếu B là vector
    return Y

# Giải UX = Y (U là tam giác trên) bằng Backward Substitution
def solve_backward_substitution(U_matrix, Y_vector_or_matrix):
    n = U_matrix.shape[0]

    if Y_vector_or_matrix.ndim == 1: # Nếu Y là vector 1D
        Y_mat = Y_vector_or_matrix.reshape(-1, 1)
        is_Y_vector = True
    else:
        Y_mat = Y_vector_or_matrix
        is_Y_vector = False

    m_cols_Y = Y_mat.shape[1]
    X = np.zeros((n, m_cols_Y), dtype=float)

    for j_col_Y in range(m_cols_Y): # Với mỗi cột của Y
        for i in range(n - 1, -1, -1): # i từ n-1 xuống 0
            sum_val = 0.0
            for k in range(i + 1, n): # k > i
                sum_val += U_matrix[i, k] * X[k, j_col_Y]
            
            if U_matrix[i, i] == 0:
                print(f"U[{i},{i}] bằng 0 trong quá trình thế ngược. Ma trận U suy biến.")
                return np.full_like(X, np.nan) # Trả về NaN
            X[i, j_col_Y] = (Y_mat[i, j_col_Y] - sum_val) / U_matrix[i, i]
            
    if is_Y_vector:
        return X.flatten() # Trả về X dạng vector nếu Y là vector
    return X


# --- Dữ liệu đầu vào của bạn ---
A = np.array([[2., -2., -3.],
              [-2., 5., 4.],
              [-3., 4., 5.]])

B = np.array([[7., 2.],
              [-12., 3.],
              [-12., 5.]])

print("Ma trận A:\n", A)
print("Ma trận B:\n", B)

# Phân tích Cholesky: A = Q_lower @ Q_lower.T
print("\n--- Phân tích Cholesky (A = Q_lower @ Q_lower.T) ---")
Q_lower = find_Q_cholesky(A)

if Q_lower is not None:
    print("\nMa trận tam giác dưới Q_lower (hay L):\n", Q_lower)
    
    # Kiểm tra Q_lower @ Q_lower.T == A
    print("\nKiểm tra Q_lower @ Q_lower.T:\n", Q_lower @ Q_lower.T)
    if np.allclose(Q_lower @ Q_lower.T, A):
        print("Q_lower @ Q_lower.T bằng A (đã kiểm tra).")
    else:
        print("CẢNH BÁO: Q_lower @ Q_lower.T KHÔNG bằng A!")

    # Giải hệ AX = B, tức là (Q_lower @ Q_lower.T)X = B
    # Bước 1: Giải Q_lower * Y = B  => Y
    print("\n--- Bước 1: Giải Q_lower * Y = B ---")
    Y_solution = solve_forward_substitution(Q_lower, B)
    if Y_solution is not None and not np.isnan(Y_solution).any():
        print("Ma trận Y:\n", Y_solution)

        # Bước 2: Giải Q_lower.T * X = Y => X
        # Q_lower.T là ma trận tam giác trên
        Q_upper_transpose = Q_lower.T
        print("\n--- Bước 2: Giải Q_lower.T * X = Y ---")
        X_solution = solve_backward_substitution(Q_upper_transpose, Y_solution)
        
        if X_solution is not None and not np.isnan(X_solution).any():
            print("Ma trận nghiệm X:\n", X_solution)

            # Kiểm tra A @ X == B
            print("\nKiểm tra A @ X:\n", A @ X_solution)
            if np.allclose(A @ X_solution, B):
                print("A @ X bằng B (đã kiểm tra nghiệm).")
            else:
                print("CẢNH BÁO: A @ X KHÔNG bằng B!")
        else:
            print("Không thể giải Q_lower.T * X = Y.")
    else:
        print("Không thể giải Q_lower * Y = B.")
else:
    print("Không thể thực hiện phân tích Cholesky.")


print("\n\n--- Để so sánh, sử dụng numpy.linalg.cholesky ---")
try:
    L_numpy = np.linalg.cholesky(A) # Trả về ma trận tam giác dưới L
    print("L (từ np.linalg.cholesky):\n", L_numpy)
    print("Kiểm tra L @ L.T (numpy):\n", L_numpy @ L_numpy.T)

    # Giải với numpy
    # LY = B => Y = inv(L)B
    # L.T X = Y => X = inv(L.T)Y = inv(L.T)inv(L)B
    # Hoặc giải tuần tự
    Y_numpy = solve_forward_substitution(L_numpy, B) # Dùng lại hàm của chúng ta
    X_numpy = solve_backward_substitution(L_numpy.T, Y_numpy) # Dùng lại hàm của chúng ta
    print("\nX (sử dụng L từ numpy và hàm giải tự viết):\n", X_numpy)

    # Hoặc cách trực tiếp hơn để giải hệ đối xứng xác định dương
    # X_direct_solve = np.linalg.solve(A, B)
    # print("\nX (từ np.linalg.solve(A,B)):\n", X_direct_solve)

except np.linalg.LinAlgError as e:
    print(f"Lỗi khi dùng np.linalg.cholesky: {e}. Ma trận có thể không xác định dương.")