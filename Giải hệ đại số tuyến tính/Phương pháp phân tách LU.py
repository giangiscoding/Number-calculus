import numpy as np

def find_LU_crout(A):
    n = A.shape[0]
    L = np.zeros((n, n), dtype=float)
    U = np.zeros((n, n), dtype=float)

    np.fill_diagonal(U, 1.0) # U có đường chéo là 1 (Crout)

    for j in range(n): # Duyệt qua các cột để tính toán
        # Tính các phần tử của cột j trong L
        for i in range(j, n): # i >= j
            sum_val = 0.0
            for k in range(j): # k < j
                sum_val += L[i, k] * U[k, j]
            L[i, j] = A[i, j] - sum_val

        # Kiểm tra L[j,j] trước khi chia
        if L[j, j] == 0:
            print(f"L[{j},{j}] is zero. Matrix is singular or LU decomposition (Crout) fails without pivoting.")
            return None, None # Hoặc raise một exception

        # Tính các phần tử của hàng j trong U (trừ U[j,j] đã là 1)
        for i in range(j + 1, n): # i > j (chỉ các phần tử bên phải đường chéo)
            sum_val = 0.0
            for k in range(j): # k < j
                sum_val += L[j, k] * U[k, i]
            U[j, i] = (A[j, i] - sum_val) / L[j, j]
            
    return L, U

def find_Y(L, B): # Giải Ly = B (Forward substitution)
    n = L.shape[0]
    if B.ndim == 1: # Nếu B là vector 1D
        B_mat = B.reshape(-1, 1)
    else:
        B_mat = B
        
    m_cols_B = B_mat.shape[1]
    Y = np.zeros((n, m_cols_B), dtype=float)

    for j_col_B in range(m_cols_B): # Với mỗi cột của B
        for i in range(n):
            sum_val = 0.0
            for k in range(i): # k < i
                sum_val += L[i, k] * Y[k, j_col_B]
            if L[i, i] == 0:
                print(f"L[{i},{i}] is zero during forward substitution. Singular L matrix.")
                # Có thể trả về lỗi hoặc xử lý khác
                return np.full_like(Y, np.nan)
            Y[i, j_col_B] = (B_mat[i, j_col_B] - sum_val) / L[i, i]
    
    if B.ndim == 1: # Trả về Y dạng vector nếu B là vector
        return Y.flatten()
    return Y

def find_X(U, Y): # Giải Ux = Y (Backward substitution)
    n = U.shape[0]
    if Y.ndim == 1: # Nếu Y là vector 1D
        Y_mat = Y.reshape(-1, 1)
    else:
        Y_mat = Y

    m_cols_Y = Y_mat.shape[1]
    X = np.zeros((n, m_cols_Y), dtype=float)

    for j_col_Y in range(m_cols_Y): # Với mỗi cột của Y
        for i in range(n - 1, -1, -1): # i từ n-1 xuống 0
            sum_val = 0.0
            for k in range(i + 1, n): # k > i
                sum_val += U[i, k] * X[k, j_col_Y]
            if U[i, i] == 0:
                print(f"U[{i},{i}] is zero during backward substitution. Singular U matrix.")
                # Có thể trả về lỗi hoặc xử lý khác
                return np.full_like(X, np.nan)
            X[i, j_col_Y] = (Y_mat[i, j_col_Y] - sum_val) / U[i, i]
            
    if Y.ndim == 1: # Trả về X dạng vector nếu Y là vector
        return X.flatten()
    return X

# --- Dữ liệu đầu vào của bạn ---
A = np.array([[2., 2., -3.],
              [-4., -3., 4.],
              [2., 1., 2.]])

B = np.array([[9., 3.],
              [-15., 2.],
              [3., 1.]])

print("Ma trận A:\n", A)
print("Ma trận B:\n", B)

# Sử dụng phiên bản Crout
print("\n--- Phân tích LU (Crout) ---")
L_crout, U_crout = find_LU_crout(A)

if L_crout is not None and U_crout is not None:
    print("\nMa trận L (Crout):\n", L_crout)
    print("Ma trận U (Crout) (đường chéo là 1):\n", U_crout)

    # Kiểm tra L @ U == A
    print("\nKiểm tra L @ U:\n", L_crout @ U_crout)
    if np.allclose(L_crout @ U_crout, A):
        print("L @ U bằng A (đã kiểm tra).")
    else:
        print("CẢNH BÁO: L @ U KHÔNG bằng A!")

    print("\n--- Giải LY = B ---")
    Y_solution = find_Y(L_crout, B)
    if Y_solution is not None:
        print("Ma trận Y:\n", Y_solution)

        print("\n--- Giải UX = Y ---")
        X_solution = find_X(U_crout, Y_solution)
        if X_solution is not None:
            print("Ma trận nghiệm X:\n", X_solution)

            # Kiểm tra A @ X == B
            print("\nKiểm tra A @ X:\n", A @ X_solution)
            if np.allclose(A @ X_solution, B):
                print("A @ X bằng B (đã kiểm tra nghiệm).")
            else:
                print("CẢNH BÁO: A @ X KHÔNG bằng B!")
        else:
            print("Không thể giải UX = Y.")
    else:
        print("Không thể giải LY = B.")
else:
    print("Không thể thực hiện phân tích LU.")

print("\n\n--- Để so sánh, sử dụng scipy.linalg.lu ---")
from scipy.linalg import lu, solve_triangular

# Phân tích P@A = L@U (scipy trả về P, L, U)
P_scipy, L_scipy, U_scipy = lu(A)
print("P (scipy):\n", P_scipy) # Ma trận hoán vị
print("L (scipy) (đường chéo là 1 - Doolittle convention):\n", L_scipy)
print("U (scipy):\n", U_scipy)
print("P@A:\n", P_scipy @ A)
print("L@U (scipy):\n", L_scipy @ U_scipy)

# Giải với scipy
# LY = PB
# UX = Y
if B.ndim == 1:
    Y_scipy = solve_triangular(L_scipy, P_scipy @ B, lower=True, unit_diagonal=True)
    X_scipy = solve_triangular(U_scipy, Y_scipy, lower=False, unit_diagonal=False)
else: # B là ma trận
    Y_scipy_cols = []
    X_scipy_cols = []
    for col_idx in range(B.shape[1]):
        b_col = B[:, col_idx]
        y_col = solve_triangular(L_scipy, P_scipy @ b_col, lower=True, unit_diagonal=True)
        x_col = solve_triangular(U_scipy, y_col, lower=False, unit_diagonal=False)
        Y_scipy_cols.append(y_col)
        X_scipy_cols.append(x_col)
    Y_scipy = np.column_stack(Y_scipy_cols)
    X_scipy = np.column_stack(X_scipy_cols)


print("\nY (scipy):\n", Y_scipy)
print("X (scipy):\n", X_scipy)