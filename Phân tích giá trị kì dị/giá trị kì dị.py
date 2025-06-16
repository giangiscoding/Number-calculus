import numpy as np
from numpy.linalg import eig, norm, svd, det, solve
import sympy # Thư viện tính toán biểu tượng, dùng cho solve trong powerMethod

# Định dạng in ấn cho numpy arrays (tương tự format short của MATLAB)
np.set_printoptions(precision=4, suppress=True)

def eig_vector_zero(M, tol):
    """
    Nhận vào ma trận M và đưa ra ma trận trực chuẩn S chứa các vector riêng
    ứng với giá trị riêng lambda = 0 của ma trận M.
    Tương đương với việc tìm một cơ sở trực chuẩn cho không gian null (không gian hạt nhân) của M.
    """
    # Sử dụng SVD để tìm không gian null một cách ổn định hơn Gauss-Jordan tự viết
    # M = U_svd * Sigma_svd * Vh_svd
    # Các vector riêng ứng với giá trị riêng 0 của M (hoặc M.T @ M)
    # chính là các vector trong không gian null của M.
    # Trong SVD, các vector kỳ dị phải (cột của Vh.T hay V) ứng với các giá trị kỳ dị bằng 0
    # sẽ tạo thành một cơ sở cho không gian null của M.
    U_svd, s_svd, Vh_svd = svd(M)
    null_space_vectors = Vh_svd[s_svd < tol, :].T # Các hàng của Vh ứng với s_svd gần 0, chuyển vị thành cột

    if null_space_vectors.shape[1] == 0:
        return np.zeros((M.shape[0], 0)) # Trả về ma trận rỗng nếu không có vector nào

    # Trực chuẩn hóa Gram-Schmidt (nếu cần, SVD thường trả về các vector trực giao)
    # S = gramSmidth(null_space_vectors) # SVD đã cho các vector trực giao rồi nên bước này có thể không cần thiết
                                       # hoặc nếu muốn đảm bảo chuẩn hóa
    # Các cột của Vh.T (tức là V) đã trực chuẩn
    return null_space_vectors


def gj_elimination(A_matrix, tol):
    """
    Thực hiện phép khử Gauss-Jordan trên ma trận A.
    Trả về ma trận sau khi khử và danh sách các cột trục (pivot columns).
    """
    A = np.array(A_matrix, dtype=float) # Làm việc trên bản sao
    m_rows, n_cols = A.shape
    pivot_row = 0
    pivot_col_indices = []

    for j_col in range(n_cols): # Duyệt qua các cột
        if pivot_row >= m_rows:
            break

        # Tìm hàng có phần tử lớn nhất (theo trị tuyệt đối) trong cột hiện tại (từ pivot_row trở xuống)
        i_max_in_col = pivot_row + np.argmax(np.abs(A[pivot_row:, j_col]))

        if np.abs(A[i_max_in_col, j_col]) < tol:
            # Phần tử trục quá nhỏ, bỏ qua cột này
            continue

        # Hoán đổi hàng pivot hiện tại với hàng có phần tử trục lớn nhất
        A[[pivot_row, i_max_in_col]] = A[[i_max_in_col, pivot_row]]

        # Chuẩn hóa hàng pivot (chia cho phần tử trục để nó bằng 1)
        A[pivot_row, :] = A[pivot_row, :] / A[pivot_row, j_col]

        # Khử các phần tử khác trong cột trục về 0
        for i_row in range(m_rows):
            if i_row != pivot_row:
                A[i_row, :] = A[i_row, :] - A[i_row, j_col] * A[pivot_row, :]
        
        pivot_col_indices.append(j_col) # Lưu chỉ số cột trục (0-based)
        pivot_row += 1
        
    return A, pivot_col_indices


def power_method(A_matrix, tol, max_iter):
    """
    Phương pháp lũy thừa để tìm giá trị riêng trội nhất và vector riêng tương ứng.
    Hàm này cố gắng xử lý trường hợp giá trị riêng thực trội, cặp giá trị riêng đối nhau,
    và cặp giá trị riêng phức liên hợp.

    LƯU Ý: Việc triển khai cho trường hợp phức và đối nhau trong code MATLAB gốc
    có thể không hoàn toàn ổn định hoặc chính xác trong mọi trường hợp.
    Đặc biệt, việc sử dụng symbolic math (syms, det, solve) rất tốn kém
    và không phải là cách tiếp cận số học tiêu chuẩn cho vấn đề này.
    Thư viện như NumPy/SciPy có các hàm tìm giá trị riêng hiệu quả và ổn định hơn nhiều.
    Phiên bản Python này sẽ cố gắng mô phỏng, nhưng với những hạn chế đó.
    """
    A = np.array(A_matrix, dtype=float)
    n = A.shape[0]
    x_initial = np.ones(n) # Vector khởi tạo

    eigenvalues_found = []
    eigenvectors_found = np.zeros((n, 0))

    # --- Trường hợp 1: Giá trị riêng thực trội ---
    y1 = x_initial.copy()
    y1_normalized_prev = np.zeros(n)
    is_converged = False
    for m_iter in range(max_iter):
        y1_normalized = y1 / norm(y1)
        if norm(y1_normalized - y1_normalized_prev) < tol or \
           norm(y1_normalized + y1_normalized_prev) < tol: # Kiểm tra song song hoặc đối song song
            is_converged = True
            break
        y1_normalized_prev = y1_normalized.copy()
        
        y_next = A @ y1_normalized # Phép nhân Ay_k
        # Ước lượng giá trị riêng lambda = (y_k+1 . y_k) / (y_k . y_k)
        # Hoặc dùng thành phần Rayleigh: (y_k.T @ A @ y_k) / (y_k.T @ y_k)
        # Code MATLAB dùng y2(index)./y1(index) với y2 = A*y1, y1 = A*y0 (tức là A^2 * y0 và A*y0)
        # Điều này hơi khác với power method chuẩn.
        # Ta sẽ dùng ước lượng lambda từ y_next và y1_normalized
        if norm(y1_normalized) > tol: # Tránh chia cho 0
            # lambda_est = np.dot(y_next, y1_normalized) / np.dot(y1_normalized, y1_normalized)
            # Hoặc lấy trung bình các tỷ lệ
            non_zero_indices = np.abs(y1_normalized) > tol
            if np.any(non_zero_indices):
                 lambda_est_components = y_next[non_zero_indices] / y1_normalized[non_zero_indices]
                 lambda_est = np.mean(lambda_est_components)
            else:
                lambda_est = 0 # Hoặc xử lý khác
        else:
            lambda_est = 0 # Hoặc xử lý khác

        y1 = y_next # Chuẩn bị cho vòng lặp tiếp

    if is_converged:
        eigenvalues_found.append(lambda_est)
        eigenvectors_found = np.hstack((eigenvectors_found, (y1 / norm(y1)).reshape(-1, 1)))
        return np.array(eigenvalues_found), eigenvectors_found

    # --- Trường hợp 2: Cặp giá trị riêng thực đối nhau |lambda1| = |-lambda1| ---
    # Logic này trong MATLAB dựa trên việc A^2 có giá trị riêng lambda^2 là trội.
    # y1 = x, y2 = Ay1, y3 = Ay2. Nếu y1, y3 song song: lambda^2 = y3/y1
    y1_case2 = x_initial.copy()
    y1_normalized_prev_case2 = np.zeros(n)
    is_converged_case2 = False
    lambda_sq_est = 0

    for m_iter in range(max_iter):
        y1_normalized_case2 = y1_case2 / norm(y1_case2)
        if norm(y1_normalized_case2 - y1_normalized_prev_case2) < tol or \
           norm(y1_normalized_case2 + y1_normalized_prev_case2) < tol:
            is_converged_case2 = True # Cho A^2
            break
        y1_normalized_prev_case2 = y1_normalized_case2.copy()
        
        # Áp dụng A^2
        y_next_sq = A @ (A @ y1_normalized_case2) # A*A*y_k
        non_zero_indices_case2 = np.abs(y1_normalized_case2) > tol
        if np.any(non_zero_indices_case2):
            lambda_sq_est_components = y_next_sq[non_zero_indices_case2] / y1_normalized_case2[non_zero_indices_case2]
            lambda_sq_est = np.mean(lambda_sq_est_components)
        else:
            lambda_sq_est = 0
        y1_case2 = y_next_sq


    if is_converged_case2 and lambda_sq_est > 0: # Cần lambda_sq_est > 0 để căn bậc hai có nghĩa
        lambda1_val = np.sqrt(lambda_sq_est)
        eigenvalues_found.extend([lambda1_val, -lambda1_val])
        
        # Tính vector riêng: v1 = Ay + lambda1*y, v2 = Ay - lambda1*y
        # y ở đây nên là vector hội tụ của A (không phải A^2)
        # Để đơn giản, ta chỉ trả về các giá trị riêng nếu tìm được
        # Việc tìm vector riêng cho trường hợp này phức tạp hơn power method chuẩn.
        # Code MATLAB gốc có: v1 = y2+lambda1*y1; v2 = y2-lambda1*y1; với y1, y2 là từ phép lặp với A.
        # Ta sẽ bỏ qua phần tìm vector riêng phức tạp này và chỉ trả về giá trị riêng.
        # Hoặc, sử dụng phương pháp QR hoặc hàm eig của numpy để có kết quả đầy đủ và chính xác.
        print("Power method: Phát hiện trường hợp giá trị riêng đối nhau (ước lượng).")
        # Trả về các giá trị riêng, vector riêng tạm thời là rỗng
        return np.array(eigenvalues_found), np.zeros((n,0))


    # --- Trường hợp 3: Cặp giá trị riêng phức liên hợp ---
    # Logic này trong MATLAB sử dụng định thức và giải phương trình bậc hai tượng trưng.
    # Đây là cách rất không hiệu quả và không ổn định về mặt số học.
    # y1 = A^(2m+2)*x, y2 = A^(2m)*x, y3 = A^(2m+1)*x
    # det([1, y2(r), y2(s); z, y3(r), y3(s); z^2, y1(r), y1(s)]) = 0
    # Trong Python, việc này sẽ dùng thư viện symbolic math (sympy).
    # Tuy nhiên, Power Method chuẩn không được thiết kế để tìm giá trị riêng phức một cách trực tiếp như vậy.
    # Các phương pháp như QR iteration phù hợp hơn.
    # Vì tính phức tạp và không ổn định, ta sẽ bỏ qua việc triển khai chính xác phần này
    # và chỉ đưa ra cảnh báo.
    print("Power method: Không hội tụ về giá trị riêng thực trội hoặc cặp đối nhau. "
          "Trường hợp giá trị riêng phức hoặc các trường hợp khác phức tạp hơn "
          "không được triển khai đầy đủ/ổn định trong phiên bản này. "
          "Nên sử dụng np.linalg.eig.")

    return np.array(eigenvalues_found), eigenvectors_found # Trả về những gì đã tìm được


def deflation_wielandt(A_matrix, lambda_val, v_eigenvector, w_eigenvector_adj=None):
    """
    Thực hiện phép xuống thang Wielandt để loại bỏ giá trị riêng đã biết.
    A_new = A - lambda * v * w.T / (w.T @ v)
    Nếu A đối xứng, w = v.
    """
    A = np.array(A_matrix, dtype=float)
    v = v_eigenvector.reshape(-1, 1) # Đảm bảo là vector cột

    if w_eigenvector_adj is None: # Trường hợp A đối xứng, w = v
        w = v
    else:
        w = w_eigenvector_adj.reshape(-1, 1)

    # Đảm bảo w.T @ v khác 0
    wTv = w.T @ v
    if np.abs(wTv[0,0]) < 1e-9: # Ngưỡng nhỏ
        print("Cảnh báo: w.T @ v gần bằng 0 trong phép xuống thang. Kết quả có thể không chính xác.")
        return A # Trả về ma trận gốc nếu không thể xuống thang

    A_deflated = A - (lambda_val * (v @ w.T)) / wTv[0,0]
    return A_deflated


def gram_schmidt_orthogonalization(S_matrix):
    """
    Thực hiện trực chuẩn hóa Gram-Schmidt cho các cột của ma trận S.
    """
    S = np.array(S_matrix, dtype=float) # Làm việc trên bản sao
    m_rows, n_cols = S.shape
    Q_orthonormal = np.zeros_like(S)

    for j_col in range(n_cols):
        v_j = S[:, j_col].copy() # Lấy cột j
        for i_col in range(j_col): # Trừ đi các hình chiếu lên các vector đã trực chuẩn hóa trước đó
            q_i = Q_orthonormal[:, i_col]
            projection_coeff = np.dot(q_i, v_j) # q_i.T @ v_j
            v_j = v_j - projection_coeff * q_i
        
        norm_v_j = norm(v_j)
        if norm_v_j < 1e-9: # Ngưỡng nhỏ, nếu vector trở thành 0 (phụ thuộc tuyến tính)
            # print(f"Cảnh báo: Vector cột {j_col} trở thành zero sau khi trực giao hóa. Các cột có thể phụ thuộc tuyến tính.")
            Q_orthonormal[:, j_col] = np.zeros(m_rows) # Hoặc bỏ qua, hoặc xử lý khác
        else:
            Q_orthonormal[:, j_col] = v_j / norm_v_j
            
    return Q_orthonormal

# --- Phần chính của script ---
tol = 1e-3  # Tham số điều chỉnh
max_iteration = 100 # Số lần lặp tối đa

# Chú ý: Ma trận A trong MATLAB được nhập theo cột rồi chuyển vị.
# A_matlab_style = np.array([
#     [6, 10, 3, -4],
#     [7, 9, 4, -2],
#     [5, -2, 3, 7],
#     [4, -10, -5, 14],
#     [-8, 7, 4, -15],
#     [9, -10, -1, 19]
# ])
# A = A_matlab_style.T # Chuyển vị để giống A trong MATLAB

A = np.array([ # Nhập trực tiếp ma trận A (đã chuyển vị như trong MATLAB)
    [3, 3, 7],
    [3, 9, 10],
    [8, 2, 10],
    [2, 10, 9]
])

print("Ma trận A:\n", A)

M = A.T @ A  # Ma trận Gram M = A^T * A
M_1 = A @ A.T # Ma trận M' = A * A^T (M_prime)

print("\nM = A.T @ A:\n", M)
print("M_1 = A @ A.T:\n", M_1)

# Tìm vector riêng ứng với giá trị riêng 0
# V_2_null_space là các cột tạo thành cơ sở cho không gian null của M
# U_2_null_space là các cột tạo thành cơ sở cho không gian null của M_1
V_2_null_space = eig_vector_zero(M, tol)
U_2_null_space = eig_vector_zero(M_1, tol)

print("\nCác vector riêng phải ứng với giá trị kỳ dị 0 (cơ sở không gian null của A.T @ A), V_2:\n", V_2_null_space)
print("Các vector riêng trái ứng với giá trị kỳ dị 0 (cơ sở không gian null của A @ A.T), U_2:\n", U_2_null_space)


# Xác định hạng của A (rank)
# Sử dụng SVD để tính hạng một cách ổn định
# Hoặc có thể dùng Gauss-Jordan (GJ) như trong MATLAB nhưng SVD tốt hơn
# _, rank_A_cols_M = gj_elimination(M.copy(), tol) # GJ trên M
# rank_A = len(rank_A_cols_M)
# Hoặc từ SVD của A:
U_svd_A, s_svd_A, Vh_svd_A = svd(A)
rank_A = np.sum(s_svd_A > tol) # Số giá trị kỳ dị lớn hơn tol

print(f"\nHạng của A (tính từ SVD của A): {rank_A}")

# Khởi tạo các danh sách/mảng để lưu kết quả SVD
lambdas_sq = [] # Lưu các giá trị lambda^2 (giá trị riêng của M hoặc M_1)
V_singular_vectors = np.zeros((M.shape[0], 0))  # Các vector kỳ dị phải
U_singular_vectors = np.zeros((M_1.shape[0], 0)) # Các vector kỳ dị trái

M_current = M.copy() # Ma trận để xuống thang

print("\n--- Bắt đầu quá trình tìm giá trị kỳ dị và vector kỳ dị bằng Power Method và xuống thang ---")
# Vì M và M_1 là đối xứng, w_1 (vector riêng trái của M) sẽ bằng v_1 (vector riêng phải của M)
# Phép xuống thang Wielandt cho ma trận đối xứng: M_new = M - lambda * v * v.T / (v.T @ v)
# v.T @ v = norm(v)^2. Nếu v đã chuẩn hóa, v.T @ v = 1.

for i in range(rank_A): # Chỉ cần tìm rank_A giá trị kỳ dị khác 0
    # Tìm giá trị riêng trội nhất (lambda_sq_i) và vector riêng (v_i) của M_current
    # Hàm power_method của MATLAB phức tạp hơn và cố gắng xử lý nhiều trường hợp.
    # Để đơn giản và ổn định hơn, ta có thể dùng np.linalg.eig cho M_current ở mỗi bước,
    # hoặc một phiên bản Power Method đơn giản hơn nếu bắt buộc.
    # Ở đây, ta sẽ dùng một cách tiếp cận đơn giản hơn cho power method.
    
    # Sử dụng np.linalg.eig để lấy giá trị riêng và vector riêng một cách chính xác hơn
    # cho mục đích minh họa xuống thang, thay vì power_method tự viết phức tạp.
    eigenvalues_M_curr, eigenvectors_M_curr = eig(M_current)
    
    # Tìm giá trị riêng trội nhất (lớn nhất theo trị tuyệt đối)
    idx_dominant = np.argmax(np.abs(eigenvalues_M_curr))
    lambda_sq_i = eigenvalues_M_curr[idx_dominant]
    if lambda_sq_i < tol : # Nếu giá trị riêng lớn nhất đã quá nhỏ
        print(f"Giá trị riêng trội nhất {lambda_sq_i:.4e} quá nhỏ, dừng sớm tại bước {i+1}.")
        break
    v_i = eigenvectors_M_curr[:, idx_dominant].real # Lấy phần thực nếu có phần ảo nhỏ
    v_i = v_i / norm(v_i) # Chuẩn hóa vector riêng

    # lambda_sq_i, v_i_matrix = power_method(M_current, tol * 1e-2, max_iteration * 2) # Tăng độ chính xác cho power method
    # if v_i_matrix.shape[1] == 0:
    #     print(f"Power method không tìm thấy giá trị riêng ở bước {i+1}. Dừng.")
    #     break
    # lambda_sq_i = lambda_sq_i[0] # Lấy giá trị riêng đầu tiên tìm được
    # v_i = v_i_matrix[:,0] # Lấy vector riêng đầu tiên

    if lambda_sq_i < 0: # Giá trị riêng của M=A.T@A phải không âm
        print(f"Cảnh báo: Giá trị riêng lambda_sq_i = {lambda_sq_i:.4f} là âm. Có thể có vấn đề số học.")
        # Có thể dừng hoặc lấy trị tuyệt đối tùy theo yêu cầu
        lambda_sq_i = np.abs(lambda_sq_i)


    # Tính vector kỳ dị trái u_i
    # u_i = (1 / sigma_i) * A @ v_i, với sigma_i = sqrt(lambda_sq_i)
    sigma_i = np.sqrt(lambda_sq_i)
    if sigma_i < tol : # Nếu sigma quá nhỏ
         print(f"Giá trị kỳ dị sigma_i {sigma_i:.4e} quá nhỏ, dừng sớm tại bước {i+1}.")
         break
    u_i = (1 / sigma_i) * (A @ v_i)
    u_i = u_i / norm(u_i) # Chuẩn hóa u_i

    lambdas_sq.append(lambda_sq_i)
    V_singular_vectors = np.hstack((V_singular_vectors, v_i.reshape(-1, 1)))
    U_singular_vectors = np.hstack((U_singular_vectors, u_i.reshape(-1, 1)))

    # Xuống thang ma trận M_current
    # Vì M_current đối xứng, w_i = v_i
    M_current = deflation_wielandt(M_current, lambda_sq_i, v_i, v_i)
    # Hoặc đơn giản hơn: M_current = M_current - lambda_sq_i * np.outer(v_i, v_i)
    # (nếu v_i đã được chuẩn hóa, v_i.T @ v_i = 1)
    
    print(f'\nGiá trị riêng (lambda^2) thứ {i+1} của A.T@A là {lambda_sq_i:10.7f}')
    print(f'  Giá trị kỳ dị (sigma) thứ {i+1} là {sigma_i:10.7f}')
    print(f'  Vector kỳ dị phải v_{i+1} là:\n{v_i}')
    print(f'  Vector kỳ dị trái u_{i+1} là:\n{u_i}')
    # print(f'  Ma trận M sau khi xuống thang lần {i+1}:\n{M_current}')


# --- Sắp xếp các giá trị kỳ dị và vector tương ứng từ lớn đến nhỏ ---
# (Các giá trị riêng của A.T@A đã được tìm từ lớn đến nhỏ bởi power method và xuống thang)
sorted_indices = np.argsort(lambdas_sq)[::-1] # Sắp xếp giảm dần
lambdas_sq_sorted = np.array(lambdas_sq)[sorted_indices]
sigmas_sorted = np.sqrt(lambdas_sq_sorted)

V_singular_vectors_sorted = V_singular_vectors[:, sorted_indices]
U_singular_vectors_sorted = U_singular_vectors[:, sorted_indices]


print("\n\n--- Khai triển SVD rút gọn (Reduced SVD) ---")
# Sigma_reduced là ma trận đường chéo với các giá trị kỳ dị khác 0
num_significant_singular_values = len(sigmas_sorted) # Hoặc rank_A
Sigma_reduced_diag = sigmas_sorted[:num_significant_singular_values]
Sigma_reduced_matrix = np.diag(Sigma_reduced_diag)

print("Ma trận Sigma (rút gọn):\n", Sigma_reduced_matrix)
print("Ma trận U (rút gọn):\n", U_singular_vectors_sorted)
print("Ma trận V (rút gọn):\n", V_singular_vectors_sorted)

# Kiểm tra U.T @ U = I, V.T @ V = I
print("\nKiểm tra tính trực giao của U rút gọn (U.T @ U):")
print(U_singular_vectors_sorted.T @ U_singular_vectors_sorted)
print("Kiểm tra tính trực giao của V rút gọn (V.T @ V):")
print(V_singular_vectors_sorted.T @ V_singular_vectors_sorted)

# Kiểm tra A_approx = U_reduced @ Sigma_reduced @ V_reduced.T
A_reconstructed_reduced = U_singular_vectors_sorted @ Sigma_reduced_matrix @ V_singular_vectors_sorted.T
print("\nMa trận A tái tạo từ SVD rút gọn (U_red @ Sigma_red @ V_red.T):\n", A_reconstructed_reduced)
print(f"Sai số so với A gốc (norm Frobenius): {norm(A - A_reconstructed_reduced):.2e}")


print("\n\n--- Khai triển SVD đầy đủ (Full SVD) ---")
# Tạo ma trận Sigma đầy đủ
Sigma_full = np.zeros(A.shape, dtype=float) # Kích thước m x n (4, 6)
num_singular_values_to_place = len(sigmas_sorted) # Số lượng giá trị kỳ dị thực sự tìm được

# Xác định kích thước của khối đường chéo để điền
# Nó sẽ là min(số hàng của Sigma_full, số cột của Sigma_full)
# Hoặc đơn giản là điền vào các phần tử Sigma_full[i,i]
# cho đến khi hết giá trị kỳ dị hoặc hết đường chéo.
diag_len = min(Sigma_full.shape) # min(4,6) = 4

for i in range(min(diag_len, num_singular_values_to_place)):
    Sigma_full[i, i] = sigmas_sorted[i]

print("Ma trận Sigma (đầy đủ):\n", Sigma_full)

# Kết hợp với các vector từ không gian null để tạo U_full và V_full
# Các cột của U_singular_vectors_sorted đã trực chuẩn với nhau
# Các cột của U_2_null_space cũng nên được trực chuẩn hóa và trực giao với U_singular_vectors_sorted
# Tuy nhiên, nếu U_singular_vectors_sorted đã bao gồm tất cả các hướng cần thiết (đủ rank_A cột),
# và U_2_null_space là cơ sở cho phần bù trực giao, thì việc hstack đơn giản có thể hoạt động
# nếu chúng đã trực giao với nhau.
# Một cách an toàn hơn là trực chuẩn hóa toàn bộ tập hợp.

# Nối các vector kỳ dị tìm được với các vector trong không gian null
# và sau đó trực chuẩn hóa toàn bộ (nếu cần thiết và các tập không trực giao sẵn)
if U_2_null_space.shape[1] > 0:
    U_combined = np.hstack((U_singular_vectors_sorted, U_2_null_space))
    # U_full_ortho = gram_schmidt_orthogonalization(U_combined) # Đảm bảo trực chuẩn
    # Tuy nhiên, SVD của A sẽ cho U và V đã trực chuẩn.
    # U_svd_A và Vh_svd_A.T là U_full và V_full chính xác.
else:
    # U_full_ortho = U_singular_vectors_sorted
    pass

if V_2_null_space.shape[1] > 0:
    V_combined = np.hstack((V_singular_vectors_sorted, V_2_null_space))
    # V_full_ortho = gram_schmidt_orthogonalization(V_combined)
else:
    # V_full_ortho = V_singular_vectors_sorted
    pass

# Sử dụng U và V từ np.linalg.svd(A) để có U_full và V_full chính xác và trực chuẩn
U_full_exact = U_svd_A
V_full_exact = Vh_svd_A.T # Vh là V.T, nên Vh.T là V

print("Ma trận U (đầy đủ - từ np.linalg.svd):\n", U_full_exact)
print("Ma trận V (đầy đủ - từ np.linalg.svd):\n", V_full_exact)

# Kiểm tra A = U_full @ Sigma_full @ V_full.T
A_reconstructed_full = U_full_exact @ Sigma_full @ V_full_exact.T
print("\nMa trận A tái tạo từ SVD đầy đủ (U_full @ Sigma_full @ V_full.T):\n", A_reconstructed_full)
print(f"Sai số so với A gốc (norm Frobenius): {norm(A - A_reconstructed_full):.2e}")