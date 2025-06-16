import numpy as np

def phuong_phap_jacobi(A_matrix, b_vector, x0_initial, epsilon=1e-6, max_iterations=100, convergence_check_type="absolute_difference"):
    """
    Giải hệ phương trình tuyến tính Ax = b bằng phương pháp lặp Jacobi.

    Tham số:
        A_matrix (np.ndarray): Ma trận hệ số A (vuông).
        b_vector (np.ndarray): Vector vế phải b.
        x0_initial (np.ndarray): Vector nghiệm khởi tạo x0.
        epsilon (float): Ngưỡng sai số cho điều kiện dừng.
        max_iterations (int): Số lần lặp tối đa để tránh vòng lặp vô hạn.
        convergence_check_type (str): Loại kiểm tra hội tụ:
            "absolute_difference": norm(x_new - x_old) < epsilon
            "relative_difference": norm(x_new - x_old) / norm(x_new) < epsilon
            "residual": norm(A @ x_new - b) < epsilon

    Trả về:
        x_final (np.ndarray): Nghiệm cuối cùng tìm được.
        iteration_count (int): Số lần lặp đã thực hiện.
        final_error (float): Sai số cuối cùng dựa trên loại kiểm tra hội tụ.
        history (list): Danh sách các nghiệm x qua từng bước lặp (tùy chọn, có thể bỏ nếu không cần).
    """
    A = np.array(A_matrix, dtype=float)
    b = np.array(b_vector, dtype=float)
    x_old = np.array(x0_initial, dtype=float)

    n_size = A.shape[0]
    if A.shape[1] != n_size:
        raise ValueError("Ma trận A phải là ma trận vuông.")
    if b.shape[0] != n_size or x_old.shape[0] != n_size:
        raise ValueError("Kích thước của b hoặc x0 không khớp với A.")

    # Kiểm tra các phần tử trên đường chéo chính
    diagonal_A = np.diag(A)
    if np.any(diagonal_A == 0):
        print("Cảnh báo: Có phần tử bằng 0 trên đường chéo chính của A. "
              "Phương pháp Jacobi có thể không hội tụ hoặc gây lỗi chia cho 0.")
        # Hoặc raise lỗi:
        # raise ValueError("Phần tử trên đường chéo chính bằng 0, không thể thực hiện Jacobi.")

    x_new = np.zeros_like(x_old)
    iteration_count = 0
    final_error = float('inf')
    history = [x_old.copy()] # Lưu nghiệm ban đầu

    print(f"Nghiệm khởi tạo x(0): {x_old}")

    for k_iter in range(max_iterations):
        iteration_count = k_iter + 1
        for i_row in range(n_size):
            # Tính tổng sigma = sum(A[i,j] * x_old[j]) for j != i
            sigma = 0.0
            for j_col in range(n_size):
                if i_row != j_col:
                    sigma += A[i_row, j_col] * x_old[j_col]
            
            if diagonal_A[i_row] == 0: # Xử lý trường hợp đường chéo bằng 0
                print(f"Lỗi: Phần tử đường chéo A[{i_row},{i_row}] bằng 0 tại lần lặp {iteration_count}.")
                # Có thể chọn dừng hoặc gán một giá trị lớn để báo hiệu không hội tụ
                x_new[i_row] = float('nan') # Not a Number
                final_error = float('inf')
                return x_new, iteration_count, final_error, history # Trả về sớm
            
            x_new[i_row] = (b[i_row] - sigma) / diagonal_A[i_row]

        history.append(x_new.copy())
        print(f"x({iteration_count}): {x_new}")

        # Kiểm tra điều kiện hội tụ
        if convergence_check_type == "absolute_difference":
            current_error = np.linalg.norm(x_new - x_old, ord=np.inf) # Chuẩn vô cùng (max absolute diff)
        elif convergence_check_type == "relative_difference":
            norm_x_new = np.linalg.norm(x_new, ord=np.inf)
            if norm_x_new == 0: # Tránh chia cho 0 nếu x_new là vector không
                current_error = np.linalg.norm(x_new - x_old, ord=np.inf)
            else:
                current_error = np.linalg.norm(x_new - x_old, ord=np.inf) / norm_x_new
        elif convergence_check_type == "residual":
            residual = b - A @ x_new
            current_error = np.linalg.norm(residual, ord=np.inf)
        else:
            raise ValueError("Loại kiểm tra hội tụ không hợp lệ. Chọn từ: 'absolute_difference', 'relative_difference', 'residual'.")

        final_error = current_error
        print(f"  Sai số ({convergence_check_type}): {final_error:.2e}")

        if final_error < epsilon:
            print(f"\nPhương pháp Jacobi hội tụ sau {iteration_count} lần lặp.")
            return x_new, iteration_count, final_error, history

        x_old = x_new.copy() # Cập nhật x_old cho lần lặp tiếp theo

    print(f"\nPhương pháp Jacobi đạt số lần lặp tối đa ({max_iterations}) mà chưa đạt ngưỡng sai số.")
    return x_new, iteration_count, final_error, history

# --- Cách khác để triển khai phần tính toán x_new (sử dụng ma trận D, L, U) ---
def phuong_phap_jacobi_matrix_form(A_matrix, b_vector, x0_initial, epsilon=1e-6, max_iterations=100, convergence_check_type="absolute_difference"):
    A = np.array(A_matrix, dtype=float)
    b = np.array(b_vector, dtype=float)
    x_old = np.array(x0_initial, dtype=float)
    n_size = A.shape[0]

    # Tách ma trận A
    D_mat = np.diag(np.diag(A))
    L_mat = np.tril(A, k=-1) # k=-1 để không lấy đường chéo
    U_mat = np.triu(A, k=1)  # k=1 để không lấy đường chéo
    
    # Ma trận lặp T_jacobi = -D_inv @ (L + U)
    # Vector c_jacobi = D_inv @ b
    # x_new = T_jacobi @ x_old + c_jacobi
    
    # Kiểm tra D_mat có phần tử 0 trên đường chéo không
    if np.any(np.diag(D_mat) == 0):
        raise ValueError("Phần tử trên đường chéo chính của A bằng 0, không thể tính D nghịch đảo.")
    
    D_inv = np.diag(1.0 / np.diag(D_mat)) # Chỉ hoạt động nếu D_mat là ma trận đường chéo
    T_jacobi = -D_inv @ (L_mat + U_mat)
    c_jacobi = D_inv @ b

    iteration_count = 0
    final_error = float('inf')
    history = [x_old.copy()]

    print(f"Ma trận lặp T_jacobi:\n{T_jacobi}")
    # Kiểm tra điều kiện hội tụ tiên nghiệm: norm(T_jacobi) < 1
    norm_T_inf = np.linalg.norm(T_jacobi, ord=np.inf)
    print(f"Chuẩn vô cùng của T_jacobi: {norm_T_inf:.4f}")
    if norm_T_inf >= 1:
        print("Cảnh báo: Chuẩn của ma trận lặp T_jacobi >= 1. Phương pháp có thể không hội tụ.")

    print(f"\nNghiệm khởi tạo x(0): {x_old}")

    for k_iter in range(max_iterations):
        iteration_count = k_iter + 1
        x_new = T_jacobi @ x_old + c_jacobi
        
        history.append(x_new.copy())
        print(f"x({iteration_count}): {x_new}")

        if convergence_check_type == "absolute_difference":
            current_error = np.linalg.norm(x_new - x_old, ord=np.inf)
        elif convergence_check_type == "relative_difference":
            norm_x_new = np.linalg.norm(x_new, ord=np.inf)
            current_error = np.linalg.norm(x_new - x_old, ord=np.inf) / (norm_x_new if norm_x_new != 0 else 1e-16)
        elif convergence_check_type == "residual":
            current_error = np.linalg.norm(b - A @ x_new, ord=np.inf)
        else:
            raise ValueError("Loại kiểm tra hội tụ không hợp lệ.")
        
        final_error = current_error
        print(f"  Sai số ({convergence_check_type}): {final_error:.2e}")

        if final_error < epsilon:
            print(f"\nPhương pháp Jacobi (dạng ma trận) hội tụ sau {iteration_count} lần lặp.")
            return x_new, iteration_count, final_error, history
        
        x_old = x_new.copy()

    print(f"\nPhương pháp Jacobi (dạng ma trận) đạt số lần lặp tối đa ({max_iterations}) mà chưa đạt ngưỡng sai số.")
    return x_new, iteration_count, final_error, history


if __name__ == '__main__':
    # Ví dụ sử dụng
    A_example = np.array([[10.0, -1.0,  2.0,  0.0],
                          [-1.0, 11.0, -1.0,  3.0],
                          [ 2.0, -1.0, 10.0, -1.0],
                          [ 0.0,  3.0, -1.0,  8.0]])

    b_example = np.array([6.0, 25.0, -11.0, 15.0])

    x0_example = np.array([0.0, 0.0, 0.0, 0.0]) # Nghiệm khởi tạo

    epsilon_target = 1e-6
    max_iter_limit = 50

    print("--- Chạy phương pháp Jacobi (dạng thành phần) ---")
    try:
        nghiem_jacobi, so_lap_jacobi, sai_so_jacobi, lich_su_jacobi = phuong_phap_jacobi(
            A_example, b_example, x0_example,
            epsilon=epsilon_target,
            max_iterations=max_iter_limit,
            convergence_check_type="absolute_difference" # hoặc "relative_difference" hoặc "residual"
        )
        print("\nKết quả cuối cùng (Jacobi - dạng thành phần):")
        print(f"Nghiệm x = {nghiem_jacobi}")
        print(f"Số lần lặp: {so_lap_jacobi}")
        print(f"Sai số cuối cùng: {sai_so_jacobi:.2e}")

        # Kiểm tra với nghiệm chính xác
        # x_exact = np.linalg.solve(A_example, b_example)
        # print("\nNghiệm chính xác (từ np.linalg.solve):")
        # print(x_exact)
        # print(f"Sai số so với nghiệm chính xác: {np.linalg.norm(nghiem_jacobi - x_exact, ord=np.inf):.2e}")

    except ValueError as e:
        print(f"Lỗi: {e}")

    print("\n\n--- Chạy phương pháp Jacobi (dạng ma trận) ---")
    try:
        nghiem_jacobi_mat, so_lap_jacobi_mat, sai_so_jacobi_mat, lich_su_jacobi_mat = phuong_phap_jacobi_matrix_form(
            A_example, b_example, x0_example,
            epsilon=epsilon_target,
            max_iterations=max_iter_limit,
            convergence_check_type="absolute_difference"
        )
        print("\nKết quả cuối cùng (Jacobi - dạng ma trận):")
        print(f"Nghiệm x = {nghiem_jacobi_mat}")
        print(f"Số lần lặp: {so_lap_jacobi_mat}")
        print(f"Sai số cuối cùng: {sai_so_jacobi_mat:.2e}")
    except ValueError as e:
        print(f"Lỗi: {e}")

    # Ví dụ ma trận không chéo trội (có thể không hội tụ hoặc hội tụ chậm)
    # A_not_diag_dominant = np.array([[1.0, 2.0], [3.0, 1.0]])
    # b_not_diag_dominant = np.array([3.0, 4.0])
    # x0_not_diag_dominant = np.array([0.0, 0.0])
    # print("\n\n--- Thử với ma trận không chéo trội ---")
    # try:
    #     phuong_phap_jacobi_matrix_form(
    #         A_not_diag_dominant, b_not_diag_dominant, x0_not_diag_dominant,
    #         epsilon=1e-3, max_iterations=50
    #     )
    # except ValueError as e:
    #     print(f"Lỗi: {e}")