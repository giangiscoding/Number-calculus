import numpy as np

def phuong_phap_lap_don(alpha_matrix, beta_vector, x0_initial, epsilon=0.5e-3, max_iterations=100):
    """
    Thực hiện phương pháp lặp đơn để giải hệ phương trình x = alpha*x + beta.

    Tham số:
        alpha_matrix (np.ndarray): Ma trận alpha.
        beta_vector (np.ndarray): Vector beta.
        x0_initial (np.ndarray): Vector nghiệm khởi tạo x0.
        epsilon (float): Ngưỡng sai số cho điều kiện dừng.
        max_iterations (int): Số lần lặp tối đa để tránh vòng lặp vô hạn.

    Trả về:
        x_final (np.ndarray): Nghiệm cuối cùng tìm được.
        iteration_count (int): Số lần lặp đã thực hiện.
        final_error (float): Sai số cuối cùng ước lượng.
    """
    alpha = np.array(alpha_matrix, dtype=float)
    beta = np.array(beta_vector, dtype=float).reshape(-1, 1) # Đảm bảo beta là vector cột
    x0 = np.array(x0_initial, dtype=float).reshape(-1, 1)   # Đảm bảo x0 là vector cột

    if alpha.shape[0] != alpha.shape[1]:
        raise ValueError("Ma trận alpha phải là ma trận vuông.")
    if alpha.shape[0] != beta.shape[0] or alpha.shape[0] != x0.shape[0]:
        raise ValueError("Kích thước của alpha, beta, và x0 không khớp.")

    # Tính chuẩn cột (norm 1) của ma trận alpha
    # norm(matrix, 1) trong MATLAB tương đương với norm(matrix, ord=1) trong NumPy
    q_norm_alpha = np.linalg.norm(alpha, ord=1)
    print(f"Hệ số co q (chuẩn 1 của alpha): {q_norm_alpha:8.7f}")

    if q_norm_alpha >= 1:
        print("Cảnh báo: Chuẩn của alpha (q) >= 1. Phương pháp lặp có thể không hội tụ.")
        # Trong trường hợp này, he_so_co sẽ âm hoặc chia cho 0 nếu q_norm_alpha = 1.
        # Để tránh lỗi, ta có thể dừng hoặc xử lý đặc biệt.
        # Ở đây, ta sẽ tiếp tục nhưng sai số ước lượng có thể không đáng tin cậy.
        if q_norm_alpha == 1:
            he_so_co = float('inf') # Hoặc một giá trị lớn
        else:
            he_so_co = q_norm_alpha / (1 - q_norm_alpha) # Sẽ âm nếu q_norm_alpha > 1
    else:
        he_so_co = q_norm_alpha / (1 - q_norm_alpha)

    sai_so_ss = 1.0  # Giá trị khởi tạo cho sai_so_ss để bắt đầu vòng lặp
    iteration_count = 0

    print(f"Nghiệm khởi tạo x0: [{x0[0,0]:10.7f}  {x0[1,0]:10.7f}  {x0[2,0]:10.7f}]")

    while sai_so_ss > epsilon and iteration_count < max_iterations:
        # x1 = alpha * x0 + beta
        x1 = alpha @ x0 + beta  # Phép nhân ma trận trong NumPy

        iteration_count += 1
        print(f"Lần lặp {iteration_count}:")
        print(f"  x({iteration_count}) = [{x1[0,0]:10.7f}  {x1[1,0]:10.7f}  {x1[2,0]:10.7f}]")

        # ss = heso * norm(x1-x0, 1)
        # norm(vector, 1) trong MATLAB tính tổng các giá trị tuyệt đối của các phần tử vector
        # tương đương với norm(vector, ord=1) trong NumPy
        norm_diff_x1_x0 = np.linalg.norm(x1 - x0, ord=1)
        
        if q_norm_alpha < 1: # Chỉ ước lượng sai số nếu điều kiện hội tụ tiên nghiệm được thỏa mãn
            sai_so_ss = he_so_co * norm_diff_x1_x0
        else: # Nếu q >= 1, dùng sai số hậu nghiệm (dù không chặt chẽ như công thức trên)
            sai_so_ss = norm_diff_x1_x0
            if iteration_count == 1 and q_norm_alpha >=1: # In cảnh báo một lần
                 print(f"  Cảnh báo: q >= 1, sai số ước lượng dựa trên ||x1-x0||.")


        print(f"  Sai số ước lượng lần lặp thứ {iteration_count} là {sai_so_ss:10.7f}")

        x0 = x1.copy() # Cập nhật x0 cho lần lặp tiếp theo

    if iteration_count == max_iterations and sai_so_ss > epsilon:
        print(f"\nPhương pháp lặp đạt số lần lặp tối đa ({max_iterations}) mà chưa đạt ngưỡng sai số.")
    else:
        print(f"\nPhương pháp lặp hội tụ sau {iteration_count} lần lặp.")

    return x0, iteration_count, sai_so_ss

if __name__ == '__main__':
    alpha_mat = np.array([
        [-0.2, 0.3, 0.1],
        [-0.1, 0.25, 0.0],
        [0.3, -0.14, -0.2]
    ])

    beta_vec = np.array([4, -2, 1]) # Sẽ được reshape thành vector cột trong hàm

    x0_vec = np.array([0, 0, 0])    # Sẽ được reshape thành vector cột trong hàm
    
    # Ngưỡng sai số từ code MATLAB
    epsilon_target = 0.5e-3

    print("--- Bắt đầu phương pháp lặp đơn ---")
    nghiem_cuoi_cung, so_lan_lap, sai_so_cuoi_cung = phuong_phap_lap_don(
        alpha_mat,
        beta_vec,
        x0_vec,
        epsilon=epsilon_target
    )

    print("\n--- Kết quả cuối cùng ---")
    print(f"Nghiệm tìm được x = [{nghiem_cuoi_cung[0,0]:10.7f}  {nghiem_cuoi_cung[1,0]:10.7f}  {nghiem_cuoi_cung[2,0]:10.7f}]")
    print(f"Số lần lặp: {so_lan_lap}")
    print(f"Sai số ước lượng cuối cùng: {sai_so_cuoi_cung:10.7f}")

    # Để kiểm tra, có thể tìm nghiệm chính xác của x = alpha*x + beta
    # (I - alpha)x = beta  => x = (I - alpha)^-1 * beta
    I_mat = np.eye(alpha_mat.shape[0])
    try:
        x_exact = np.linalg.solve(I_mat - alpha_mat, beta_vec.reshape(-1,1))
        print("\nNghiệm chính xác (từ (I-alpha)x = beta):")
        print(f"  x_exact = [{x_exact[0,0]:10.7f}  {x_exact[1,0]:10.7f}  {x_exact[2,0]:10.7f}]")
    except np.linalg.LinAlgError:
        print("\nKhông thể tính nghiệm chính xác do ma trận (I-alpha) suy biến.")