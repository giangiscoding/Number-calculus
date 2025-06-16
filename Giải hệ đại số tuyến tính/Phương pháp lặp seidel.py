import numpy as np

def phuong_phap_gauss_seidel(A_matrix, b_vector, x0_initial, epsilon=1e-6, max_iterations=100, convergence_check_type="absolute_difference"):
    """
    Giải hệ phương trình tuyến tính Ax = b bằng phương pháp lặp Gauss-Seidel.

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
        history (list): Danh sách các nghiệm x qua từng bước lặp.
    """
    A = np.array(A_matrix, dtype=float)
    b = np.array(b_vector, dtype=float)
    x_current = np.array(x0_initial, dtype=float) # Nghiệm ở bước lặp hiện tại (sẽ được cập nhật)

    n_size = A.shape[0]
    if A.shape[1] != n_size:
        raise ValueError("Ma trận A phải là ma trận vuông.")
    if b.shape[0] != n_size or x_current.shape[0] != n_size:
        raise ValueError("Kích thước của b hoặc x0 không khớp với A.")

    # Kiểm tra các phần tử trên đường chéo chính
    diagonal_A = np.diag(A)
    if np.any(diagonal_A == 0):
        print("Cảnh báo: Có phần tử bằng 0 trên đường chéo chính của A. "
              "Phương pháp Gauss-Seidel có thể không hội tụ hoặc gây lỗi chia cho 0.")
        # Hoặc raise lỗi:
        # raise ValueError("Phần tử trên đường chéo chính bằng 0, không thể thực hiện Gauss-Seidel.")

    iteration_count = 0
    final_error = float('inf')
    history = [x_current.copy()] # Lưu nghiệm ban đầu

    print(f"Nghiệm khởi tạo x(0): {x_current}")

    for k_iter in range(max_iterations):
        iteration_count = k_iter + 1
        x_previous_iter_for_error_check = x_current.copy() # Lưu lại nghiệm của vòng lặp trước để kiểm tra hội tụ

        for i_row in range(n_size):
            # Tính sigma1 = sum(A[i,j] * x_current[j]) for j < i (sử dụng x mới cập nhật)
            sigma1 = 0.0
            for j_col in range(i_row): # j từ 0 đến i_row - 1
                sigma1 += A[i_row, j_col] * x_current[j_col] # x_current[j_col] đã được cập nhật ở vòng lặp i_row này

            # Tính sigma2 = sum(A[i,j] * x_current[j]) for j > i (sử dụng x từ vòng lặp k_iter trước)
            # Lưu ý: trong cách triển khai này, x_current[j_col] cho j_col > i_row vẫn là giá trị từ vòng lặp k_iter trước
            # cho đến khi nó được cập nhật ở vòng lặp i_row tương ứng.
            sigma2 = 0.0
            for j_col in range(i_row + 1, n_size): # j từ i_row + 1 đến n_size - 1
                sigma2 += A[i_row, j_col] * x_current[j_col] # x_current[j_col] là giá trị từ x_previous_iter_for_error_check

            if diagonal_A[i_row] == 0:
                print(f"Lỗi: Phần tử đường chéo A[{i_row},{i_row}] bằng 0 tại lần lặp {iteration_count}.")
                x_current[i_row] = float('nan')
                final_error = float('inf')
                return x_current, iteration_count, final_error, history # Trả về sớm
            
            x_current[i_row] = (b[i_row] - sigma1 - sigma2) / diagonal_A[i_row]

        history.append(x_current.copy())
        print(f"x({iteration_count}): {x_current}")

        # Kiểm tra điều kiện hội tụ
        if convergence_check_type == "absolute_difference":
            current_error = np.linalg.norm(x_current - x_previous_iter_for_error_check, ord=np.inf)
        elif convergence_check_type == "relative_difference":
            norm_x_current = np.linalg.norm(x_current, ord=np.inf)
            if norm_x_current == 0:
                current_error = np.linalg.norm(x_current - x_previous_iter_for_error_check, ord=np.inf)
            else:
                current_error = np.linalg.norm(x_current - x_previous_iter_for_error_check, ord=np.inf) / norm_x_current
        elif convergence_check_type == "residual":
            residual = b - A @ x_current
            current_error = np.linalg.norm(residual, ord=np.inf)
        else:
            raise ValueError("Loại kiểm tra hội tụ không hợp lệ.")

        final_error = current_error
        print(f"  Sai số ({convergence_check_type}): {final_error:.2e}")

        if final_error < epsilon:
            print(f"\nPhương pháp Gauss-Seidel hội tụ sau {iteration_count} lần lặp.")
            return x_current, iteration_count, final_error, history

    print(f"\nPhương pháp Gauss-Seidel đạt số lần lặp tối đa ({max_iterations}) mà chưa đạt ngưỡng sai số.")
    return x_current, iteration_count, final_error, history

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

    print("--- Chạy phương pháp Gauss-Seidel ---")
    try:
        nghiem_seidel, so_lap_seidel, sai_so_seidel, lich_su_seidel = phuong_phap_gauss_seidel(
            A_example, b_example, x0_example,
            epsilon=epsilon_target,
            max_iterations=max_iter_limit,
            convergence_check_type="absolute_difference"
        )
        print("\nKết quả cuối cùng (Gauss-Seidel):")
        print(f"Nghiệm x = {nghiem_seidel}")
        print(f"Số lần lặp: {so_lap_seidel}")
        print(f"Sai số cuối cùng: {sai_so_seidel:.2e}")

        # So sánh với nghiệm chính xác
        # x_exact = np.linalg.solve(A_example, b_example)
        # print("\nNghiệm chính xác (từ np.linalg.solve):")
        # print(x_exact)
        # print(f"Sai số so với nghiệm chính xác: {np.linalg.norm(nghiem_seidel - x_exact, ord=np.inf):.2e}")

    except ValueError as e:
        print(f"Lỗi: {e}")

    # So sánh với Jacobi (sử dụng code Jacobi từ câu trả lời trước)
    # print("\n\n--- Để so sánh, chạy lại Jacobi với cùng tham số ---")
    # try:
    #     nghiem_jacobi, so_lap_jacobi, _, _ = phuong_phap_jacobi( # giả sử hàm này đã tồn tại
    #         A_example, b_example, x0_example,
    #         epsilon=epsilon_target,
    #         max_iterations=max_iter_limit,
    #         convergence_check_type="absolute_difference"
    #     )
    #     print(f"Jacobi: {so_lap_jacobi} lần lặp, nghiệm {nghiem_jacobi}")
    #     print(f"Gauss-Seidel: {so_lap_seidel} lần lặp, nghiệm {nghiem_seidel}")
    # except NameError:
    #     print("Hàm phuong_phap_jacobi chưa được định nghĩa để so sánh.")
    # except Exception as e:
    #     print(f"Lỗi khi chạy Jacobi để so sánh: {e}")