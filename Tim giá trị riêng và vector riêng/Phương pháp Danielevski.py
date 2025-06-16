import numpy as np
from numpy.linalg import inv, norm # Lưu ý: inv() không được sử dụng tường minh cho nghịch đảo của phép biến đổi tương tự; chúng được xây dựng trực tiếp.

def danilevski(ma_tran_goc_A, nguong_cat=1e-5):
    """
    Phương pháp Danilevski tìm giá trị riêng và vector riêng của ma trận ma_tran_goc_A
    
    Tham số:
        ma_tran_goc_A (np.ndarray): Ma trận vuông đầu vào
        nguong_cat (float): Ngưỡng để xem một giá trị là bằng 0
    
    Trả về:
        cac_gia_tri_rieng (np.ndarray): Mảng các giá trị riêng
        cac_vector_rieng (np.ndarray): Ma trận các vector riêng (sắp xếp theo cột)
        ma_tran_P_chuyen_co_so (np.ndarray): Ma trận chuyển cơ sở sao cho P @ ma_tran_goc_A @ P_nghich_dao gần với dạng Frobenius.
        ma_tran_P_nghich_dao (np.ndarray): Ma trận nghịch đảo của ma_tran_P_chuyen_co_so (P_nghich_dao = P^{-1})
        dang_Frobenius_F (np.ndarray): Dạng Frobenius của ma_tran_goc_A (F = P @ ma_tran_goc_A @ P_nghich_dao)
    """
    ma_tran_A_hien_tai = np.copy(ma_tran_goc_A) # Làm việc trên một bản sao
    kich_thuoc_n = ma_tran_A_hien_tai.shape[0]
    
    # cac_bien_doi_P_toan_cuc, cac_bien_doi_Pinv_toan_cuc sẽ tích lũy các phép biến đổi cho ma trận gốc ma_tran_goc_A
    cac_bien_doi_P_toan_cuc = np.eye(kich_thuoc_n)
    cac_bien_doi_Pinv_toan_cuc = np.eye(kich_thuoc_n)
    
    danh_sach_gia_tri_rieng_thu_thap = []
    
    khoi_A_dang_xu_ly = np.copy(ma_tran_A_hien_tai) # Đây là ma trận con sẽ bị thu nhỏ dần
    
    # do_lech_khoi_hien_tai theo dõi vị trí góc trên-trái của khoi_A_dang_xu_ly
    # trong hệ tọa độ của ma trận gốc ma_tran_goc_A.
    do_lech_khoi_hien_tai = 0

    while khoi_A_dang_xu_ly.shape[0] > 0:
        # P_bien_doi_khoi, Pinv_bien_doi_khoi là các phép biến đổi cho khoi_A_dang_xu_ly
        # Kích thước của chúng là khoi_A_dang_xu_ly.shape
        cac_gtri_rieng_tu_khoi_con, P_bien_doi_khoi, Pinv_bien_doi_khoi, khoi_A_tren_trai_tiep_theo = \
            xu_ly_khoi(khoi_A_dang_xu_ly, nguong_cat)
        
        danh_sach_gia_tri_rieng_thu_thap.extend(list(cac_gtri_rieng_tu_khoi_con))
        
        # Nhúng P_bien_doi_khoi và Pinv_bien_doi_khoi vào các ma trận kích thước n x n
        # S_buoc_day_du áp dụng P_bien_doi_khoi vào phần con hiện tại của ma trận
        S_buoc_day_du = np.eye(kich_thuoc_n)
        Sinv_buoc_day_du = np.eye(kich_thuoc_n)
        
        kich_thuoc_khoi_hien_tai = khoi_A_dang_xu_ly.shape[0]
        
        # Phép biến đổi P_bien_doi_khoi được suy ra cho khoi_A_dang_xu_ly.
        # Khối này nằm ở vị trí ma_tran_goc_A[do_lech_khoi_hien_tai : do_lech_khoi_hien_tai + kich_thuoc_khoi_hien_tai, ...]
        chi_so_bat_dau = do_lech_khoi_hien_tai
        chi_so_ket_thuc = do_lech_khoi_hien_tai + kich_thuoc_khoi_hien_tai
        
        S_buoc_day_du[chi_so_bat_dau:chi_so_ket_thuc, chi_so_bat_dau:chi_so_ket_thuc] = P_bien_doi_khoi
        Sinv_buoc_day_du[chi_so_bat_dau:chi_so_ket_thuc, chi_so_bat_dau:chi_so_ket_thuc] = Pinv_bien_doi_khoi
        
        # Tích lũy các phép biến đổi toàn cục:
        # P_toan_cuc_moi = S_buoc_day_du @ P_toan_cuc_cu
        # Pinv_toan_cuc_moi = Pinv_toan_cuc_cu @ Sinv_buoc_day_du
        # Thứ tự này đảm bảo Pinv_toan_cuc là nghịch đảo của P_toan_cuc
        cac_bien_doi_P_toan_cuc = S_buoc_day_du @ cac_bien_doi_P_toan_cuc
        cac_bien_doi_Pinv_toan_cuc = cac_bien_doi_Pinv_toan_cuc @ Sinv_buoc_day_du
        
        # Các giá trị riêng được trích xuất tương ứng với các hàng/cột "bị loại bỏ"
        # từ góc dưới-phải của khoi_A_dang_xu_ly để có được khoi_A_tren_trai_tiep_theo.
        # khoi_A_tren_trai_tiep_theo là khối mới cần xử lý.
        # Góc trên-trái của nó trong tọa độ gốc vẫn là do_lech_khoi_hien_tai.
        khoi_A_dang_xu_ly = khoi_A_tren_trai_tiep_theo
        
        # Nếu khoi_A_tren_trai_tiep_theo rỗng, vòng lặp sẽ kết thúc.

    cac_gia_tri_rieng = np.array(danh_sach_gia_tri_rieng_thu_thap)
    
    # Ma trận dạng Frobenius
    # Lưu ý: Phương pháp Danilevsky có thể không tạo ra một ma trận Frobenius hoàn hảo duy nhất
    # nếu ma trận bị chia thành các khối. F sẽ có dạng tam giác trên theo khối,
    # với các khối Frobenius (hoặc khối 1x1 cho các giá trị riêng thực) trên đường chéo.
    dang_Frobenius_F = cac_bien_doi_P_toan_cuc @ ma_tran_goc_A @ cac_bien_doi_Pinv_toan_cuc 
    dang_Frobenius_F[np.abs(dang_Frobenius_F) < nguong_cat] = 0.0
    
    # Tính các vector riêng
    # Vector riêng được tính cho dạng Frobenius cuối cùng F, sau đó biến đổi ngược lại.
    # y_i là vector riêng cho F. x_i = Pinv @ y_i là vector riêng cho A.
    # Đối với ma trận đồng hành (dạng Frobenius), nếu p(λ) = λ^n - (c_{n-1}λ^{n-1} + ... + c_1λ + c_0) = 0,
    # và F = [[c_{n-1}, ..., c_1, c_0], [1, 0,...,0,0], ..., [0,...,1,0,0]],
    # một vector riêng y cho giá trị riêng λ là [λ^{n-1}, λ^{n-2}, ..., λ, 1]^T.
    # Code Python mẫu sử dụng F[0,:] cho các hệ số, ngụ ý F là:
    # F = [[-p_{n-1}, -p_{n-2}, ..., -p_0], [1, 0, ..., 0], ..., [0, ..., 1, 0]] (nếu hệ số đầu của đa thức là 1)
    # hoặc F = [[a_{n-1}, a_{n-2}, ..., a_0], [1,0,...,0],...,[0,...,0,1,0]] từ mô tả bài toán.
    # Các hệ số cho np.roots là [coeff_n, coeff_{n-1}, ..., coeff_0].
    # Đối với F[0,:] = [c1, c2, ..., cn], đa thức đặc trưng là lambda^n - c1*lambda^{n-1} - ... - cn = 0
    # Code Python cho tinh_tri_rieng_frobenius: p = [1, -F[0,:]] cho roots.
    # Code Python cho tinh_vector_rieng: v = [λ^(N-1), λ^(N-2), ..., 1] (N = Pinv.shape[0])
    # Điều này là tiêu chuẩn cho ma trận đồng hành dạng C_top:
    # [[0, 0, ..., -c0], [1,0,...,-c1], ..., [0,..,1,-c_{n-1}]] (hệ số của x^n + c_{n-1}x^{n-1}...+c0)
    # Đối với F = [[a_1, ..., a_n], [1,0,...], ...], đa thức đặc trưng là det(lambda*I - F).
    # lambda - a1, -a2, ...
    # -1, lambda, 0, ...
    # Điều này dẫn đến lambda^n - a1*lambda^{n-1} - ... - an = 0.
    # Vector riêng y = [lambda^{n-1}, lambda^{n-2}, ..., lambda, 1]^T.
    # Điều này có vẻ nhất quán.

    so_luong_gtri_rieng = len(cac_gia_tri_rieng)
    cac_vector_rieng = np.zeros((kich_thuoc_n, so_luong_gtri_rieng), dtype=np.complex128) # Giá trị riêng có thể phức
    for i in range(so_luong_gtri_rieng):
        cac_vector_rieng[:, i] = tinh_vector_rieng(cac_gia_tri_rieng[i], cac_bien_doi_Pinv_toan_cuc, kich_thuoc_n) 
    
    return cac_gia_tri_rieng, cac_vector_rieng, cac_bien_doi_P_toan_cuc, cac_bien_doi_Pinv_toan_cuc, dang_Frobenius_F

def xu_ly_khoi(khoi_A_ban_dau, nguong_cat):
    """
    Xử lý một khối ma trận khoi_A_ban_dau để đưa về dạng Frobenius ở góc dưới phải (nếu có thể)
    hoặc tách các giá trị riêng.
    Trả về:
        cac_gtri_rieng_da_trich_xuat (np.ndarray): Các giá trị riêng được trích xuất từ khối này.
        P_cho_khoi (np.ndarray): Ma trận biến đổi tích lũy cho khoi_A_ban_dau.
        Pinv_cho_khoi (np.ndarray): Nghịch đảo của P_cho_khoi.
        khoi_A_con_lai_tren_trai (np.ndarray): Khối ma trận còn lại (góc trên trái) sau khi tách.
    """
    kich_thuoc_khoi_hien_tai_n = khoi_A_ban_dau.shape[0]
    if kich_thuoc_khoi_hien_tai_n == 0:
        return np.array([]), np.eye(0), np.eye(0), np.array([])
    
    ma_tran_A_lam_viec = np.copy(khoi_A_ban_dau)
    
    # P_cuc_bo, Pinv_cuc_bo là các phép biến đổi tích lũy cho khoi_A_ban_dau NÀY
    P_cuc_bo = np.eye(kich_thuoc_khoi_hien_tai_n)
    Pinv_cuc_bo = np.eye(kich_thuoc_khoi_hien_tai_n)
    
    # Danilevsky làm việc từ hàng thứ (n-1) lên hàng thứ 1 (0-indexed: n-2 xuống 0)
    # `count` trong MATLAB là số thứ tự hàng (1-indexed) của phần tử A(k,k-1) cần được zero hóa hoặc đặt thành 1.
    # MATLAB: count = n-1 (nhắm vào A(n, n-1)), sau đó n-2 (nhắm vào A(n-1,n-2)), ..., xuống 1 (nhắm vào A(2,1)).
    # Tương đương Python cho chỉ số hàng `k` trong A[k, k-1]: `k` từ `n-1` xuống `1`.
    # `A[k, k-1]` tương ứng với `A(k+1, k)` của MATLAB.
    
    # `chi_so_hang_truc` là chỉ số hàng k, chúng ta quan tâm đến A[chi_so_hang_truc, chi_so_hang_truc-1]
    chi_so_hang_truc = kich_thuoc_khoi_hien_tai_n - 1 # Bắt đầu với hàng thứ (n-1) (0-indexed)
                                 # Chúng ta muốn A[n-1, n-2] là 1 (nếu không tạo khối)
                                 # A[chi_so_hang_truc, chi_so_hang_truc-1]
    
    da_hoan_thanh_khoi = False # Trở thành true khi một khối được xử lý hoàn toàn hoặc ma trận bị tách
    
    cac_gtri_rieng_trich_xuat_lan_nay = np.array([])

    while not da_hoan_thanh_khoi:
        # Giai đoạn 1: Tạo các số 1 trên đường chéo phụ từ dưới lên
        # `chi_so_hang_truc` là k trong A[k, k-1]
        # Lặp trong khi A[chi_so_hang_truc, chi_so_hang_truc-1] khác không
        # và chi_so_hang_truc >= 1 (để chi_so_hang_truc-1 >= 0)
        while chi_so_hang_truc >= 1 and np.abs(ma_tran_A_lam_viec[chi_so_hang_truc, chi_so_hang_truc - 1]) > nguong_cat:
            # Biến đổi hàng chi_so_hang_truc-1
            # bien_doi_hang(A, k) trong code Python mẫu (k là hàng 1-indexed đang được sửa đổi, A[k-1,...])
            # A[k-1,:] là hàng trục. S[k-2,:] = A[k-1,:].
            # Ở đây k là (chi_so_hang_truc)+1 trong đánh số 1-based cho bien_doi_hang.
            # Hàng mà phần tử đường chéo phụ A[chi_so_hang_truc, chi_so_hang_truc-1] được sử dụng là chi_so_hang_truc.
            # Ma trận biến đổi M tác động lên hàng chi_so_hang_truc-1.
            A_moi, S_lap, Sinv_lap = bien_doi_hang(ma_tran_A_lam_viec, chi_so_hang_truc) # Truyền chỉ số hàng 0-indexed
            ma_tran_A_lam_viec = A_moi
            P_cuc_bo = S_lap @ P_cuc_bo
            Pinv_cuc_bo = Pinv_cuc_bo @ Sinv_lap
            chi_so_hang_truc -= 1
        
        # Sau vòng lặp: hoặc chi_so_hang_truc < 1 (đã xử lý tất cả các hàng)
        # HOẶC A[chi_so_hang_truc, chi_so_hang_truc-1] bằng không.

        if chi_so_hang_truc < 1: # Đã biến đổi thành công về dạng Frobenius (hoặc đã đến đỉnh)
            da_hoan_thanh_khoi = True
            # Toàn bộ ma trận A hiện tại bây giờ là một khối Frobenius
            # (hoặc đã trở thành 1x1 hoặc 0x0)
            if kich_thuoc_khoi_hien_tai_n > 0:
                 cac_gtri_rieng_trich_xuat_lan_nay = tinh_tri_rieng_frobenius(ma_tran_A_lam_viec)
            # khoi_A_con_lai_tren_trai sẽ rỗng
            khoi_A_con_lai_tren_trai = np.array([]) # rỗng
        else: # A[chi_so_hang_truc, chi_so_hang_truc-1] == 0 (hoặc rất nhỏ)
            # Ma trận đã bị tách hoặc cần xử lý đặc biệt cho trục bằng không
            # A = [A_tl, A_tr;
            #      0,    A_br]
            # trong đó A_br bắt đầu từ hàng chi_so_hang_truc trở đi.
            # A[chi_so_hang_truc, :chi_so_hang_truc] là đoạn hàng quyết định việc tách.
            # (Trong MATLAB Goi5: all(A(count+1,1:count)==0)==false -> nghĩa là chưa tách, gọi Goi3)
            # (count trong MATLAB là chi_so_hang_truc-1 theo nghĩa 0-indexed của Python cho cột)
            # A[chi_so_hang_truc, 0 : chi_so_hang_truc]
            
            # Nếu A[chi_so_hang_truc, 0...chi_so_hang_truc-1] toàn là số không, ma trận bị tách.
            if np.all(np.abs(ma_tran_A_lam_viec[chi_so_hang_truc, :chi_so_hang_truc]) < nguong_cat):
                # Ma trận bị tách: A_br là A[chi_so_hang_truc:, chi_so_hang_truc:]
                # A_tl là A[:chi_so_hang_truc, :chi_so_hang_truc]
                # A_tr là A[:chi_so_hang_truc, chi_so_hang_truc:]
                # Khối A[chi_so_hang_truc:, chi_so_hang_truc:] là một khối Frobenius (hoặc có thể được tạo thành).
                # Điều này được xử lý bởi trich_xuat_tri_rieng
                A_moi, S_lap, Sinv_lap, gtri_rieng_khi_tach = trich_xuat_tri_rieng(ma_tran_A_lam_viec, chi_so_hang_truc, nguong_cat)
                ma_tran_A_lam_viec = A_moi
                P_cuc_bo = S_lap @ P_cuc_bo
                Pinv_cuc_bo = Pinv_cuc_bo @ Sinv_lap
                cac_gtri_rieng_trich_xuat_lan_nay = gtri_rieng_khi_tach
                
                # Ma trận còn lại để xử lý tiếp là A[:chi_so_hang_truc, :chi_so_hang_truc]
                # khoi_A_con_lai_tren_trai = ma_tran_A_lam_viec[:chi_so_hang_truc, :chi_so_hang_truc] # Logic này cần xem lại ở cuối hàm
                da_hoan_thanh_khoi = True # Xử lý khối này đã xong
            else:
                # A[chi_so_hang_truc, chi_so_hang_truc-1] bằng không, nhưng A[chi_so_hang_truc, :chi_so_hang_truc] không toàn là số không.
                # Đây là "trường hợp ngoại lệ" trong Danilevsky.
                # Cần hoán vị các cột (và hàng) để đưa một phần tử khác không đến A[chi_so_hang_truc, chi_so_hang_truc-1].
                # Code Python mẫu có xu_ly_hang_khong cho việc này.
                # MATLAB Goi3 dùng cho trường hợp này.
                # k trong `xu_ly_hang_khong(A, count+1)` trong đó count+1 là 1-based.
                # Vậy nó là `chi_so_hang_truc` theo 0-based.
                A_moi, S_lap, Sinv_lap = xu_ly_hang_khong(ma_tran_A_lam_viec, chi_so_hang_truc, nguong_cat)
                ma_tran_A_lam_viec = A_moi
                P_cuc_bo = S_lap @ P_cuc_bo
                Pinv_cuc_bo = Pinv_cuc_bo @ Sinv_lap
                # Sau bước này, A[chi_so_hang_truc, chi_so_hang_truc-1] phải khác không,
                # để vòng lặp chính `while chi_so_hang_truc >= 1 and ...` có thể tiếp tục.
                # Vòng lặp `while not da_hoan_thanh_khoi` tiếp tục; `chi_so_hang_truc` không thay đổi.
    
    # Tại điểm này, `da_hoan_thanh_khoi` là true.
    # `cac_gtri_rieng_trich_xuat_lan_nay` chứa các giá trị riêng từ khối dưới-phải đã xử lý.
    # `khoi_A_con_lai_tren_trai` là phần còn lại. Nếu toàn bộ ma trận trở thành Frobenius, nó sẽ rỗng.
    
    # Code Python mẫu ngụ ý A2 là A[:count, :count] trong đó count là `chi_so_hang_truc` của Python.
    # `chi_so_hang_truc` này là chỉ số hàng phía trên khối đã tách.
    if chi_so_hang_truc < 0: # Điều này có thể xảy ra nếu kich_thuoc_khoi_hien_tai_n là 0 hoặc 1.
        khoi_A_con_lai_tren_trai = np.array([])
    elif chi_so_hang_truc == 0: # kich_thuoc_khoi_hien_tai_n là 1, đã xử lý.
         khoi_A_con_lai_tren_trai = np.array([])
    else: # Trường hợp này xảy ra nếu ma trận bị tách, chi_so_hang_truc được đặt đúng bởi logic tách.
         khoi_A_con_lai_tren_trai = ma_tran_A_lam_viec[:chi_so_hang_truc, :chi_so_hang_truc]


    return cac_gtri_rieng_trich_xuat_lan_nay, P_cuc_bo, Pinv_cuc_bo, khoi_A_con_lai_tren_trai


def bien_doi_hang(ma_tran_A_dau_vao, chi_so_hang_k_truc, nguong_cat=1e-9): # chi_so_hang_k_truc là 0-indexed
    """
    Biến đổi hàng chi_so_hang_k_truc-1 của ma_tran_A_dau_vao để đưa ma_tran_A_dau_vao[chi_so_hang_k_truc, chi_so_hang_k_truc-1] thành 1.
    S tác động lên hàng chi_so_hang_k_truc-1.
    MATLAB Goi2(A, k): k là 1-indexed. S(k-1,:) = A(k,:). Nghĩa là hàng thứ (k-1) của S là hàng thứ k của A.
    Python: Chỉ số hàng mục tiêu cho S là chi_so_hang_k_truc-1. Dữ liệu từ A[chi_so_hang_k_truc, :].
    Vậy S[chi_so_hang_k_truc-1, :] = A[chi_so_hang_k_truc, :]
    Và Sinv được xây dựng dựa trên điều này.
    """
    kich_thuoc_n = ma_tran_A_dau_vao.shape[0]
    ma_tran_S = np.eye(kich_thuoc_n, dtype=ma_tran_A_dau_vao.dtype)
    
    # chi_so_hang_k_truc là hàng chứa trục A[chi_so_hang_k_truc, chi_so_hang_k_truc-1]
    if chi_so_hang_k_truc < 1 or chi_so_hang_k_truc >= kich_thuoc_n: 
        # Trường hợp này lý tưởng không nên xảy ra nếu logic đúng, hoặc xử lý ma trận 1x1.
        # Đối với ma trận 1x1, chi_so_hang_k_truc có thể là 0, không cần biến đổi.
        return ma_tran_A_dau_vao, ma_tran_S, np.copy(ma_tran_S)

    # Phần tử trục là A[chi_so_hang_k_truc, chi_so_hang_k_truc-1]
    gia_tri_truc = ma_tran_A_dau_vao[chi_so_hang_k_truc, chi_so_hang_k_truc - 1]
    if np.abs(gia_tri_truc) < nguong_cat:
        # Hàm này giả định gia_tri_truc khác không.
        # Nếu bằng không, xu_ly_khoi nên đã chuyển đến xu_ly_hang_khong hoặc trich_xuat_tri_rieng.
        # raise ValueError("Phần tử trục bằng không trong bien_doi_hang, không thể tiếp tục.")
        # Hoặc, trả về ma trận đơn vị nếu trường hợp này nghĩa là không có thao tác. Để an toàn:
        return ma_tran_A_dau_vao, ma_tran_S, np.copy(ma_tran_S)


    # Hàng chi_so_hang_k_truc-1 của S được đặt bằng hàng chi_so_hang_k_truc của A
    ma_tran_S[chi_so_hang_k_truc - 1, :] = ma_tran_A_dau_vao[chi_so_hang_k_truc, :]
    
    ma_tran_S_nghich_dao = np.eye(kich_thuoc_n, dtype=ma_tran_A_dau_vao.dtype)
    ma_tran_S_nghich_dao[chi_so_hang_k_truc - 1, :] = -ma_tran_A_dau_vao[chi_so_hang_k_truc, :] / gia_tri_truc
    ma_tran_S_nghich_dao[chi_so_hang_k_truc - 1, chi_so_hang_k_truc - 1] = 1.0 / gia_tri_truc
    
    A_sau_bien_doi = ma_tran_S @ ma_tran_A_dau_vao @ ma_tran_S_nghich_dao
    return A_sau_bien_doi, ma_tran_S, ma_tran_S_nghich_dao

def xu_ly_hang_khong(ma_tran_A_dau_vao, chi_so_hang_k_truc, nguong_cat=1e-9): # chi_so_hang_k_truc là hàng 0-indexed
    """
    Xử lý trường hợp đặc biệt khi A[chi_so_hang_k_truc, chi_so_hang_k_truc-1] == 0 nhưng
    A[chi_so_hang_k_truc, :chi_so_hang_k_truc] không phải toàn số 0.
    Tìm cột j < chi_so_hang_k_truc-1 sao cho A[chi_so_hang_k_truc, j] != 0.
    Hoán vị cột j và cột chi_so_hang_k_truc-1.
    (Tương ứng với MATLAB Goi3)
    """
    kich_thuoc_n = ma_tran_A_dau_vao.shape[0]
    # Chúng ta cần tìm một phần tử khác không A[chi_so_hang_k_truc, j] trong đó j < chi_so_hang_k_truc-1
    # và hoán vị cột j với cột chi_so_hang_k_truc-1.
    # Điều này cũng đòi hỏi hoán vị hàng j với hàng chi_so_hang_k_truc-1 để duy trì tính tương tự.
    
    # Tìm j sao cho A[chi_so_hang_k_truc, j] khác không, j < chi_so_hang_k_truc-1
    # (MATLAB dường như ngụ ý j có thể là bất kỳ trong 1:count, count ở đây là chi_so_hang_k_truc-1)
    chi_so_cot_j_hoan_vi = -1
    for j_tiem_nang in range(chi_so_hang_k_truc - 1): # Các cột từ 0 đến chi_so_hang_k_truc-2
        if np.abs(ma_tran_A_dau_vao[chi_so_hang_k_truc, j_tiem_nang]) > nguong_cat:
            chi_so_cot_j_hoan_vi = j_tiem_nang
            break
    
    if chi_so_cot_j_hoan_vi == -1:
        # Không nên xảy ra nếu hàm này được gọi đúng
        # (tức là, A[chi_so_hang_k_truc, :chi_so_hang_k_truc] không toàn là số không, và A[chi_so_hang_k_truc,chi_so_hang_k_truc-1]==0)
        # Điều này ngụ ý tất cả A[chi_so_hang_k_truc, :chi_so_hang_k_truc-1] đều bằng không, nghĩa là đây là trường hợp tách.
        # Để an toàn, trả về ma trận đơn vị nếu không tìm thấy hoán vị phù hợp.
        return ma_tran_A_dau_vao, np.eye(kich_thuoc_n), np.eye(kich_thuoc_n)

    # Phép biến đổi tương tự: Hoán vị cột chi_so_cot_j_hoan_vi với chi_so_hang_k_truc-1
    # VÀ hàng chi_so_cot_j_hoan_vi với chi_so_hang_k_truc-1
    # Điều này được thực hiện bởi P_hoan_vi * A * P_hoan_vi (vì P_hoan_vi_nghich_dao = P_hoan_vi cho hoán vị)
    
    S_hoan_vi = np.eye(kich_thuoc_n, dtype=ma_tran_A_dau_vao.dtype)
    # Hoán vị các hàng chi_so_cot_j_hoan_vi và chi_so_hang_k_truc-1 trong S_hoan_vi để tạo ma trận hoán vị hàng
    S_hoan_vi[[chi_so_cot_j_hoan_vi, chi_so_hang_k_truc - 1], :] = S_hoan_vi[[chi_so_hang_k_truc - 1, chi_so_cot_j_hoan_vi], :]
    
    # Sinv_hoan_vi chính là S_hoan_vi cho loại hoán vị này.
    Sinv_hoan_vi = S_hoan_vi 
    
    A_sau_bien_doi = S_hoan_vi @ ma_tran_A_dau_vao @ Sinv_hoan_vi # A_moi = P A P_inv
    
    return A_sau_bien_doi, S_hoan_vi, Sinv_hoan_vi


def trich_xuat_tri_rieng(ma_tran_A_dau_vao, chi_so_hang_k_tach_khoi, nguong_cat): # chi_so_hang_k_tach_khoi là hàng đầu tiên của khoi_F
    """
    Trích xuất giá trị riêng từ khối Frobenius khoi_F (ma_tran_A_dau_vao[chi_so_hang_k_tach_khoi:, chi_so_hang_k_tach_khoi:])
    và biến đổi ma trận để không ảnh hưởng ma_tran_A_dau_vao[:chi_so_hang_k_tach_khoi, :chi_so_hang_k_tach_khoi].
    (Tương ứng với MATLAB Goi4)
    ma_tran_A_dau_vao = [ khoi_A2  khoi_B  ]
                       [ 0        khoi_F  ]
    chi_so_hang_k_tach_khoi chỉ ra điểm bắt đầu của hàng '0' và khối 'F'.
    khoi_A2 là ma_tran_A_dau_vao[:chi_so_hang_k_tach_khoi, :chi_so_hang_k_tach_khoi]
    khoi_B là ma_tran_A_dau_vao[:chi_so_hang_k_tach_khoi, chi_so_hang_k_tach_khoi:]
    khoi_F là ma_tran_A_dau_vao[chi_so_hang_k_tach_khoi:, chi_so_hang_k_tach_khoi:]
    ma_tran_khong_Z là ma trận không, thường có kích thước khoi_B.shape[1] x khoi_B.shape[0] (hoặc ma trận không có kích thước khoi_B.T)
    """
    kich_thuoc_n = ma_tran_A_dau_vao.shape[0]
    ban_sao_A = np.copy(ma_tran_A_dau_vao) # Làm việc trên bản sao nếu các bước trung gian sửa đổi nó

    khoi_F = ban_sao_A[chi_so_hang_k_tach_khoi:, chi_so_hang_k_tach_khoi:]
    khoi_A2 = ban_sao_A[:chi_so_hang_k_tach_khoi, :chi_so_hang_k_tach_khoi:]
    khoi_B = ban_sao_A[:chi_so_hang_k_tach_khoi, chi_so_hang_k_tach_khoi:]

    if khoi_F.shape[0] == 0: # Không có khoi_F để xử lý
        return ban_sao_A, np.eye(kich_thuoc_n), np.eye(kich_thuoc_n), np.array([])
    
    cac_gtri_rieng_cua_F = tinh_tri_rieng_frobenius(khoi_F)
    
    # S_tong_cong, Sinv_tong_cong tích lũy các phép biến đổi cho hàm này
    S_tong_cong = np.eye(kich_thuoc_n, dtype=ma_tran_A_dau_vao.dtype)
    Sinv_tong_cong = np.eye(kich_thuoc_n, dtype=ma_tran_A_dau_vao.dtype)

    # Phần này của Goi4 / trich_xuat_tri_rieng nhằm mục đích zero hóa khoi_B bằng các phép biến đổi
    # S = [I, M; 0, I], Sinv = [I, -M; 0, I]
    # A_moi = S @ A @ Sinv = [I,M;0,I] @ [A2,B;0,F] @ [I,-M;0,I]
    #         = [A2, B+M*F-A2*M; 0, F]
    # Chúng ta muốn B+M*F-A2*M = 0. Đây là một phương trình Sylvester cho M.
    # Cấu trúc code MATLAB khác:
    # Nó dường như lặp qua các cột của B.
    # for col = 1:size(B,2)-1 ... Bplus(:,col+1) = B(:,col); Bminus = -Bplus
    # S1 = [eye, Bminus; Z, eye]; S = S1*S
    # B = A2*Bplus + B + Bminus*F;
    # Điều này ngụ ý biến đổi B lặp đi lặp lại.
    # Đặt B_hien_tai = B.
    # Trong mỗi bước, M thực tế chỉ có một cột khác không (Bminus).
    # M = Bminus. Chúng ta muốn zero hóa B_hien_tai[:, col]. Điều này ngụ ý cột (col+1) của M nên triệt tiêu B_hien_tai[:,col].
    # Phần này của thuật toán có thể phức tạp và có thể yêu cầu triển khai cẩn thận
    # khớp với biến thể Danilevsky cụ thể.

    # Code Python mẫu cho trich_xuat_tri_rieng dường như theo logic zero hóa lặp đi lặp lại B này.
    # Hãy điều chỉnh logic đó.
    ma_tran_khong_Z_cho_khoi = np.zeros((khoi_F.shape[0], khoi_A2.shape[0]), dtype=ma_tran_A_dau_vao.dtype) # Z = zeros(size(B'))

    B_hien_tai = np.copy(khoi_B) # B sẽ được sửa đổi

    for chi_so_cot in range(B_hien_tai.shape[1] -1): # MATLAB: 1 đến size(B,2)-1
        B_cong_M_cot = np.zeros_like(B_hien_tai)
        B_cong_M_cot[:, chi_so_cot + 1] = B_hien_tai[:, chi_so_cot] # M có một cột hoạt động
        B_tru_M_cot = -B_cong_M_cot

        S1_lap = np.block([
            [np.eye(khoi_A2.shape[0]), B_tru_M_cot],
            [ma_tran_khong_Z_cho_khoi, np.eye(khoi_F.shape[0])]
        ])
        S_tong_cong = S1_lap @ S_tong_cong
        
        S1inv_lap = np.block([
            [np.eye(khoi_A2.shape[0]), B_cong_M_cot], # Lưu ý: đây là -B_tru_M_cot
            [ma_tran_khong_Z_cho_khoi, np.eye(khoi_F.shape[0])]
        ])
        Sinv_tong_cong = Sinv_tong_cong @ S1inv_lap # Tích lũy Pinv_moi = Pinv_cu @ Sinv_lap

        # Cập nhật B_hien_tai dựa trên phép biến đổi A_moi = S1_lap @ trang_thai_A_hien_tai @ S1inv_lap
        # B_moi = khoi_A2 @ B_cong_M_cot + B_hien_tai + B_tru_M_cot @ khoi_F
        B_hien_tai = khoi_A2 @ B_cong_M_cot + B_hien_tai + B_tru_M_cot @ khoi_F
        # Cập nhật ma trận tổng thể ban_sao_A cho nhất quán nếu cần, hoặc chỉ theo dõi B_hien_tai
        ban_sao_A[:chi_so_hang_k_tach_khoi, chi_so_hang_k_tach_khoi:] = B_hien_tai
        # ban_sao_A[chi_so_hang_k_tach_khoi:, :chi_so_hang_k_tach_khoi] phải giữ nguyên là không do cấu trúc S1, S1inv

    # Xử lý nếu cột cuối cùng của B không phải là không (MATLAB: if all(B(:,end)==0) == false)
    if not np.all(np.abs(B_hien_tai[:, -1]) < nguong_cat):
        # Hoán vị các cột của F và các hàng/cột tương ứng của B
        # W = eye; W = [W(:,end), W(:,1:end-1)]; (Chuyển cột cuối của F lên đầu)
        W_hoan_vi = np.eye(khoi_F.shape[0], dtype=ma_tran_A_dau_vao.dtype)
        if khoi_F.shape[0] > 0: # Tránh lỗi trên khoi_F 0x0
            W_hoan_vi = np.hstack([W_hoan_vi[:, -1:], W_hoan_vi[:, :-1]])
        
        khoi_U_hoan_vi = np.block([
            [np.eye(khoi_A2.shape[0]), np.zeros_like(B_hien_tai)], 
            [ma_tran_khong_Z_cho_khoi, W_hoan_vi]
        ])
        
        W_hoan_vi_nghich_dao = np.eye(khoi_F.shape[0], dtype=ma_tran_A_dau_vao.dtype)
        if khoi_F.shape[0] > 0:
            W_hoan_vi_nghich_dao = np.hstack([W_hoan_vi[:, 1:], W_hoan_vi[:, :1]]) # Chuyển cột đầu ra sau cùng

        khoi_U_hoan_vi_nghich_dao = np.block([
            [np.eye(khoi_A2.shape[0]), np.zeros_like(B_hien_tai)],
            [ma_tran_khong_Z_cho_khoi, W_hoan_vi_nghich_dao]
        ])

        S_tong_cong = khoi_U_hoan_vi @ S_tong_cong
        Sinv_tong_cong = Sinv_tong_cong @ khoi_U_hoan_vi_nghich_dao

        # Cập nhật khoi_F và B_hien_tai
        khoi_F = W_hoan_vi @ khoi_F @ W_hoan_vi_nghich_dao # F_moi = W @ F @ W_nghich_dao
        B_hien_tai = B_hien_tai @ W_hoan_vi_nghich_dao      # B_moi = B @ W_nghich_dao

        # Cập nhật ban_sao_A
        ban_sao_A[:chi_so_hang_k_tach_khoi, chi_so_hang_k_tach_khoi:] = B_hien_tai
        ban_sao_A[chi_so_hang_k_tach_khoi:, chi_so_hang_k_tach_khoi:] = khoi_F
    
    # Sau các phép biến đổi này, khối B (hoặc các phần của nó) phải bằng không.
    # Ma trận ban_sao_A đã được biến đổi.
    A_da_bien_doi = S_tong_cong @ ma_tran_A_dau_vao @ Sinv_tong_cong 
                                                    
    A_da_bien_doi[np.abs(A_da_bien_doi) < nguong_cat] = 0.0

    return A_da_bien_doi, S_tong_cong, Sinv_tong_cong, cac_gtri_rieng_cua_F


def tinh_tri_rieng_frobenius(ma_tran_F):
    """
    Tính giá trị riêng từ khối Frobenius ma_tran_F.
    Giả sử ma_tran_F ở dạng ma trận đồng hành trong đó hàng đầu tiên chứa các hệ số.
    F = [[a_1, a_2, ..., a_n],
         [1,   0,   ..., 0  ],
         [0,   1,   ..., 0  ],
         ...,
         [0,   0,   ...,1,0]]
    Đa thức đặc trưng: lambda^n - a_1*lambda^(n-1) - ... - a_n = 0
    Hệ số cho np.roots: [1, -a_1, -a_2, ..., -a_n]
    """
    kich_thuoc_cua_F_n = ma_tran_F.shape[0]
    if kich_thuoc_cua_F_n == 0:
        return np.array([])
    if kich_thuoc_cua_F_n == 1:
        return np.array([ma_tran_F[0,0]])
    
    # Hệ số cho đa thức: lambda^n - F[0,0]*lambda^(n-1) - F[0,1]*lambda^(n-2) ... - F[0,n-1]
    # np.roots mong đợi [coeff_n, coeff_{n-1}, ..., coeff_0]
    # Vì vậy, đối với p(x) = c_n*x^n + ... + c_1*x + c_0, truyền [c_n, ..., c_0]
    # Đa thức của chúng ta là x^n - F[0,0]*x^{n-1} - ... - F[0,n-1]*x^0
    # Vậy các hệ số là [1, -F[0,0], -F[0,1], ..., -F[0,n-1]]
    
    # Logic từ code Python mẫu/MATLAB Goi1:
    # MATLAB Goi1: p = (-1)^(n+1)*[1, -1*F(1,:)];
    # Nếu n=1, p = [1, -F(1,1)]. Nghiệm của x - F(1,1)=0 là F(1,1). Khớp.
    # Nếu n=2, F=[f1 f2; 1 0]. p = (-1)^3 * [1, -f1, -f2] = [-1, f1, f2].
    # Nghiệm của -x^2+f1*x+f2=0  => x^2-f1*x-f2=0. Điều này đúng cho F=[f1 f2; 1 0].
    # Code Python mẫu: p = [(-1)**(n+1)] ; p.extend(-F[0, :])
    # Điều này ngụ ý đa thức đặc trưng: (-1)^(n+1) * (lambda^n + sum(F[0,j] lambda^{n-1-j}))
    # Điều này tương ứng với dạng Frobenius hoặc định nghĩa hơi khác.
    
    # Sử dụng dạng đơn giản hơn, phổ biến dựa trực tiếp vào F[0,:]:
    he_so_da_thuc = np.zeros(kich_thuoc_cua_F_n + 1, dtype=ma_tran_F.dtype)
    he_so_da_thuc[0] = 1.0
    if kich_thuoc_cua_F_n > 0: # chỉ gán nếu có hàng đầu tiên
        he_so_da_thuc[1:] = -ma_tran_F[0, :]
    
    if kich_thuoc_cua_F_n == 0: return np.array([])
    return np.roots(he_so_da_thuc)

def tinh_vector_rieng(gia_tri_rieng_lambda, ma_tran_Pinv, kich_thuoc_goc_n):
    """
    Tính vector riêng từ giá trị riêng gia_tri_rieng_lambda và ma_tran_Pinv.
    kich_thuoc_goc_n là kích thước của bài toán gốc (và của ma_tran_Pinv).
    Vector riêng y cho ma trận Frobenius F (liên quan bởi P A Pinv = F)
    cho giá trị riêng λ là y = [λ^(n-1), λ^(n-2), ..., λ, 1]^T.
    Vector riêng x cho A là x = Pinv @ y.
    """
    kich_thuoc_n_cho_vector_mu = kich_thuoc_goc_n # Kích thước cho vector lũy thừa
    # Tạo vector [λ^(n-1), λ^(n-2), ..., 1]
    if kich_thuoc_n_cho_vector_mu == 0: return np.array([])
    
    y_vector_cho_F = np.array([gia_tri_rieng_lambda**(kich_thuoc_n_cho_vector_mu - 1 - i) for i in range(kich_thuoc_n_cho_vector_mu)], dtype=np.complex128)
    
    vector_rieng_cua_A = ma_tran_Pinv @ y_vector_cho_F 
    return vector_rieng_cua_A

if __name__ == "__main__":
    # Ma trận đầu vào
    A = np.array([
    [6.7562, 4.4584, 5.1176, 3.5945, 3.4311],
    [4.4584, 4.3051, 4.3327, 1.9363, 1.5788],
    [5.1176, 4.3327, 5.1287, 3.5097, 3.3354],
    [3.5945, 1.9363, 3.5097, 4.8411, 4.8972],
    [3.4311, 1.5788, 3.3354, 4.8972, 5.0688]
])
    
    # Chạy thuật toán
    cac_gtri_rieng_tim_duoc, cac_vector_rieng_tim_duoc, ma_tran_P_kq, ma_tran_Pinv_kq, F_dang_Frobenius_kq = danilevski(A, nguong_cat=1e-9)
    
    print("Các giá trị riêng tìm được:")
    # Sắp xếp giá trị riêng để so sánh nhất quán nếu cần, ví dụ sắp xếp theo phần thực
    chi_so_sap_xep = np.argsort(np.real(cac_gtri_rieng_tim_duoc))[::-1] # Phần thực giảm dần
    cac_gtri_rieng_tim_duoc = cac_gtri_rieng_tim_duoc[chi_so_sap_xep]
    cac_vector_rieng_tim_duoc = cac_vector_rieng_tim_duoc[:, chi_so_sap_xep]

    print(cac_gtri_rieng_tim_duoc)
    
    print("\nCác vector riêng (theo cột):")
    print(cac_vector_rieng_tim_duoc)
    
    print("\nMa trận chuyển cơ sở P:")
    print(ma_tran_P_kq)

    print("\nMa trận nghịch đảo Pinv:")
    print(ma_tran_Pinv_kq)
    
    print("\nDạng Frobenius F = P @ A @ Pinv:")
    print(F_dang_Frobenius_kq)
    
    # Kiểm tra P @ Pinv = I
    I_kiem_tra = ma_tran_P_kq @ ma_tran_Pinv_kq
    print(f"\nKiểm tra P @ Pinv (norm(P@Pinv - I)): {norm(I_kiem_tra - np.eye(A.shape[0])):.2e}")

    # Kiểm tra kết quả A*v - λ*v ≈ 0
    print("\nKiểm tra A*v - λ*v ≈ 0:")
    for i in range(len(cac_gtri_rieng_tim_duoc)):
        Av = A @ cac_vector_rieng_tim_duoc[:, i]
        lambda_v = cac_gtri_rieng_tim_duoc[i] * cac_vector_rieng_tim_duoc[:, i]
        sai_so_kiem_tra = norm(Av - lambda_v)
        print(f"λ = {cac_gtri_rieng_tim_duoc[i]:.4f}, sai số: {sai_so_kiem_tra:.2e}")
        if sai_so_kiem_tra > 1e-5: 
             print(f"  Cảnh báo: Sai số cao cho giá trị riêng {cac_gtri_rieng_tim_duoc[i]}")

    print("\nSo sánh với giá trị riêng từ NumPy:")
    cac_gtri_rieng_numpy = np.linalg.eigvals(A)
    print(np.sort(cac_gtri_rieng_numpy)[::-1]) # Sắp xếp để so sánh