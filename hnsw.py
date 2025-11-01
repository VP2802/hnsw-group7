# src/hnsw_wrapper.py
# Dùng thư viện hnswlib để xây dựng và quản lý chỉ mục HNSW
import hnswlib
import numpy as np

class HNSWIndex:
    def __init__(self, dim, space) :
        """ Hàm khởi tạo lớp HNSWIndex
        dim: số chiều của vector
        space: không gian khoảng cách ('l2' cho khoảng cách Euclidean, 'ip' cho tích vô hướng, 'cosine' cho khoảng cách cosine)
        """
        self.dim = dim
        self.space = space
        self.index = hnswlib.Index(space = space, dim = dim)
        # Tạo 1 index rỗng bằng API của thư viện hnsw
        self.is_initialized = False # Index chưa khởi tạo

    def init_index(self, max_elements, M = 16, ef_construction = 200, random_seed = 42) :
        """ Khởi tạo index (hàm khởi tạo)
        - max_elements: Số lượng vector tối đa mà index có thể chứa
        - M: Số lượng cạnh(edges) tối đa cho mỗi node trong đồ thị
        - ef_construction: Tham số kiểm soát độ chính xác của quá trình xây dựng index
        + Giá trị lớn hơn sẽ làm tăng độ chính xác nhưng cũng làm chậm quá trình xây dựng index
        - random_seed: seed của mỗi tập dữ liệu ngẫu nhiên
        """
        # dùng init_index của thư viện hnsw để đưa vào
        self.index.init_index(max_elements = max_elements, ef_construction = ef_construction, M = M, random_seed = random_seed, allow_replace_deleted = False)
        #Không cho phép thêm phần tử mới vào vị trí bị xóa
        self.is_initialized = True # Cho biết index đã khởi tạo

    def add_items(self, data, ids = None, num_threads=-1) :
        """ Hàm thêm vector vào index
        data: numpy array có shape (num_elements, dim) chứa các vector cần thêm vào index
        ids: numpy array có shape (num_elements,) hoặc None. Nếu None, các ID sẽ được tự động gán theo thứ tự.
        num_threads: số lượng luồng để sử dụng khi thêm vector (mặc định là -1, sử dụng tất cả các luồng có sẵn)
        """
        if not self.is_initialized :
            raise ValueError("Index is not initialized. Call init_index() before adding items") # Kiểm tra xem index đã được khởi tạo chưa 
        self.index.add_items(data, ids, num_threads = num_threads, replace_deleted = False) #add trong API của hnsw
    
    def set_query_params(self, ef = 50) :
        """ Cấu hình tham số truy vấn.
        ef: Tham số kiểm soát độ chính xác của truy vấn
        + Giá trị lớn hơn sẽ làm tăng độ chính xác nhưng cũng làm chậm quá trình truy vấn
        + Nên đặt ef >= k (số lượng lân cận cần tìm)
        """
        self.index.set_ef(ef)

    def knn_query(self, queries, k = 1) :
        """ Biểu diễn hàm truy vấn k-NN.
        queries: numpy array có shape (num_queries, dim) chứa các vector truy vấn
        k: số lượng lân cận cần tìm
        Trả về: (labels, distances)
        + labels: numpy array có shape (num_queries, k) chứa ID của k lân cận gần nhất cho mỗi truy vấn
        + distances: numpy array có shape (num_queries, k) chứa khoảng cách tương ứng
        """
        if not self.is_initialized :
            raise ValueError("Index is not initialized. Call init_index() before querying.")
        labels, distances = self.index.knn_query(queries, k = k)
        return labels, distances
    
    def save_index(self, path) :
        """ Lưu index vào file.
        path: đường dẫn file để lưu index
        """
        if not self.is_initialized :
            raise ValueError("Index is not initialized. Call init_index() before saving.")
        self.index.save_index(path)
    
    def load_index(self, path, max_elements) :
        """ Tải index từ file.
        path: đường dẫn file để tải index
        max_elements: Số lượng vector tối đa mà index có thể chứa (phải giống với giá trị khi khởi tạo index ban đầu)
        """
        self.index.load_index(path, max_elements = max_elements)
        self.is_initialized = True

    def get_current_count(self) :
        """ Trả về số lượng vector hiện có trong index.
        """
        if not self.is_initialized :
            raise ValueError("Index is not initialized. Call init_index() before getting current count.")
        return self.index.get_current_count()
    def get_max_elements(self) :
        """ Trả về số lượng vector tối đa mà index có thể chứa.
        """
        if not self.is_initialized :
            raise ValueError("Index is not initialized. Call init_index() before getting max elements.")
        return self.index.get_max_elements()
    def get_index_params(self) :
        """ Trả về các tham số cấu hình của index.
        """
        if not self.is_initialized :
            raise ValueError("Index is not initialized. Call init_index() before getting index params.")
        return {
            "space": self.index.space,
            "dim": self.index.dim,
            "M": self.index.M,
            "ef_construction": self.index.ef_construction,
            "max_elements": self.index.get_max_elements(),
            "element_count": self.index.get_current_count()
        }
    def get_query_params(self) :
        """ Trả về các tham số cấu hình truy vấn.
        """
        if not self.is_initialized :
            raise ValueError("Index is not initialized. Call init_index() before getting query params.")
        return {
            "ef" : self.index.ef,
            "num_threads": self.index.num_threads
        }
    def resize_index(self, new_max_elements) :
        """ Thay đổi kích thước tối đa của index.
        new_max_elements: Kích thước tối đa mới cho index (phải lớn hơn hoặc bằng số lượng vector hiện có)
        """
        if not self.is_initialized :
            raise ValueError("Index is not initialized. Call init_index() before resizing index.")
        self.index.resize_index(new_max_elements)
    
    def mark_deleted(self, id) :
        """ Đánh dấu một vector là đã bị xóa.
        """
        if not self.is_initialized :
            raise ValueError("Index is not initialized. Call init_index() before deleting items.")
        self.index.mark_deleted(id)
        # trong hnsw Khi xóa, phần tử không bị xoá khỏi bộ nhớ, nhưng bị bỏ qua trong kết quả tìm kiếm.
    def unmark_deleted(self, id) :
        """ Bỏ đánh dấu một vector đã bị xóa.
        """ 
        if not self.is_initialized :
            raise ValueError("Index is not initialized. Call init_index() before undeleting items.")
        self.index.unmark_deleted(id)

    def info(self):
        if not self.is_initialized:
            return "Index not initialized."
        params = self.get_index_params()
        query_params = self.get_query_params()
        return {
            "Index params": params,
            "Query params": query_params
    }

    