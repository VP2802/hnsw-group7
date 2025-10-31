import numpy as np
from typing import Tuple, Optional

def generate_dataset(num_data: int, dim: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Sinh dữ liệu random bằng NumPy.
    
    Args:
        num_data (int): Số lượng vector dữ liệu.
        dim (int): Số chiều của vector.
        seed (int, optional): Giá trị random seed để tái lập kết quả.
    
    Returns:
        np.ndarray: Ma trận (num_data x dim) chứa các vector dữ liệu.
    """
    if seed is not None:
        np.random.seed(seed)
    data = np.random.random((num_data, dim)).astype(np.float32)
    return data


def generate_queries(num_queries: int, dim: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Sinh các vector query random.
    
    Args:
        num_queries (int): Số lượng vector query.
        dim (int): Số chiều của vector.
        seed (int, optional): Giá trị random seed.
    
    Returns:
        np.ndarray: Ma trận (num_queries x dim) chứa các vector query.
    """
    if seed is not None:
        np.random.seed(seed + 1)  # khác seed dataset để tránh trùng
    queries = np.random.random((num_queries, dim)).astype(np.float32)
    return queries


def load_text_dataset(sentences: list[str], dim: int, seed: Optional[int] = None) -> np.ndarray:
    """
    (Tùy chọn nâng cao) Mã hóa văn bản thành vector ngẫu nhiên mô phỏng embedding.
    
    Args:
        sentences (list[str]): Danh sách câu.
        dim (int): Kích thước vector embedding.
    
    Returns:
        np.ndarray: Ma trận (len(sentences) x dim).
    """
    if seed is not None:
        np.random.seed(seed)
    embeddings = np.random.random((len(sentences), dim)).astype(np.float32)
    return embeddings




