import numpy as np
import heapq
from typing import List, Tuple, Union
import time
from dataset import generate_dataset, generate_queries, load_text_dataset
class BruteForceSearch:
    """
    Brute-force linear search implementation for nearest neighbor search
    Serves as ground truth baseline for comparing with HNSW
    """
    
    def __init__(self, metric: str = 'euclidean'): #tạo placeholder cho dataset
        """
        Args:
            metric: Distance metric ('euclidean' or 'cosine')
        """
        self.metric = metric
        self.data = None
        self.dimension = None
        
    def fit(self, data: np.ndarray) -> None:#nạp dữ liệu
        """
        Store the dataset for brute-force search
        
        Args:
            data: numpy array of shape (n_samples, n_features)
        """
        self.data = data.astype(np.float32) #xài float32 thay vì float64 để tối ưu mem và speed nhưng vẫn đảm bảo precision
        self.dimension = data.shape[1]
        print(f"BruteForce: Loaded {len(data)} vectors of dimension {self.dimension}")
    
    def euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float: #tính độ dài euclide
        """Calculate Euclidean distance between two vectors"""
        return np.sqrt(np.sum((a - b) ** 2))
    
    def cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float: #tính độ dài cosine , giá trị thuộc đoạn [0,2]
        """Calculate cosine distance between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return 1 - (dot_product / (norm_a * norm_b))
    
    def _single_query(self, query: np.ndarray, k: int = 1) -> Tuple[List[int], List[float]]: #sử dụng heap thay sort 
        #Ứng dung: dataset nhỏ (k<100) sẽ tối ưu hơn, hoặc chỉ cần lấy top 10, top 50 với dataset lớn
        """
        Perform brute-force search for a single query vector
        
        Args:
            query: Query vector of shape (n_features,)
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (indices, distances) for k nearest neighbors
        """
        if self.data is None:
            raise ValueError("Must call fit() before querying")
        
        # Calculate distances to all points
        distances = []
        for i, vector in enumerate(self.data):
            if self.metric == 'euclidean':
                dist = self.euclidean_distance(query, vector)
            elif self.metric == 'cosine':
                dist = self.cosine_distance(query, vector)
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")
            distances.append((dist, i))
        
        # Get k smallest distances
        heapq.heapify(distances) #tạo heap : O(n)
        k_smallest = heapq.nsmallest(k, distances) #lấy k phần tử nhỏ nhất: O(k*log(n))
        # Tổng độ phức tạp: O(n + k*log(n))
        #Tách indices và distances
        indices = [idx for _, idx in k_smallest]
        distances = [dist for dist, _ in k_smallest]
        
        return indices, distances
    
    def query(self, query: np.ndarray, k: int = 1) -> Tuple[List[int], List[float]]:
        """
        Query the brute-force index
        
        Args:
            query: Query vector or batch of queries
            k: Number of nearest neighbors to return
            
        Returns:
            For single query: (indices, distances)
            For batch queries: list of (indices, distances) tuples
        """
        if len(query.shape) == 1:
            # Single query (1-dimension)
            return self._single_query(query, k)
        else:
            # Batch queries (2+ dimensions)
            results = []
            for q in query:
                results.append(self._single_query(q, k))
            return results
    
    def optimized_query(self, query: np.ndarray, k: int = 1) -> Tuple[List[int], List[float]]:
        """
        Optimized version using vectorized operations
        """
        if self.data is None:
            raise ValueError("Must call fit() before querying")
        
        if self.metric == 'euclidean':
            # Vectorized Euclidean distance: sqrt(sum((A-B)^2))
            differences = self.data - query #trừ toàn bộ ma trận
            distances = np.sqrt(np.sum(differences ** 2, axis=1)) #tính toán toàn bộ
        elif self.metric == 'cosine':
            # Vectorized cosine distance: 1 - (A·B)/(||A||·||B||)
            query_norm = np.linalg.norm(query)
            data_norms = np.linalg.norm(self.data, axis=1)
            dot_products = np.dot(self.data, query)
            distances = 1 - (dot_products / (data_norms * query_norm))
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        # Get k smallest distances
        if k == 1:
            min_idx = np.argmin(distances)
            return [min_idx], [distances[min_idx]]
        else:
            indices = np.argpartition(distances, k)[:k] #phân vùng nhanh -> tìm k indices có giá trị nhỏ nhất (chưa sort)
            # Sort the k smallest
            sorted_indices = indices[np.argsort(distances[indices])] # chỉ sort k phần tử đã chọn => độ phức tạp: O(n+klogk) thay vì full sort: O(nlogn)
            return sorted_indices.tolist(), distances[sorted_indices].tolist()

    def batch_query(self, queries: np.ndarray, k: int = 1, verbose: bool = False) -> List[Tuple[List[int], List[float]]]:
        """
        Perform batch queries with timing information
        
        Args:
            queries: Query vectors of shape (n_queries, n_features)
            k: Number of nearest neighbors to return
            verbose: Whether to print timing information
            
        Returns:
            List of (indices, distances) tuples for each query
        """
        start_time = time.time() #bắt đầu đếm thời gian để benchmark
        
        results = []
        for i, query in enumerate(queries): #duyệt từng index và distance
            indices, distances = self.optimized_query(query, k) 
            results.append((indices, distances))
            
            if verbose and i % 1000 == 0: #hiển thị tiến độ mỗi 1000 queries
                print(f"Processed {i}/{len(queries)} queries")
        
        end_time = time.time()
        
        if verbose:
            total_time = end_time - start_time #tổng thời gian thực thi
            qps = len(queries) / total_time # số queries per sec
            print(f"BruteForce Batch Query: {len(queries)} queries in {total_time:.4f}s "
                  f"({qps:.2f} QPS), k={k}")
        
        return results

    
def test_1M_vectors_top10():
    """
    Test hiệu năng với 1 triệu vectors 512 chiều, tìm top 10 nearest neighbors
    """
    print("=" * 70)
    print("TEST 1M VECTORS - TOP 10 NEAREST NEIGHBORS")
    print("=" * 70)
    
    # Tạo dataset 1 triệu vectors 512 chiều
    print("🚀 Generating 1,000,000 vectors × 512 dimensions...")
    start_time = time.time()
    
    np.random.seed(42)
    dataset = np.random.randn(1000000, 512).astype(np.float32)
    
    # Normalize vectors (quan trọng cho cosine distance)
    norms = np.linalg.norm(dataset, axis=1, keepdims=True)
    dataset = dataset / norms
    
    # Tạo query vectors
    queries = np.random.randn(10, 512).astype(np.float32)
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries = queries / norms
    
    print(f"✅ Dataset: {dataset.shape}")
    print(f"✅ Queries: {queries.shape}")
    print(f"⏱️  Generation time: {time.time() - start_time:.2f}s")
    
    # Test với Euclidean distance
    print("\n--- Euclidean Distance Test ---")
    bf_euclidean = BruteForceSearch(metric='euclidean')
    bf_euclidean.fit(dataset)
    
    # Test single query
    print(f"\n🎯 Testing single query (top 10)...")
    query_start = time.time()
    indices, distances = bf_euclidean.optimized_query(queries[0], k=10)
    query_time = time.time() - query_start
    
    print(f"⏱️  Query time: {query_time:.4f}s")
    print(f"🚀 QPS: {1/query_time:.2f}")
    print(f"📊 Top 10 results:")
    for i, (idx, dist) in enumerate(zip(indices, distances)):
        print(f"    {i+1:2d}. Index: {idx:7d}, Distance: {dist:.6f}")
    
    # Test với Cosine distance
    print("\n--- Cosine Distance Test ---")
    bf_cosine = BruteForceSearch(metric='cosine')
    bf_cosine.fit(dataset)
    
    cosine_start = time.time()
    cosine_indices, cosine_distances = bf_cosine.optimized_query(queries[0], k=10)
    cosine_time = time.time() - cosine_start
    
    print(f"⏱️  Query time: {cosine_time:.4f}s")
    print(f"🚀 QPS: {1/cosine_time:.2f}")
    print(f"📊 Top 10 results:")
    for i, (idx, dist) in enumerate(zip(cosine_indices, cosine_distances)):
        print(f"    {i+1:2d}. Index: {idx:7d}, Distance: {dist:.6f}")
    
    # So sánh kết quả Euclidean vs Cosine
    print(f"\n--- Comparison ---")
    common_indices = set(indices) & set(cosine_indices)
    print(f"📈 Common indices in top 10: {len(common_indices)}")
    print(f"📈 Euclidean top 1: {indices[0]}, Cosine top 1: {cosine_indices[0]}")
    
    # Benchmark với multiple queries
    print(f"\n--- Batch Query Benchmark ---")
    test_queries = queries[:5]  # Test với 5 queries
    
    # Euclidean batch
    euclidean_start = time.time()
    euclidean_results = bf_euclidean.batch_query(test_queries, k=10, verbose=False)
    euclidean_batch_time = time.time() - euclidean_start
    
    # Cosine batch  
    cosine_start = time.time()
    cosine_results = bf_cosine.batch_query(test_queries, k=10, verbose=False)
    cosine_batch_time = time.time() - cosine_start
    
    print(f"📊 5 queries batch performance:")
    print(f"   Euclidean: {euclidean_batch_time:.2f}s ({5/euclidean_batch_time:.2f} QPS)")
    print(f"   Cosine:    {cosine_batch_time:.2f}s ({5/cosine_batch_time:.2f} QPS)")
    
    # Performance estimates
    print(f"\n--- Performance Estimates ---")
    print(f"⏳ 100 queries: {query_time * 100:.1f}s ({query_time * 100/60:.1f} minutes)")
    print(f"⏳ 1,000 queries: {query_time * 1000:.1f}s ({query_time * 1000/60:.1f} minutes)")
    print(f"⏳ 10,000 queries: {query_time * 10000:.1f}s ({query_time * 10000/3600:.1f} hours)")
    
    # Memory info
    dataset_memory = dataset.nbytes / (1024**3)
    print(f"\n💾 Memory usage: {dataset_memory:.2f} GB")
    
    print("\n" + "=" * 70)
    print("TEST 1M VECTORS COMPLETED!")
    print("=" * 70)
    
    return query_time

def test_1M_vectors_optimized():
    """
    Test nhanh với optimized query để so sánh hiệu năng
    """
    print("\n" + "=" * 70)
    print("QUICK TEST: OPTIMIZED vs NORMAL QUERY")
    print("=" * 70)
    
    
    np.random.seed(42)
    small_dataset = np.random.randn(1000000, 512).astype(np.float32)
    norms = np.linalg.norm(small_dataset, axis=1, keepdims=True)
    small_dataset = small_dataset / norms
    
    small_query = np.random.randn(512).astype(np.float32)
    small_query = small_query / np.linalg.norm(small_query)
    
    print(f"Dataset: {small_dataset.shape}")
    
    bf = BruteForceSearch(metric='euclidean')
    bf.fit(small_dataset)
    
    # Test normal query
    print(f"\n🔍 Testing normal query...")
    normal_start = time.time()
    normal_indices, normal_distances = bf.query(small_query, k=10)
    normal_time = time.time() - normal_start
    
    # Test optimized query
    print(f"🔍 Testing optimized query...")
    optimized_start = time.time()
    optimized_indices, optimized_distances = bf.optimized_query(small_query, k=10)
    optimized_time = time.time() - optimized_start
    
    print(f"\n📊 RESULTS COMPARISON:")
    print(f"   Normal query:    {normal_time:.4f}s")
    print(f"   Optimized query: {optimized_time:.4f}s")
    print(f"   ⚡ Speedup: {normal_time/optimized_time:.2f}x")
    
    # Verify results match
    if normal_indices == optimized_indices and np.allclose(normal_distances, optimized_distances):
        print(f"   ✅ Results match perfectly!")
    else:
        print(f"   ⚠️  Results differ (expected with floating point)")
        
    return optimized_time

if __name__ == "__main__":
    print("BRUTE-FORCE TEST SUITE")
    print("Choose test to run:")
    print("1. Quick optimized vs normal test (1M vectors)")
    print("2. Full 1M vectors test (top 10 neighbors)")
    print("3. Both tests")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        test_1M_vectors_optimized()
    elif choice == "2":
        test_1M_vectors_top10()
    elif choice == "3":
        test_1M_vectors_optimized()
        test_1M_vectors_top10()
    else:
        print("Running quick test...")
        test_1M_vectors_optimized()