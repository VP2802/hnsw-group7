import numpy as np
import heapq
from typing import List, Tuple, Union
import time

class BruteForceSearch:
    """
    Brute-force linear search implementation for nearest neighbor search
    Serves as ground truth baseline for comparing with HNSW
    """
    
    def __init__(self, metric: str = 'euclidean'):
        """
        Args:
            metric: Distance metric ('euclidean' or 'cosine')
        """
        self.metric = metric
        self.data = None
        self.dimension = None
        self._data_norms = None  # Cache for cosine distance
        self._data_sq = None     # Cache for euclidean distance
        
    def fit(self, data: np.ndarray) -> None:
        """
        Store the dataset for brute-force search
        
        Args:
            data: numpy array of shape (n_samples, n_features)
        """
        self.data = data.astype(np.float32)
        self.dimension = data.shape[1]
        
        # Precompute for cosine distance
        if self.metric == 'cosine':
            self._data_norms = np.linalg.norm(self.data, axis=1)
            # Avoid division by zero
            self._data_norms[self._data_norms == 0] = 1e-10
        
        # Precompute for euclidean distance  
        if self.metric == 'euclidean':
            self._data_sq = np.sum(self.data ** 2, axis=1)
            
        print(f"BruteForce: Loaded {len(data)} vectors of dimension {self.dimension}")
    
    def euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate Euclidean distance between two vectors"""
        return np.sqrt(np.sum((a - b) ** 2))
    
    def cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine distance between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return 1 - (dot_product / (norm_a * norm_b))
    
    def _single_query_heap(self, query: np.ndarray, k: int = 1) -> Tuple[List[int], List[float]]:
        """
        Optimized version using fixed-size max-heap 
        Độ phức tạp: O(n*log(k)) thay vì O(n + k*log(n))
        """
        if self.data is None:
            raise ValueError("Must call fit() before querying")
        
        heap = []  # Max-heap: lưu (-distance, index)
        
        for i, vector in enumerate(self.data):
            if self.metric == 'euclidean':
                dist = self.euclidean_distance(query, vector)
            elif self.metric == 'cosine':
                dist = self.cosine_distance(query, vector)
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")
            
            # Dùng negative distance để tạo max-heap
            if len(heap) < k:
                heapq.heappush(heap, (-dist, i))
            else:
                if dist < -heap[0][0]:
                    heapq.heappushpop(heap, (-dist, i))
        
        # Extract and sort results
        sorted_results = sorted([(-dist, idx) for dist, idx in heap])
        indices = [idx for _, idx in sorted_results]
        distances = [dist for dist, _ in sorted_results]
        
        return indices, distances
    
    def _single_query_vectorized(self, query: np.ndarray, k: int = 1) -> Tuple[List[int], List[float]]:
        """
        Vectorized version - optimized for large datasets
        """
        if self.data is None:
            raise ValueError("Must call fit() before querying")
        
        # Calculate distances - OPTIMIZED VERSION
        if self.metric == 'euclidean':
            # Optimized Euclidean: ||a-b||² = ||a||² + ||b||² - 2a·b
            query_sq = np.dot(query, query)  # ||query||²
            dot_products = np.dot(self.data, query)  # a·b
            
            # Tính squared distances
            squared_distances = self._data_sq + query_sq - 2 * dot_products
            
            # Xử lý các giá trị âm do floating point errors
            squared_distances = np.maximum(squared_distances, 0)
            distances = np.sqrt(squared_distances)
            
        elif self.metric == 'cosine':
            query_norm = np.linalg.norm(query)
            if query_norm == 0:
                query_norm = 1e-10
            distances = 1 - np.dot(self.data, query) / (self._data_norms * query_norm)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        return self._get_top_k_optimized(distances, k)
    
    def _get_top_k_optimized(self, distances: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
        """
        Optimized top-k selection with adaptive strategy
        """
        n = len(distances)
        
        if k == 1:
            min_idx = np.argmin(distances)
            return [min_idx], [distances[min_idx]]
        elif k >= n:
            sorted_indices = np.argsort(distances)
        elif k < min(100, n // 4):  # Heuristic: k nhỏ
            indices = np.argpartition(distances, k)[:k]
            sorted_indices = indices[np.argsort(distances[indices])]
        else:
            sorted_indices = np.argsort(distances)[:k]
        
        return sorted_indices.tolist(), distances[sorted_indices].tolist()
    
    def query(self, query: np.ndarray, k: int = 1, method: str = 'auto') -> Tuple[List[int], List[float]]:
        """
        Query the brute-force index with method selection
        
        Args:
            query: Query vector or batch of queries
            k: Number of nearest neighbors to return
            method: 'heap', 'vectorized', or 'auto'
            
        Returns:
            For single query: (indices, distances)
            For batch queries: list of (indices, distances) tuples
        """
        if len(query.shape) == 1:
            # Single query
            if method == 'heap':
                return self._single_query_heap(query, k)
            elif method == 'vectorized':
                return self._single_query_vectorized(query, k)
            else:  # 'auto'
                # Tự động chọn method tốt nhất
                if k < 50 and len(self.data) > 1000:
                    return self._single_query_heap(query, k)
                else:
                    return self._single_query_vectorized(query, k)
        else:
            # Batch queries
            results = []
            for q in query:
                results.append(self._single_query_vectorized(q, k))
            return results
    
    def optimized_query(self, query: np.ndarray, k: int = 1) -> Tuple[List[int], List[float]]:
        """
        Optimized version - compact form (giữ nguyên interface cũ)
        """
        return self._single_query_vectorized(query, k)
    
    def batch_query(self, queries: np.ndarray, k: int = 1, verbose: bool = False, 
                   batch_size: int = 1000) -> List[Tuple[List[int], List[float]]]:
        """
        Perform batch queries with timing information và memory optimization
        
        Args:
            queries: Query vectors of shape (n_queries, n_features)
            k: Number of nearest neighbors to return
            verbose: Whether to print timing information
            batch_size: Number of queries to process at once
            
        Returns:
            List of (indices, distances) tuples for each query
        """
        start_time = time.time()
        n_queries = len(queries)
        results = []
        
        for i in range(0, n_queries, batch_size):
            end_idx = min(i + batch_size, n_queries)
            batch_queries = queries[i:end_idx]
            
            # Process batch
            for j, query in enumerate(batch_queries):
                indices, distances = self._single_query_vectorized(query, k)
                results.append((indices, distances))
                
                if verbose and (i + j) % 1000 == 0:
                    print(f"Processed {i + j}/{n_queries} queries")
        
        end_time = time.time()
        
        if verbose:
            total_time = end_time - start_time
            qps = n_queries / total_time
            print(f"BruteForce Batch Query: {n_queries} queries in {total_time:.4f}s "
                  f"({qps:.2f} QPS), k={k}")
        
        return results
    
    def benchmark_queries(self, queries: np.ndarray, k_values: List[int] = [1, 10, 100], 
                         num_queries: int = None) -> dict:
        """
        Benchmark performance với số lượng query tự cài đặt
        
        Args:
            queries: Tất cả query vectors
            k_values: Các giá trị k khác nhau để test
            num_queries: Số query sử dụng (nếu None dùng tất cả)
            
        Returns:
            Dictionary chứa kết quả benchmark
        """
        if num_queries is None:
            num_queries = len(queries)
        
        test_queries = queries[:num_queries]
        benchmark_results = {}
        
        print(f"=== BruteForce Benchmark: {num_queries} queries ===")
        
        for k in k_values:
            start_time = time.time()
            
            # Thực hiện queries
            results = self.batch_query(test_queries, k=k, verbose=False)
            
            end_time = time.time()
            total_time = end_time - start_time
            qps = num_queries / total_time
            
            benchmark_results[k] = {
                'total_time': total_time,
                'qps': qps,
                'num_queries': num_queries,
                'k': k
            }
            
            print(f"k={k}: {total_time:.4f}s ({qps:.2f} QPS)")
        
        return benchmark_results
    
    def evaluate_accuracy(self, queries: np.ndarray, ground_truth: List[Tuple[List[int], List[float]]], 
                         k: int = 10) -> dict:
        """
        Đánh giá độ chính xác so với ground truth
        
        Args:
            queries: Query vectors
            ground_truth: Kết quả ground truth (từ cùng method)
            k: Number of neighbors to consider
            
        Returns:
            Dictionary chứa các metrics đánh giá
        """
        results = self.batch_query(queries, k=k, verbose=False)
        
        total_recall = 0.0
        total_precision = 0.0
        
        for i, (result_indices, truth_indices) in enumerate(zip(
            [r[0] for r in results], 
            [gt[0] for gt in ground_truth]
        )):
            result_set = set(result_indices)
            truth_set = set(truth_indices[:k])
            
            # Tính recall và precision
            intersection = len(result_set.intersection(truth_set))
            recall = intersection / len(truth_set) if truth_set else 0.0
            precision = intersection / len(result_set) if result_set else 0.0
            
            total_recall += recall
            total_precision += precision
        
        avg_recall = total_recall / len(results)
        avg_precision = total_precision / len(results)
        
        return {
            'recall@k': avg_recall,
            'precision@k': avg_precision,
            'num_queries': len(queries),
            'k': k
        }


# TEST CASE CHO 1M VECTORS
def test_large_scale_1M():
    """
    Test case với 1M vectors dimension 512, 100 queries, top 100
    """
    print("=" * 70)
    print("LARGE SCALE TEST: 1M vectors, dim=512, 100 queries, top-100")
    print("=" * 70)
    
    from dataset import generate_dataset, generate_queries
    
    # Parameters
    n_vectors = 1000000  # 1M vectors
    dimension = 512      # High dimension
    n_queries = 100     # 100 queries  
    k = 100             # Top-100
    seed = 42           # For reproducible results
    
    print(f"Generating {n_vectors:,} vectors of dimension {dimension}...")
    start_time = time.time()
    
    # Generate dataset và queries
    dataset = generate_dataset(num_data=n_vectors, dim=dimension, seed=seed)
    queries = generate_queries(num_queries=n_queries, dim=dimension, seed=seed)
    
    gen_time = time.time() - start_time
    print(f"✓ Dataset generated in {gen_time:.2f}s")
    print(f"✓ Dataset shape: {dataset.shape}, size: {dataset.nbytes / (1024**3):.2f} GB")
    print(f"✓ Queries shape: {queries.shape}")
    
    # Test Euclidean
    print("\n--- Testing Euclidean Distance ---")
    bf_euclidean = BruteForceSearch(metric='euclidean')
    
    fit_start = time.time()
    bf_euclidean.fit(dataset)
    fit_time = time.time() - fit_start
    print(f"✓ Data fitted in {fit_time:.2f}s")
    
    print("Running Euclidean queries...")
    euclidean_start = time.time()
    results_euclidean = bf_euclidean.batch_query(queries, k=k, verbose=True, batch_size=20)
    euclidean_time = time.time() - euclidean_start
    
    # Test Cosine
    print("\n--- Testing Cosine Distance ---")
    bf_cosine = BruteForceSearch(metric='cosine')
    
    bf_cosine.fit(dataset)
    print("Running Cosine queries...")
    cosine_start = time.time()
    results_cosine = bf_cosine.batch_query(queries, k=k, verbose=True, batch_size=20)
    cosine_time = time.time() - cosine_start
    
    # Results comparison
    print(f"\n" + "=" * 50)
    print("PERFORMANCE COMPARISON")
    print("=" * 50)
    print(f"Euclidean - Total time: {euclidean_time:.2f}s, QPS: {n_queries/euclidean_time:.2f}")
    print(f"Cosine    - Total time: {cosine_time:.2f}s, QPS: {n_queries/cosine_time:.2f}")
    print(f"Ratio: Euclidean/Cosine = {euclidean_time/cosine_time:.2f}x")
    
    # Verify results
    if results_euclidean and results_cosine:
        print(f"\nResults verification:")
        print(f"Euclidean - First query: {len(results_euclidean[0][0])} neighbors")
        print(f"Cosine    - First query: {len(results_cosine[0][0])} neighbors")
        
        # Check if distances make sense
        euclidean_dists = results_euclidean[0][1]
        cosine_dists = results_cosine[0][1]
        print(f"Euclidean distance range: [{euclidean_dists[0]:.4f} ... {euclidean_dists[-1]:.4f}]")
        print(f"Cosine distance range: [{cosine_dists[0]:.4f} ... {cosine_dists[-1]:.4f}]")
    
    print("\n✅ Large scale test completed successfully!")
    return results_euclidean, results_cosine

if __name__ == "__main__":
    print("BruteForceSearch Optimized Version")
    print("With improved Euclidean distance calculation")
    
    try:
        results_euclidean, results_cosine = test_large_scale_1M()
        print("\n🎉 Benchmark completed successfully!")
        
    except MemoryError:
        print("\n💥 MEMORY ERROR: Not enough RAM for 1M vectors")
        print("Try reducing n_vectors to 100000")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()