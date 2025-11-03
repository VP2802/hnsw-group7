import numpy as np
import os
import sys

# Import HNSW
current_dir = '/content'
sys.path.insert(0, os.path.join(current_dir, "hnsw-group7", "src"))
from hnsw import HNSWIndex

class SimpleHNSW:
    def __init__(self, dim=128):
        self.dim = dim
        # Dùng thư mục trong Colab (sẽ mất khi runtime reset)
        self.data_dir = "/content/my_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.dataset_path = f"{self.data_dir}/dataset.npy"
        self.index_path = f"{self.data_dir}/hnsw_index.bin"
        self.index = None
        
    def generate_dataset(self, num_vectors, dim, seed=42):
        """Tạo dataset ngẫu nhiên (thay thế cho dataset.py)"""
        np.random.seed(seed)
        return np.random.random((num_vectors, dim)).astype(np.float32)
        
    def generate_queries(self, num_queries, dim, seed=123):
        """Tạo queries ngẫu nhiên (thay thế cho dataset.py)"""
        np.random.seed(seed)
        return np.random.random((num_queries, dim)).astype(np.float32)
        
    def build_index(self, num_vectors, M=16, ef_construction=200):
        """Build index với N vectors"""
        print(f"🔄 Building index với {num_vectors:,} vectors...")
        
        # Tạo dataset
        dataset = self.generate_dataset(num_vectors, self.dim, seed=42)
        dataset = dataset.astype(np.float32)
        dataset /= np.linalg.norm(dataset, axis=1, keepdims=True)
        np.save(self.dataset_path, dataset)
        
        # Build index
        self.index = HNSWIndex(dim=self.dim, space='cosine')
        self.index.init_index(max_elements=num_vectors, M=M, 
                            ef_construction=ef_construction, random_seed=42)
        
        # Add vectors
        chunk_size = 50000
        for i in range(0, num_vectors, chunk_size):
            end_idx = min(i + chunk_size, num_vectors)
            chunk = dataset[i:end_idx]
            ids = np.arange(i, end_idx)
            self.index.add_items(chunk, ids=ids)
            print(f"  ✅ Added {end_idx:,}/{num_vectors:,}")
        
        self.index.set_query_params(ef=100)
        self.index.save_index(self.index_path)
        print(f"✅ Index saved: {self.index_path}")
        return self
        
    def load_index(self, num_vectors):
        """Load index có sẵn"""
        if not os.path.exists(self.index_path):
            print("❌ Index chưa tồn tại! Hãy build trước.")
            return None
        
        #Kiểm tra số vectors có hợp lệ không
        dataset = np.load(self.dataset_path)
        actual_vectors = len(dataset)
        if num_vectors > actual_vectors:
            print(f"❌ Lỗi: Yêu cầu {num_vectors:,} vectors nhưng index có sẵn chỉ có {actual_vectors:,} vectors!")
            return None
            
        self.index = HNSWIndex(dim=self.dim, space='cosine')
        self.index.load_index(self.index_path, max_elements=num_vectors)
        self.index.set_query_params(ef=100)
        print("✅ Index loaded!")
        return self
        
    def search(self, num_queries, k=10):
        """Search với N queries và K kết quả"""
        if self.index is None:
            print("❌ Chưa load index!")
            return
        
        # Kiểm tra k không vượt quá số vectors
        dataset = np.load(self.dataset_path)
        if k > len(dataset):
            print(f"❌ Lỗi: k={k} vượt quá số vectors trong index ({len(dataset)})!")
            k = len(dataset)
            print(f"⚠️ Đã tự động điều chỉnh k thành {k}")
            
        print(f"🔍 Searching {num_queries} queries, top-{k}...")
        
        # Tạo queries
        queries = self.generate_queries(num_queries, self.dim, seed=123)
        queries = queries.astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        
        # Search
        import time
        start = time.time()
        labels, distances = self.index.knn_query(queries, k=k)
        elapsed = time.time() - start
        
        # Kết quả
        print(f"⏱️ Time: {elapsed:.2f}s | QPS: {num_queries/elapsed:.1f}")
        
        # Menu in kết quả
        while True:
            print(f"\n🎯 Chọn cách in kết quả:")
            print("1. In tất cả queries")
            print("2. In N queries đầu tiên")
            print("3. In query tại vị trí cụ thể")
            print("4. Thoát")
            
            choice = input("Chọn (1/2/3/4): ").strip()
            
            if choice == "4":
                break
                
            elif choice == "1":
                # In tất cả queries
                print(f"\n📊 In tất cả {num_queries} queries:")
                for query_idx in range(num_queries):
                    print(f"Query {query_idx + 1}:")
                    for i, (idx, dist) in enumerate(zip(labels[query_idx][:k], distances[query_idx][:k])):
                        print(f"  {i+1:2d}. idx={idx:6d}, dist={dist:.4f}")
                    print()
                    
            elif choice == "2":
                # In N queries đầu
                try:
                    n = int(input(f"Nhập số queries đầu muốn in (1-{num_queries}): ").strip())
                    n = max(1, min(n, num_queries))
                    
                    print(f"\n📊 In {n} queries đầu tiên:")
                    for query_idx in range(n):
                        print(f"Query {query_idx + 1}:")
                        for i, (idx, dist) in enumerate(zip(labels[query_idx][:k], distances[query_idx][:k])):
                            print(f"  {i+1:2d}. idx={idx:6d}, dist={dist:.4f}")
                        print()
                        
                except ValueError:
                    print("❌ Số không hợp lệ!")
                    
            elif choice == "3":
                # In query tại vị trí cụ thể
                try:
                    pos = int(input(f"Nhập vị trí query (1-{num_queries}): ").strip())
                    if 1 <= pos <= num_queries:
                        query_idx = pos - 1
                        print(f"\n📊 Query tại vị trí {pos}:")
                        for i, (idx, dist) in enumerate(zip(labels[query_idx][:k], distances[query_idx][:k])):
                            print(f"  {i+1:2d}. idx={idx:6d}, dist={dist:.4f}")
                    else:
                        print(f"❌ Vị trí phải từ 1 đến {num_queries}!")
                        
                except ValueError:
                    print("❌ Số không hợp lệ!")
                    
            else:
                print("❌ Lựa chọn không hợp lệ!")
        
        return labels, distances

def main():
    print("🚀 Simple HNSW Manager trên Colab")
    print("1. Build index mới")
    print("2. Load index có sẵn")
    print("3. Thoát")
    
    choice = input("Chọn (1/2/3): ").strip()
    
    if choice == "3":
        return
        
    hnsw = SimpleHNSW(dim=128)
    
    if choice == "1":
        num_vectors = int(input("Nhập số vectors (N): ").strip() or "10000")
        hnsw.build_index(num_vectors)
    elif choice == "2":
        num_vectors = int(input("Nhập số vectors trong index: ").strip() or "10000")
        hnsw.load_index(num_vectors)
    else:
        print("❌ Lựa chọn không hợp lệ!")
        return
        
    # Search loop
    while hnsw.index:
        print("\n🎯 Search Menu")
        print("1. Tìm kiếm")
        print("2. Thoát")
        
        sub_choice = input("Chọn (1/2): ").strip()
        
        if sub_choice == "2":
            break
        elif sub_choice == "1":
            num_queries = int(input("Số queries: ").strip() or "5")
            k = int(input("Số kết quả mỗi query (K): ").strip() or "10")
            hnsw.search(num_queries, k)
        else:
            print("❌ Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()