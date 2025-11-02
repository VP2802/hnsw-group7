import numpy as np
import os
import sys
from dataset import generate_dataset, generate_queries

# Import HNSW
current_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(current_dir, "hnsw-group7", "src"))
from hnsw import HNSWIndex

class SimpleHNSW:
    def __init__(self, dim=128):
        self.dim = dim
        self.data_dir = "my_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.dataset_path = f"{self.data_dir}/dataset.npy"
        self.index_path = f"{self.data_dir}/hnsw_index.bin"
        self.index = None
        
    def build_index(self, num_vectors, M=16, ef_construction=200):
        """Build index với N vectors"""
        print(f"🔄 Building index với {num_vectors:,} vectors...")
        
        # Tạo dataset
        dataset = generate_dataset(num_vectors, self.dim, seed=42)
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
            print(f"⚠️  Đã tự động điều chỉnh k thành {k}")
            
        print(f"🔍 Searching {num_queries} queries, top-{k}...")
        
        # Tạo queries
        queries = generate_queries(num_queries, self.dim, seed=123)
        queries = queries.astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        
        # Search
        import time
        start = time.time()
        labels, distances = self.index.knn_query(queries, k=k)
        elapsed = time.time() - start
        
        # Kết quả
        print(f"⏱️ Time: {elapsed:.2f}s | QPS: {num_queries/elapsed:.1f}")
        print(f"📊 First query results:")
        for i, (idx, dist) in enumerate(zip(labels[0][:k], distances[0][:k])):
            print(f"  {i+1:2d}. idx={idx:6d}, dist={dist:.4f}")
            
        return labels, distances

def main():
    print("🚀 Simple HNSW Manager")
    print("1. Build index mới")
    print("2. Load index có sẵn")
    print("3. Thoát")
    
    choice = input("Chọn (1/2/3): ").strip()
    
    if choice == "3":
        return
        
    hnsw = SimpleHNSW(dim=128)
    
    if choice == "1":
        num_vectors = int(input("Nhập số vectors (N): ").strip() or "100000")
        hnsw.build_index(num_vectors)
    elif choice == "2":
        num_vectors = int(input("Nhập số vectors trong index: ").strip() or "100000")
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
            num_queries = int(input("Số queries: ").strip() or "100")
            k = int(input("Số kết quả mỗi query (K): ").strip() or "10")
            hnsw.search(num_queries, k)
        else:
            print("❌ Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()