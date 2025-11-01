import numpy as np
import time
import sys
import os
import psutil

# Thêm đường dẫn đến hnsw-group7/src
current_dir = os.path.dirname(os.path.abspath(__file__))
hnsw_group7_path = os.path.join(current_dir, "hnsw-group7", "src")
sys.path.insert(0, hnsw_group7_path)

try:
    from hnsw import HNSWIndex
except ImportError as e:
    print(f"❌ Lỗi import HNSWIndex: {e}")
    sys.exit(1)

INDEX_PATH = os.path.join(current_dir, "hnsw_index.bin")
DATASET_PATH = os.path.join(current_dir, "dataset.npy")
DIM = 128
NUM_VECTORS = 1_000_000
CHUNK_SIZE = 100_000

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)

def generate_dataset():
    """Sinh dữ liệu 1M vectors và lưu lại để tái sử dụng"""
    if os.path.exists(DATASET_PATH):
        print(f"📦 Loading existing dataset from {DATASET_PATH}")
        return np.load(DATASET_PATH)
    
    print("🔄 Generating new dataset (1M × 128)...")
    np.random.seed(42)
    dataset_chunks = []
    for i in range(0, NUM_VECTORS, CHUNK_SIZE):
        end_idx = min(i + CHUNK_SIZE, NUM_VECTORS)
        chunk = np.random.randn(end_idx - i, DIM).astype(np.float32)
        chunk /= np.linalg.norm(chunk, axis=1, keepdims=True)
        dataset_chunks.append(chunk)
        print(f"  ✅ Generated chunk {i//CHUNK_SIZE + 1}")
    dataset = np.vstack(dataset_chunks)
    np.save(DATASET_PATH, dataset)
    print(f"✅ Dataset saved to {DATASET_PATH}")
    return dataset

def build_hnsw_index(dataset):
    """Build index 1 lần và lưu ra file"""
    hnsw = HNSWIndex(dim=DIM, space='cosine')
    print("🔧 Initializing HNSW index...")
    hnsw.init_index(max_elements=NUM_VECTORS, M=12, ef_construction=64, random_seed=42)

    print("➕ Adding vectors to index...")
    start = time.time()
    for i in range(0, NUM_VECTORS, CHUNK_SIZE):
        end_idx = min(i + CHUNK_SIZE, NUM_VECTORS)
        chunk = dataset[i:end_idx]
        ids = np.arange(i, end_idx)
        hnsw.add_items(chunk, ids=ids, num_threads=8)
        print(f"  ✅ Added {end_idx:,}/{NUM_VECTORS:,}")
    build_time = time.time() - start
    hnsw.set_query_params(ef=100)
    hnsw.save_index(INDEX_PATH)
    print(f"💾 Index saved to {INDEX_PATH}")
    print(f"⏱️ Build time: {build_time:.2f}s ({build_time/60:.1f} min)")
    return hnsw

def load_hnsw_index():
    """Load index từ file"""
    hnsw = HNSWIndex(dim=DIM, space='cosine')
    hnsw.load_index(INDEX_PATH, max_elements=NUM_VECTORS)
    hnsw.set_query_params(ef=100)
    print(f"✅ Loaded existing index from {INDEX_PATH}")
    return hnsw

def run_queries(hnsw, dataset, num_queries=1000, k=10):
    """Chạy query test"""
    print(f"\n🔍 Running {num_queries} queries (top-{k})...")
    np.random.seed(123)
    queries = np.random.randn(num_queries, DIM).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    start = time.time()
    labels, distances = hnsw.knn_query(queries, k=k)
    elapsed = time.time() - start

    print(f"⏱️  Total query time: {elapsed:.4f}s")
    print(f"📈  QPS: {num_queries / elapsed:.2f}")
    print(f"⏱️  Avg per query: {elapsed / num_queries:.6f}s")
    print("\n📊 Sample result (first query):")
    for i, (idx, dist) in enumerate(zip(labels[0][:5], distances[0][:5])):
        print(f"  {i+1:2d}. idx={idx:7d}, dist={dist:.6f}")

def main():
    print("="*70)
    print("🚀 HNSW TEST MODE")
    print("1️⃣ Build new index and save")
    print("2️⃣ Load existing index and test queries")
    print("3️⃣ Quick memory check")
    print("="*70)
    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "3":
        avail = psutil.virtual_memory().available / (1024**3)
        print(f"💾 Available memory: {avail:.2f} GB")
        return

    dataset = generate_dataset()

    if choice == "1":
        build_hnsw_index(dataset)
    elif choice == "2":
        if not os.path.exists(INDEX_PATH):
            print("⚠️ No existing index found! Please build it first.")
            return
        hnsw = load_hnsw_index()
        while True:
            try:
                n = int(input("\nEnter number of queries (e.g., 10 / 100 / 1000): ").strip())
                run_queries(hnsw, dataset, num_queries=n)
            except KeyboardInterrupt:
                print("\n🛑 Stopped.")
                break
            except ValueError:
                print("❌ Invalid input.")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
