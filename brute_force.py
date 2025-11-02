import numpy as np
import heapq
import os
import time
from typing import List, Tuple, Iterator


# ======================================================
# =============== DATASET PREPARATION ==================
# ======================================================
def prepare_dataset(
    num_elements=100_000,
    dim=128,
    num_queries=100,
    seed=42,
    data_dir="data",
    rebuild=False
):
    """
    Tạo hoặc load dataset (cache vào file .npy)
    """
    os.makedirs(data_dir, exist_ok=True)

    data_path = os.path.join(data_dir, f"data_{num_elements}_{dim}.npy")
    query_path = os.path.join(data_dir, f"queries_{num_queries}_{dim}.npy")

    if not rebuild and os.path.exists(data_path) and os.path.exists(query_path):
        print(f"📂 Đang load dataset có sẵn từ '{data_dir}' ...")
        data = np.load(data_path)
        queries = np.load(query_path)
        print(f"✅ Loaded: data={data.shape}, queries={queries.shape}")
        return data, queries

    print("⚙️ Đang sinh dataset mới ...")
    rng = np.random.default_rng(seed)
    data = rng.random((num_elements, dim), dtype=np.float32)
    queries = rng.random((num_queries, dim), dtype=np.float32)

    np.save(data_path, data)
    np.save(query_path, queries)
    print(f"✅ Dataset mới đã được tạo và lưu vào '{data_dir}'")
    print(f"   data shape: {data.shape}, queries shape: {queries.shape}")
    return data, queries


# ======================================================
# =============== BRUTE FORCE SEARCH ===================
# ======================================================
class BruteForceSearch:
    def __init__(self, metric: str = 'euclidean'):
        self.metric = metric
        self.data = None
        self.dimension = None
        self._data_sq = None
        self._data_norms = None

    def fit(self, data: np.ndarray):
        """Cache dữ liệu và precompute norms"""
        self.data = data
        self.dimension = data.shape[1]
        if self.metric == 'euclidean':
            self._data_sq = np.einsum('ij,ij->i', data, data)
        elif self.metric == 'cosine':
            self._data_norms = np.linalg.norm(data, axis=1)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def _single_query_vectorized(self, query: np.ndarray, k: int = 1):
        """Vectorized brute-force search"""
        if self.metric == 'euclidean':
            query_sq = np.dot(query, query)
            dots = np.dot(self.data, query)
            d2 = self._data_sq + query_sq - 2 * dots
            d2 = np.maximum(d2, 0)
            dists = np.sqrt(d2)
        elif self.metric == 'cosine':
            qn = np.linalg.norm(query)
            qn = qn if qn != 0 else 1e-10
            dists = 1 - np.dot(self.data, query) / (self._data_norms * qn)
        else:
            raise ValueError("Unsupported metric")
        idx = np.argpartition(dists, k)[:k]
        idx = idx[np.argsort(dists[idx])]
        return idx.tolist(), dists[idx].tolist()

    def optimized_query(self, query, k=1):
        return self._single_query_vectorized(query, k)

    def batch_query(self, queries: np.ndarray, k=1, verbose=False, batch_size=1000):
        """Chạy nhiều query tuần tự (batch)"""
        results = []
        start = time.perf_counter()
        for i, q in enumerate(queries):
            results.append(self.optimized_query(q, k))
            if verbose and (i + 1) % 1000 == 0:
                print(f"  Đã xử lý {i+1}/{len(queries)} queries...")
        elapsed = time.perf_counter() - start
        if verbose:
            print(f"✅ Batch {len(queries)} queries hoàn tất ({elapsed:.2f}s)")
        return results

    def batch_global_topk(self, queries: np.ndarray, global_k: int = 50,
                          batch_size: int = 256, verbose: bool = False):
        """
        Lấy ra global top-k (nhỏ nhất trên toàn bộ queries)
        """
        heap = []  # (-dist, q_idx, d_idx)
        n_queries = len(queries)
        start_all = time.perf_counter()

        for qi, q in enumerate(queries):
            if self.metric == 'euclidean':
                q_sq = np.dot(q, q)
                dots = np.dot(self.data, q)
                sq_d = self._data_sq + q_sq - 2 * dots
                sq_d = np.maximum(sq_d, 0)
                dists = np.sqrt(sq_d)
            elif self.metric == 'cosine':
                qn = np.linalg.norm(q)
                qn = qn if qn != 0 else 1e-10
                dists = 1 - np.dot(self.data, q) / (self._data_norms * qn)
            else:
                raise ValueError("Unsupported metric")

            for di, dist in enumerate(dists):
                dist = float(dist)
                if len(heap) < global_k:
                    heapq.heappush(heap, (-dist, qi, di))
                elif dist < -heap[0][0]:
                    heapq.heappushpop(heap, (-dist, qi, di))

            if verbose and (qi + 1) % 100 == 0:
                print(f"⚙️  {qi+1}/{n_queries} queries processed...")

        result = [(q, d, -neg) for (neg, q, d) in heap]
        result.sort(key=lambda x: x[2])
        elapsed_all = time.perf_counter() - start_all
        print(f"\n✅ Global top-{global_k} hoàn tất trong {elapsed_all:.2f}s")
        return result


# ======================================================
# ================ HELPER RUNNERS ======================
# ======================================================
def run_batch_bf_search(bf: BruteForceSearch, queries: np.ndarray, k: int,
                        batch_size: int = 100, verbose: bool = False):
    """Chạy batch queries mà không lưu toàn bộ kết quả"""
    total = len(queries)
    start_all = time.perf_counter()
    for i, q in enumerate(queries):
        bf.optimized_query(q, k)
        if verbose and (i + 1) % batch_size == 0:
            print(f"  Đã xử lý {i+1}/{total}")
    elapsed = time.perf_counter() - start_all
    print(f"✅ Finished {total} queries in {elapsed:.3f}s ({total/elapsed:.2f} QPS)")


# ======================================================
# ======================== MAIN ========================
# ======================================================
def main():
    print("🚀 Simple Brute Force Manager")
    print("1. Build dataset mới")
    print("2. Load dataset có sẵn")
    print("3. Thoát")

    choice = input("Chọn (1/2/3): ").strip()
    if choice == "3":
        print("👋 Thoát chương trình.")
        return

    dim = int(input("Nhập số chiều (mặc định 128): ") or "128")
    num_elements = int(input("Nhập số vector (mặc định 100000): ") or "100000")
    num_queries = int(input("Nhập số lượng query (mặc định 100): ") or "100")
    seed = int(input("Nhập seed (mặc định 42): ") or "42")

    if choice == "1":
        data, queries = prepare_dataset(num_elements, dim, num_queries, seed, rebuild=True)
    else:
        data, queries = prepare_dataset(num_elements, dim, num_queries, seed, rebuild=False)

    bf = BruteForceSearch(metric="euclidean")
    bf.fit(data)

    while True:
        print("\n🎯 Search Menu")
        print("1. Tìm kiếm 1 query")
        print("2. Chạy batch queries (benchmark)")
        print("3. Global Top-K trên toàn bộ queries")
        print("4. Thoát")

        sub = input("Chọn (1/2/3/4): ").strip()
        if sub == "4":
            print("👋 Kết thúc chương trình.")
            break

        elif sub == "1":
            k = int(input("Nhập số kết quả gần nhất (k): ") or "10")
            idx = int(input(f"Chọn chỉ số query (0 - {len(queries)-1}): ") or "0")
            query = queries[idx]
            print(f"🔍 Đang tìm {k} NN cho query {idx} ...")
            start = time.perf_counter()
            indices, dists = bf.optimized_query(query, k)
            elapsed = time.perf_counter() - start
            print(f"\nIndices: {indices}\nDistances: {np.round(dists,6)}\n⏱️ {elapsed:.6f}s")

        elif sub == "2":
            num_q = int(input(f"Nhập số query muốn chạy (mặc định {len(queries)}): ") or len(queries))
            k = int(input("Nhập k (mặc định 10): ") or "10")
            verbose = input("In tiến độ? (y/N): ").strip().lower() == 'y'
            run_batch_bf_search(bf, queries[:num_q], k, verbose=verbose)

        elif sub == "3":
            global_k = int(input("Nhập số kết quả top-k toàn cục: ") or "50")
            verbose = input("In tiến độ? (y/N): ").strip().lower() == 'y'
            results = bf.batch_global_topk(queries, global_k, verbose=verbose)
            print("\n📊 Global Top-K nhỏ nhất:")
            for rank, (q_idx, d_idx, dist) in enumerate(results[:global_k], 1):
                print(f"{rank:3d}. Query {q_idx:4d} ↔ Data {d_idx:6d} | dist = {dist:.6f}")

        else:
            print("❌ Lựa chọn không hợp lệ!")


if __name__ == "__main__":
    main()
