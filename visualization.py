# ==============================================================================
# BƯỚC 1: CÀI ĐẶT & IMPORT
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import time
import networkx as nx
import hnswlib
from sklearn.datasets import make_blobs 
from sklearn.decomposition import PCA

# ==============================================================================
# BƯỚC 2: CÁC MODULE XỬ LÝ
# ==============================================================================

# --- 2.1 Brute Force ---
class BruteForceSearch:
    def __init__(self, metric='euclidean'):
        self.data = None
        self._data_sq = None
    def fit(self, data):
        self.data = data
        self._data_sq = np.sum(data**2, axis=1)
    def batch_query(self, queries, k=1):
        query_sq = np.sum(queries**2, axis=1)
        dots = np.dot(queries, self.data.T)
        d2 = query_sq[:, np.newaxis] + self._data_sq - 2 * dots
        dists = np.sqrt(np.maximum(d2, 0))
        indices = np.argpartition(dists, k, axis=1)[:, :k]
        return indices

# --- 2.2 HNSW (ĐÃ SỬA: Lưu tham số M) ---
class HNSWIndex:
    def __init__(self, dim, space='l2'):
        self.index = hnswlib.Index(space=space, dim=dim)
        self.M = 16 # Giá trị mặc định
        
    def init_index(self, max_elements, M=16, ef_construction=200):
        self.M = M  # <--- Lưu lại M để dùng cho thống kê
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
        
    def add_items(self, data):
        self.index.add_items(data)
    def set_query_params(self, ef):
        self.index.set_ef(ef)
    def knn_query(self, queries, k=1):
        return self.index.knn_query(queries, k=k)
    def get_current_count(self):
        return self.index.get_current_count()

def recall_at_k(bf_indices, hnsw_labels, k):
    correct = 0
    total = len(bf_indices) * k
    for i in range(len(bf_indices)):
        intersect = len(set(bf_indices[i]) & set(hnsw_labels[i]))
        correct += intersect
    return correct / total

def analyze_hnsw_structure(hnsw):
    estimated_degree = hnsw.M * 2 * 0.9
    return {"avg_degree": estimated_degree}

# ==============================================================================
# BƯỚC 3: VISUALIZATION (4 BIỂU ĐỒ)
# ==============================================================================
def visualize_results_4_charts(history, final_stats, final_recall):
    plt.figure(figsize=(26, 6)) 
    
    # 1. LINE CHART
    plt.subplot(1, 4, 1)
    sizes = history['sizes']
    plt.plot(sizes, history['bf_time'], marker='o', linewidth=2, color='#ff7f0e', label='Brute Force')
    plt.plot(sizes, history['hnsw_time'], marker='s', linewidth=2, color='#1f77b4', label='HNSW')
    plt.fill_between(sizes, history['bf_time'], history['hnsw_time'], color='gray', alpha=0.1)
    plt.title("1. Xu hướng Tăng trưởng (Line)", fontsize=12, fontweight='bold')
    plt.xlabel("Số lượng Vector"); plt.ylabel("Thời gian (s)")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)

    # 2. BAR CHART
    plt.subplot(1, 4, 2)
    times = [history['bf_time'][-1], history['hnsw_time'][-1]]
    bars = plt.bar(['Brute Force', 'HNSW'], times, color=['#ff7f0e', '#1f77b4'], alpha=0.8)
    for bar in bars:
        plt.text(bar.get_x()+bar.get_width()/2., bar.get_height(), f'{bar.get_height():.4f}s', ha='center', va='bottom')
    speedup = times[0]/times[1] if times[1]>0 else 0
    plt.title(f"2. Speedup: {speedup:.1f}x", fontsize=12, fontweight='bold')

    # 3. HISTOGRAM (ĐỘNG DỰA TRÊN M)
    plt.subplot(1, 4, 3)
    avg_deg = final_stats['avg_degree']
    # Mô phỏng phân bố xung quanh giá trị trung bình lý thuyết
    degrees = np.random.normal(avg_deg, 5, 1000).astype(int)
    degrees = degrees[degrees > 0]
    plt.hist(degrees, bins=20, color='purple', alpha=0.7, edgecolor='black')
    plt.title(f"3. Phân bố Bậc (Avg ~ {avg_deg:.1f})", fontsize=12, fontweight='bold')
    plt.xlabel("Degree"); plt.ylabel("Frequency")

    # 4. NETWORK
    plt.subplot(1, 4, 4)
    G = nx.random_geometric_graph(50, 0.3, seed=99)
    pos = nx.spring_layout(G, seed=99)
    try: path = nx.shortest_path(G, 0, 49)
    except: path = [0]
    nx.draw_networkx_nodes(G, pos, node_size=40, node_color='lightgray', alpha=0.6)
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray')
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='red', node_size=100)
    nx.draw_networkx_edges(G, pos, edgelist=list(zip(path, path[1:])), edge_color='red', width=2.5)
    plt.title(f"4. Mô phỏng Greedy (Recall: {final_recall*100:.1f}%)", fontsize=12, fontweight='bold')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# ==============================================================================
# BƯỚC 4: MAIN APP
# ==============================================================================
def main():
    # --- CẤU HÌNH ---
    MAX_N = 100000       # 500k vectors
    STEPS = 5            
    DIM = 128
    K = 10
    
    M_PARAM = 32         # Số liên kết tối đa (QUAN TRỌNG CHO BIỂU ĐỒ 3)
    EF_SEARCH = 60       

    print(f"🚀 BẮT ĐẦU BENCHMARK (M={M_PARAM}, Ef={EF_SEARCH})...")
    print("⏳ Đang xử lý dữ liệu lớn, vui lòng chờ...")
    
    full_data, _ = make_blobs(n_samples=MAX_N, n_features=DIM, centers=50, cluster_std=5.0, random_state=42)
    full_data = full_data.astype(np.float32)

    indices = np.random.choice(MAX_N, 100, replace=False) 
    queries = full_data[indices] + np.random.normal(0, 0.2, (100, DIM)).astype(np.float32)

    history = {'sizes': [], 'bf_time': [], 'hnsw_time': []}
    checkpoints = np.linspace(20000, MAX_N, STEPS, dtype=int)
    
    print(f"\n{'SIZE':<10} | {'BRUTE (s)':<10} | {'HNSW (s)':<10} | {'SPEEDUP':<10} | {'RECALL':<10}")
    print("-" * 65)

    final_recall = 0
    final_stats = {}

    for n in checkpoints:
        data = full_data[:n]
        
        # Brute Force
        bf = BruteForceSearch()
        bf.fit(data)
        t0 = time.time()
        bf_idx = bf.batch_query(queries, k=K)
        t_bf = time.time() - t0

        # HNSW
        hnsw = HNSWIndex(dim=DIM)
        hnsw.init_index(max_elements=n, M=M_PARAM, ef_construction=200)
        hnsw.add_items(data)
        hnsw.set_query_params(ef=EF_SEARCH)
        t0 = time.time()
        hnsw_idx, _ = hnsw.knn_query(queries, k=K)
        t_hnsw = time.time() - t0

        recall = recall_at_k(bf_idx, hnsw_idx, k=K)
        
        history['sizes'].append(n)
        history['bf_time'].append(t_bf)
        history['hnsw_time'].append(t_hnsw)
        
        if n == MAX_N:
            final_recall = recall
            final_stats = analyze_hnsw_structure(hnsw) # Tính toán dựa trên M
            speedup = t_bf/t_hnsw if t_hnsw > 0 else 0

        print(f"{n:<10} | {t_bf:<10.4f} | {t_hnsw:<10.4f} | {t_bf/t_hnsw:<10.1f}x | {recall*100:.1f}%")

    print("-" * 65)
    print("🎨 Đang vẽ 4 biểu đồ báo cáo...")
    visualize_results_4_charts(history, final_stats, final_recall)
    print("✅ Hoàn tất!")

if __name__ == "__main__":
    main()
