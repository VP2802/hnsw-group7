"""
visualization.py

Chức năng:
 - project_data(data, n_components=2): giảm chiều dữ liệu (PCA nếu >2 chiều)
 - build_hnsw_graph(hnsw_index, data, k=None, sample=None): xây dựng đồ thị HNSW xấp xỉ bằng cách truy vấn láng giềng
 - plot_graph(G, pos, highlight_path=None, query_point=None): vẽ đồ thị HNSW
 - simulate_greedy(hnsw_index, data, query, start_node=None): mô phỏng tìm kiếm greedy trong HNSW
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional, Tuple, List
from sklearn.decomposition import PCA


# ==============================
# 1️⃣ GIẢM CHIỀU DỮ LIỆU (PCA)
# ==============================
def project_data(data: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Giảm chiều dữ liệu xuống 2D hoặc 3D để dễ trực quan hóa bằng PCA.
    Nếu dữ liệu ban đầu đã ≤ 2 chiều thì giữ nguyên.
    """
    data = np.asarray(data, dtype=np.float32)
    if data.shape[1] <= n_components:
        return data.copy()
    pca = PCA(n_components=n_components)
    proj = pca.fit_transform(data)
    return proj


# ==========================================
# 2️⃣ XÂY DỰNG ĐỒ THỊ HNSW TỪ INDEX CÓ SẴN
# ==========================================
def build_hnsw_graph(
    hnsw_index,
    data: np.ndarray,
    k: Optional[int] = None,
    sample: Optional[int] = 1000
) -> Tuple[nx.Graph, np.ndarray]:
    """
    Tạo đồ thị vô hướng xấp xỉ HNSW bằng cách truy vấn k láng giềng gần nhất
    cho mỗi node trong tập dữ liệu hoặc mẫu con của nó.
    Trả về:
      - G: đối tượng networkx.Graph
      - pos: vị trí 2D (dict[node] = (x, y))
    """

    # Lấy tổng số phần tử trong index
    n_total = getattr(hnsw_index, "get_current_count", lambda: None)()
    if n_total is None:
        n_total = getattr(hnsw_index, "data", None).shape[0] if hasattr(hnsw_index, "data") else None
    if n_total is None:
        raise ValueError("Không xác định được số lượng phần tử trong HNSW index.")

    # Mặc định k = M (số liên kết mỗi node)
    if k is None:
        try:
            k = int(hnsw_index.get_index_params().get("M", 16))
        except Exception:
            k = 16

    # Nếu dữ liệu lớn, chọn mẫu ngẫu nhiên để vẽ
    if sample is None or sample >= n_total:
        indices = np.arange(n_total)
    else:
        rng = np.random.default_rng(123)
        indices = rng.choice(n_total, size=sample, replace=False)

    G = nx.Graph()
    for idx in indices:
        G.add_node(int(idx))

    # Truy vấn theo batch để lấy láng giềng
    B = 1024
    data_ref = getattr(hnsw_index, "data", None)
    for i in range(0, len(indices), B):
        sub = indices[i:i+B]
        pts = np.asarray([data_ref[j] for j in sub], dtype=np.float32)
        labels, _ = hnsw_index.knn_query(pts, k=k+1)
        for local_i, node in enumerate(sub):
            neighs = [int(x) for x in labels[local_i] if int(x) != int(node)]
            for nb in neighs:
                G.add_edge(int(node), int(nb))

    # Tính vị trí chiếu 2D cho các node
    if data_ref is None:
        pos = {n: (0.0, 0.0) for n in G.nodes()}
        proj = None
    else:
        proj_all = project_data(data_ref, n_components=2)
        pos = {i: (float(proj_all[i, 0]), float(proj_all[i, 1])) for i in range(proj_all.shape[0])}
        proj = proj_all
    return G, pos


# ============================
# 3️⃣ HÀM VẼ ĐỒ THỊ HNSW
# ============================
def plot_graph(
    G: nx.Graph,
    pos: dict,
    highlight_path: Optional[List[int]] = None,
    query_point: Optional[Tuple[float, float]] = None,
    figsize=(9, 7),
    node_size=30
):
    """
    Vẽ đồ thị HNSW bằng networkx + matplotlib.
    - highlight_path: danh sách node trên đường tìm kiếm (vẽ đỏ)
    - query_point: điểm truy vấn (vẽ hình sao vàng)
    """
    plt.figure(figsize=figsize)
    nx.draw_networkx_edges(G, pos, alpha=0.25, width=0.8)
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color="skyblue")

    # Nếu có đường đi được highlight
    if highlight_path:
        path_edges = [(highlight_path[i], highlight_path[i+1]) for i in range(len(highlight_path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=2.0)
        nx.draw_networkx_nodes(G, pos, nodelist=highlight_path, node_color="red", node_size=node_size*2)

    # Nếu có điểm truy vấn
    if query_point is not None:
        plt.scatter([query_point[0]], [query_point[1]], c="gold", s=150, marker="*", edgecolor="k", zorder=10)

    plt.axis("equal")
    plt.title("HNSW Graph (xấp xỉ, 2D projection)")
    plt.show()


# ====================================
# 4️⃣ MÔ PHỎNG QUÁ TRÌNH GREEDY SEARCH
# ====================================
def simulate_greedy(
    hnsw_index,
    data: np.ndarray,
    query: np.ndarray,
    start_node: Optional[int] = None,
    max_steps: int = 200,
    metric: str = "euclidean"
) -> List[int]:
    """
    Mô phỏng thuật toán tìm kiếm greedy trên đồ thị HNSW:
    - Bắt đầu từ một node entry (hoặc node 0)
    - Lặp lại: di chuyển đến láng giềng gần query nhất cho đến khi không cải thiện
    Trả về danh sách node theo thứ tự ghé thăm.
    """
    data = np.asarray(data, dtype=np.float32)
    n = data.shape[0]

    # Lấy node khởi đầu
    if start_node is None:
        try:
            start_node = int(getattr(hnsw_index, "entry_point", 0))
        except Exception:
            start_node = 0

    # Lấy danh sách láng giềng
    k = int(hnsw_index.get_index_params().get("M", 16)) if hasattr(hnsw_index, "get_index_params") else 16
    labels, _ = hnsw_index.knn_query(data, k=k+1)
    neighs = {i: [int(x) for x in labels[i] if int(x) != i] for i in range(labels.shape[0])}

    # Định nghĩa hàm đo khoảng cách
    def dist(a, b):
        if metric == "euclidean":
            return float(np.linalg.norm(a - b))
        else:  # cosine distance
            an = np.linalg.norm(a) or 1e-10
            bn = np.linalg.norm(b) or 1e-10
            return float(1.0 - (float(np.dot(a, b)) / (an * bn)))

    path = [int(start_node)]
    cur = int(start_node)
    cur_dist = dist(data[cur], query)

    # Lặp cho đến khi không cải thiện được nữa
    for step in range(max_steps):
        best = cur
        best_d = cur_dist
        for nb in neighs.get(cur, []):
            dnb = dist(data[nb], query)
            if dnb < best_d:
                best = nb
                best_d = dnb
        if best == cur:
            break  # dừng khi không tiến gần hơn
        cur = best
        cur_dist = best_d
        path.append(cur)

    return path
# =====================================
# 5️⃣ ANIMATION GREEDY SEARCH (2D)
# =====================================
def demo_greedy_search_animation(data: np.ndarray, n_points: int = 20, pause_time: float = 0.5,
                          use_hnsw_graph: bool = False, hnsw_index = None):
    data_small = data[:n_points]
    proj = project_data(data_small, n_components=2)

    if use_hnsw_graph:
        if hnsw_index is None:
            raise ValueError("Cần hnsw_index nếu use_hnsw_graph=True")
        G, _ = build_hnsw_graph(hnsw_index, data_small, sample=n_points)
    else:
        G = nx.random_geometric_graph(n_points, radius=0.4)

    pos = {i: (proj[i,0], proj[i,1]) for i in range(n_points)}

    start, goal = np.random.randint(0, n_points, 2)
    print(f"Start: {start}, Goal: {goal}")

    current = start
    visited = [current]

    plt.figure(figsize=(6,6))
    while current != goal:
        neighbors = list(G.neighbors(current))
        if len(neighbors) == 0:
            print("No neighbors to move, stopping.")
            break
        candidates = [n for n in neighbors if n not in visited]
        if len(candidates) == 0:
            print("No unvisited neighbors closer, stopping.")
            break
        dists = [np.linalg.norm(proj[goal]-proj[n]) for n in candidates]
        next_node = candidates[np.argmin(dists)]
        visited.append(next_node)

        plt.cla()
        nx.draw_networkx_edges(G, pos, alpha=0.25, width=0.8)
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color="skyblue")
        if len(visited) > 1:
            path_edges = list(zip(visited[:-1], visited[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=2.0, arrowstyle='-|>', arrowsize=12)
        nx.draw_networkx_nodes(G, pos, nodelist=[current], node_color="orange", node_size=120)
        nx.draw_networkx_nodes(G, pos, nodelist=[goal], node_color="green", node_size=120)

        dist_to_goal = np.linalg.norm(proj[goal]-proj[next_node])
        plt.title(f"Greedy Search Demo\nDistance to goal: {dist_to_goal:.3f}")
        plt.axis("equal")
        plt.pause(pause_time)

        current = next_node

    print("Reached goal!")
    plt.show()
