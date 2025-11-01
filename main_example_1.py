import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
HNSW_GROUP = os.path.join(BASE_DIR, "hnsw-group7")
if os.path.isdir(HNSW_GROUP) and HNSW_GROUP not in sys.path:
    sys.path.insert(0, HNSW_GROUP)

from src.hnsw import HNSWIndex
from dataset.dataset import generate_dataset, generate_queries
import numpy as np

def main():
    # 1️⃣ Cấu hình
    dim = 32
    num_elements = 1000
    num_queries = 3
    k = 5
    seed = 42

    print("=== BẮT ĐẦU DEMO HNSW ===")

    # 2️⃣ Sinh dữ liệu
    print("🔹 Sinh dataset và query ...")
    data = generate_dataset(num_elements, dim, seed)
    queries = generate_queries(num_queries, dim, seed)
    ids = np.arange(num_elements)

    # 3️⃣ Khởi tạo index
    print("🔹 Khởi tạo HNSW index ...")
    index = HNSWIndex(dim=dim, space='l2')
    index.init_index(max_elements=num_elements, M=16, ef_construction=200)

    # 4️⃣ Thêm dữ liệu
    print("🔹 Thêm dữ liệu vào index ...")
    index.add_items(data, ids)
    index.set_query_params(ef=50)

    # 5️⃣ Thực hiện truy vấn k-NN
    print("🔹 Thực hiện truy vấn ...")
    labels, distances = index.knn_query(queries, k=k)

    for i in range(num_queries):
        print(f"\n🟢 Query {i+1}:")
        print("Nearest IDs:", labels[i])
        print("Distances  :", np.round(distances[i], 3))

    # 6️⃣ Thông tin index
    print("\n📊 Thông tin index:")
    print(index.info())

    print("\n=== KẾT THÚC DEMO ===")


if __name__ == "__main__":
    main()