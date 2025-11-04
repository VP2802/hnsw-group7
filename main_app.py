import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

from brute_force import BruteForceSearch
from hnsw import HNSWIndex
from dataset import generate_dataset, generate_queries
from evaluation import recall_at_k, analyze_hnsw_structure, visualize_results
from visualization import demo_greedy_search_animation  # chính xác theo file bạn gửi

def main():
   parser = argparse.ArgumentParser(description="Compare Brute-force vs HNSW performance")
   parser.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "cosine"])
   parser.add_argument("--n_vectors", type=int, default=100000)   # nhỏ để demo
   parser.add_argument("--n_queries", type=int, default=5)
   parser.add_argument("--dim", type=int, default=128)
   parser.add_argument("--k", type=int, default=5)
   parser.add_argument("--seed", type=int, default=123)
   parser.add_argument("--ef", type=int, default=200)
   parser.add_argument("--M", type=int, default=8)
   parser.add_argument("--show", action="store_true")
   parser.add_argument("--demo_greedy", action="store_true", help="Run greedy search animation demo")

   # Fix khi chạy notebook / Colab
   if "__file__" not in globals():
       sys.argv = ["main_app.py"]

   args = parser.parse_args()
   np.random.seed(args.seed)

   # --- Step 0: Generate dataset and queries ---
   print(f"\n📊 Generating dataset: {args.n_vectors} vectors, dim={args.dim}")
   data = generate_dataset(args.n_vectors, args.dim, args.seed)
   queries = generate_queries(args.n_queries, args.dim, args.seed)

   # --- Step 1: Brute-force search ---
   brute = BruteForceSearch(metric=args.metric)
   brute.fit(data)
   print("\n⚙️ Running Brute-force search ...")
   start_b = time.time()
   brute_results = brute.batch_query(queries, k=args.k)
   t_b = time.time() - start_b
   print(f"✅ Brute-force finished in {t_b:.3f}s")

   # --- Step 2: HNSW indexing and search ---
   print("\n⚙️ Building HNSW index ...")
   space = "l2" if args.metric == "euclidean" else "cosine"
   hnsw = HNSWIndex(dim=args.dim, space=space)
   hnsw.init_index(max_elements=args.n_vectors, M=args.M, ef_construction=args.ef)
   hnsw.add_items(data)
   hnsw.set_query_params(ef=args.ef)

   print("\n⚙️ Running HNSW search ...")
   start_h = time.time()
   hnsw_labels, _ = hnsw.knn_query(queries, k=args.k)
   t_h = time.time() - start_h
   print(f"✅ HNSW finished in {t_h:.3f}s")

   # --- Step 3: Recall evaluation ---
   brute_indices = [indices for indices, _ in brute_results]
   recall = recall_at_k(brute_indices, hnsw_labels, k=args.k)
   print(f"\n📈 Recall@{args.k}: {recall:.4f}")

   # --- Step 4: HNSW structure analysis ---
   print("\n🔍 Analyzing HNSW graph structure ...")
   stats = analyze_hnsw_structure(hnsw)
   print(f"📊 Levels: {stats['levels']}")
   print(f"📊 Avg degree: {stats['avg_degree']:.2f}")
   print(f"📊 Num edges: {stats['edges']:,}")

   # --- Step 5: Visualization ---
   visualize_results(t_b, t_h, recall, stats, hnsw, data, args.show)

   # --- Step 6: Optional greedy search animation demo ---
   if args.demo_greedy:
       print("\n🎨 Running Greedy Search Animation Demo ...")
       demo_greedy_search_animation(data, n_points=20, pause_time=0.5)

   print("\n✅ Evaluation completed successfully!\n")

if __name__ == "__main__":
   main()
