import numpy as np
import os
import time
import hnswlib
import json
from article_embedder import ArticleEmbedder

class ArticleHNSWManager:
    def __init__(self, index_dir='article_index'):
        self.index_dir = index_dir
        self.dim = 768
        self.index = None
        self.articles = []
        self.embedder = ArticleEmbedder()
        self.all_embeddings = None
        
        os.makedirs(index_dir, exist_ok=True)
    
    def get_index_info(self):
        if self.index is None:
            return {
                'dim': self.dim,
                'article_count': len(self.articles),
                'vector_count': 0,
                'index_dir': self.index_dir
            }
        
        return {
            'dim': self.dim,
            'article_count': len(self.articles),
            'vector_count': self.index.get_current_count(),
            'index_dir': self.index_dir
        }
    
    def build_index(self, articles, max_elements=10000, ef_construction=200, M=16):
        print("ĐANG XÂY DỰNG INDEX TÌM KIẾM BÀI BÁO")
        print("=" * 50)
        
        valid_articles, embeddings = self.embedder.embed_articles(articles)
        self.articles = valid_articles
        self.all_embeddings = embeddings
        
        if len(embeddings) == 0:
            print("Không có embeddings để xây dựng index!")
            return False
        
        print(f"Dữ liệu: {len(embeddings)} bài báo, {embeddings.shape[1]} chiều")
        
        # Xây dựng HNSW index
        print("Đang xây dựng HNSW index...")
        start_time = time.time()
        
        self.index = hnswlib.Index(space='cosine', dim=embeddings.shape[1])
        self.index.init_index(max_elements=max_elements, 
                            ef_construction=ef_construction, 
                            M=M)
        
        self.index.add_items(embeddings, np.arange(len(embeddings)))
        build_time = time.time() - start_time
        
        print(f"Thời gian xây dựng HNSW: {build_time:.4f}s")
        
        # Lưu metadata và index
        self._save_metadata()
        index_path = os.path.join(self.index_dir, 'article_index.bin')
        self.index.save_index(index_path)
        
        print("XÂY DỰNG INDEX HOÀN TẤT!")
        return True
    
    def _cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _save_metadata(self):
        metadata = {
            'dim': self.dim,
            'total_articles': len(self.articles),
            'articles': self.articles,
            'build_time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'embeddings': self.all_embeddings.tolist() if self.all_embeddings is not None else None
        }
        
        metadata_path = os.path.join(self.index_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def load_index(self):
        print(f"Đang tải index từ {self.index_dir}...")
        
        metadata_path = os.path.join(self.index_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Không tìm thấy metadata: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        self.dim = metadata['dim']
        self.articles = metadata['articles']
        
        if 'embeddings' in metadata and metadata['embeddings'] is not None:
            print("Đang tải embeddings từ cache...")
            self.all_embeddings = np.array(metadata['embeddings'], dtype=np.float32)
            print(f"Đã tải {len(self.all_embeddings)} embeddings")
        else:
            print("Không tìm thấy embeddings cache, cần embed lại...")
            valid_articles, embeddings = self.embedder.embed_articles(self.articles)
            self.all_embeddings = embeddings
        
        index_path = os.path.join(self.index_dir, 'article_index.bin')
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Không tìm thấy file index: {index_path}")
        
        self.index = hnswlib.Index(space='cosine', dim=self.dim)
        self.index.load_index(index_path)
        self.index.set_ef(100)
        
        print(f"Tải thành công: {len(self.articles)} bài báo")
        return True
    
    def search_with_comparison(self, query, k=10):
        if self.index is None or self.all_embeddings is None:
            raise RuntimeError("Hệ thống chưa được khởi tạo!")
        
        print(f"SO SÁNH TÌM KIẾM: '{query}'")
        print("=" * 60)
        
        query_vector = self.embedder.embed_query(query)
        
        # Brute Force
        print("BRUTE FORCE...")
        start_time = time.time()
        
        similarities = []
        for i in range(len(self.all_embeddings)):
            similarity = self._cosine_similarity(query_vector[0], self.all_embeddings[i])
            similarities.append((i, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        brute_results = similarities[:k]
        brute_time = time.time() - start_time
        
        # HNSW
        print("HNSW SEARCH...")
        start_time = time.time()
        
        labels, distances = self.index.knn_query(query_vector, k=k)
        hnsw_time = time.time() - start_time
        
        hnsw_results = []
        for i, (label, distance) in enumerate(zip(labels[0], distances[0])):
            similarity = 1 - distance
            hnsw_results.append((int(label), similarity))
        
        # So sánh
        print(f"KẾT QUẢ SO SÁNH:")
        print(f"  Brute Force: {brute_time:.4f}s")
        print(f"  HNSW: {hnsw_time:.4f}s")
        print(f"  Tốc độ tăng: {brute_time/hnsw_time:.1f}x")
        
        brute_indices = {idx for idx, _ in brute_results}
        hnsw_indices = {idx for idx, _ in hnsw_results}
        common_results = brute_indices & hnsw_indices
        accuracy = len(common_results) / k
        
        print(f"  Độ chính xác Top-{k}: {accuracy:.1%}")
        print(f"  Kết quả trùng: {len(common_results)}/{k}")
        
        return {
            'brute_force': {
                'results': [{'index': idx, 'similarity': sim} for idx, sim in brute_results],
                'time': brute_time
            },
            'hnsw': {
                'results': [{'index': idx, 'similarity': sim} for idx, sim in hnsw_results],
                'time': hnsw_time
            },
            'comparison': {
                'speedup': brute_time / hnsw_time,
                'accuracy': accuracy,
                'common_results': len(common_results)
            }
        }
    
    def benchmark_multiple_queries(self, queries, k=10):
        print(f"BENCHMARK VỚI {len(queries)} QUERIES")
        print("=" * 60)
        
        results = []
        for query in queries:
            comparison = self.search_with_comparison(query, k=k)
            results.append({'query': query, **comparison})
        
        print(f"TỔNG KẾT BENCHMARK:")
        avg_speedup = np.mean([r['comparison']['speedup'] for r in results])
        avg_accuracy = np.mean([r['comparison']['accuracy'] for r in results])
        avg_brute_time = np.mean([r['brute_force']['time'] for r in results])
        avg_hnsw_time = np.mean([r['hnsw']['time'] for r in results])
        
        print(f"  Thời gian Brute Force trung bình: {avg_brute_time:.4f}s")
        print(f"  Thời gian HNSW trung bình: {avg_hnsw_time:.4f}s")
        print(f"  Tốc độ tăng trung bình: {avg_speedup:.1f}x")
        print(f"  Độ chính xác trung bình: {avg_accuracy:.1%}")
        
        return results

def build_article_index():
    from crawl_articles import ArticleCrawler
    
    print("BENCHMARK: HNSW vs BRUTE FORCE")
    print("=" * 50)
    
    crawler = ArticleCrawler()
    articles = crawler.load_articles()
    
    if not articles:
        print("Không có dữ liệu bài báo!")
        return
    
    hnsw_mgr = ArticleHNSWManager()
    success = hnsw_mgr.build_index(articles)
    
    if success:
        test_queries = ["chứng khoán", "công nghệ", "giáo dục", "thể thao", "sức khỏe"]
        benchmark_results = hnsw_mgr.benchmark_multiple_queries(test_queries, k=10)
        print("BENCHMARK HOÀN TẤT!")

if __name__ == "__main__":
    build_article_index()