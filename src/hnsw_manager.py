import numpy as np
import os
import time
import hnswlib
import json
import pickle
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
    
    def get_available_sources(self):
        """Lấy danh sách các nguồn báo có sẵn"""
        if not self.articles:
            return []
        
        sources = set()
        for article in self.articles:
            sources.add(article['source'])
        
        return sorted(list(sources))
    
    def build_index(self, articles, max_elements=10000, ef_construction=200, M=16):
        print("ĐANG XÂY DỰNG INDEX TÌM KIẾM BÀI BÁO")
        print("=" * 50)
        
        print(f"Tổng số bài báo đầu vào: {len(articles)}")
        
        # Lọc các bài báo trùng lặp dựa trên link
        unique_articles = []
        seen_links = set()
        
        for article in articles:
            if article['link'] not in seen_links:
                unique_articles.append(article)
                seen_links.add(article['link'])
        
        print(f"Số bài báo sau khi lọc trùng: {len(unique_articles)}")
        
        valid_articles, embeddings = self.embedder.embed_articles(unique_articles)
        self.articles = valid_articles
        self.all_embeddings = embeddings
        
        if len(embeddings) == 0:
            print("Không có embeddings để xây dựng index!")
            return False
        
        print(f"Dữ liệu embedding: {len(embeddings)} bài báo, {embeddings.shape[1]} chiều")
        
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
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def _save_metadata(self):
        metadata = {
            'dim': self.dim,
            'total_articles': len(self.articles),
            'articles': self.articles,
            'build_time': time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        metadata_path = os.path.join(self.index_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Lưu embeddings riêng để tránh file quá lớn
        if self.all_embeddings is not None:
            embeddings_path = os.path.join(self.index_dir, 'embeddings.npy')
            np.save(embeddings_path, self.all_embeddings)
            print(f"Đã lưu embeddings: {embeddings_path}")
    
    def load_index(self):
        print(f"Đang tải index từ {self.index_dir}...")
        
        metadata_path = os.path.join(self.index_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Không tìm thấy metadata: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        self.dim = metadata['dim']
        self.articles = metadata['articles']
        
        # Tải embeddings từ file .npy
        embeddings_path = os.path.join(self.index_dir, 'embeddings.npy')
        if os.path.exists(embeddings_path):
            print("Đang tải embeddings từ file...")
            self.all_embeddings = np.load(embeddings_path)
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
    
    def search_by_source(self, source_name, k=20):
        """Tìm kiếm bài báo theo nguồn báo cụ thể"""
        if not self.articles:
            raise RuntimeError("Hệ thống chưa được khởi tạo!")
        
        print(f"TÌM KIẾM THEO NGUỒN BÁO: '{source_name}'")
        print("=" * 60)
        
        # Tìm tất cả bài báo từ nguồn này
        source_articles = []
        for i, article in enumerate(self.articles):
            if source_name.lower() in article['source'].lower():
                source_articles.append((i, article))
        
        print(f"Tìm thấy {len(source_articles)} bài báo từ {source_name}")
        
        if not source_articles:
            return {
                'source': source_name,
                'results': [],
                'count': 0
            }
        
        # Sắp xếp theo thời gian (nếu có)
        source_articles.sort(key=lambda x: x[1].get('published', ''), reverse=True)
        
        results = []
        for idx, (article_idx, article) in enumerate(source_articles[:k]):
            results.append({
                'index': article_idx,
                'article': article,
                'rank': idx + 1
            })
        
        return {
            'source': source_name,
            'results': results,
            'count': len(source_articles)
        }
    
    def search_with_comparison(self, query, k=10, filter_source=None):
        """Tìm kiếm với khả năng lọc theo nguồn báo"""
        if self.index is None or self.all_embeddings is None:
            raise RuntimeError("Hệ thống chưa được khởi tạo!")
        
        print(f"SO SÁNH TÌM KIẾM: '{query}'")
        if filter_source:
            print(f"LỌC THEO NGUỒN: '{filter_source}'")
        print("=" * 60)
        
        query_vector = self.embedder.embed_query(query)
        
        # Brute Force
        print("BRUTE FORCE...")
        start_time = time.time()
        
        similarities = []
        for i in range(len(self.all_embeddings)):
            # Lọc theo nguồn nếu có
            if filter_source and filter_source.lower() not in self.articles[i]['source'].lower():
                continue
                
            similarity = self._cosine_similarity(query_vector[0], self.all_embeddings[i])
            similarities.append((i, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        brute_results = similarities[:k]
        brute_time = time.time() - start_time
        
        # HNSW
        print("HNSW SEARCH...")
        start_time = time.time()
        
        labels, distances = self.index.knn_query(query_vector, k=min(k*3, len(self.articles)))  # Lấy nhiều hơn để lọc
        
        # Lọc kết quả theo nguồn nếu có
        hnsw_results = []
        for i, (label, distance) in enumerate(zip(labels[0], distances[0])):
            article_idx = int(label)
            
            # Lọc theo nguồn
            if filter_source and filter_source.lower() not in self.articles[article_idx]['source'].lower():
                continue
                
            similarity = 1 - distance
            hnsw_results.append((article_idx, similarity))
            
            if len(hnsw_results) >= k:
                break
        
        hnsw_time = time.time() - start_time
        
        # So sánh
        print(f"KẾT QUẢ SO SÁNH:")
        print(f"  Brute Force: {brute_time:.4f}s")
        print(f"  HNSW: {hnsw_time:.4f}s")
        if hnsw_time > 0:
            print(f"  Tốc độ tăng: {brute_time/hnsw_time:.1f}x")
        
        brute_indices = {idx for idx, _ in brute_results}
        hnsw_indices = {idx for idx, _ in hnsw_results}
        common_results = brute_indices & hnsw_indices
        
        accuracy = len(common_results) / min(len(brute_results), len(hnsw_results)) if min(len(brute_results), len(hnsw_results)) > 0 else 0
        
        print(f"  Độ chính xác Top-{k}: {accuracy:.1%}")
        print(f"  Kết quả trùng: {len(common_results)}/{min(len(brute_results), len(hnsw_results))}")
        
        return {
            'query': query,
            'filter_source': filter_source,
            'brute_force': {
                'results': [{'index': idx, 'similarity': sim} for idx, sim in brute_results],
                'time': brute_time,
                'count': len(brute_results)
            },
            'hnsw': {
                'results': [{'index': idx, 'similarity': sim} for idx, sim in hnsw_results],
                'time': hnsw_time,
                'count': len(hnsw_results)
            },
            'comparison': {
                'speedup': brute_time / hnsw_time if hnsw_time > 0 else 0,
                'accuracy': accuracy,
                'common_results': len(common_results)
            }
        }
    
    def display_search_results(self, search_result, show_details=True):
        """Hiển thị kết quả tìm kiếm"""
        # Xử lý kết quả tìm kiếm theo nguồn
        if 'source' in search_result:
            source = search_result['source']
            results = search_result['results']
            
            print(f"\nTOP KẾT QUẢ TỪ NGUỒN: '{source}'")
            print(f"Tổng số bài báo: {search_result['count']}")
            print("-" * 80)
            
            for i, result in enumerate(results[:10], 1):
                article = result['article']
                print(f"{i}. {article['title']}")
                print(f"   Nguồn: {article['source']} | {article['category']} | {article['language']}")
                if show_details and article.get('summary'):
                    summary = article['summary'][:150] + "..." if len(article['summary']) > 150 else article['summary']
                    print(f"   Tóm tắt: {summary}")
                print(f"   Link: {article['link']}")
                print()
            return
        
        # Xử lý kết quả tìm kiếm thông thường
        query = search_result.get('query', '')
        source = search_result.get('filter_source', '')
        hnsw_results = search_result.get('hnsw', {}).get('results', [])
        
        title = f"TOP KẾT QUẢ CHO: '{query}'"
        if source:
            title += f" [NGUỒN: {source}]"
            
        print(f"\n{title}")
        print("-" * 80)
        
        for i, result in enumerate(hnsw_results[:5], 1):
            idx = result['index']
            similarity = result['similarity']
            article = self.articles[idx]
            
            print(f"{i}. [{similarity:.3f}] {article['title']}")
            print(f"   Nguồn: {article['source']} | {article['category']} | {article['language']}")
            if show_details and article.get('summary'):
                summary = article['summary'][:150] + "..." if len(article['summary']) > 150 else article['summary']
                print(f"   Tóm tắt: {summary}")
            print(f"   Link: {article['link']}")
            print()

    def benchmark_multiple_queries(self, queries, k=10):
        results = {}
        for query in queries:
            res = self.search_with_comparison(query, k=k)
        
        # Convert numpy types to Python native types for JSON serialization
        query_result = {
            'query': query,
            'filter_source': res.get('filter_source'),
            'brute_force': {
                'results': [
                    {
                        'index': int(item['index']),
                        'similarity': float(item['similarity'])  # Convert float32 to float
                    } for item in res['brute_force']['results']
                ],
                'time': float(res['brute_force']['time']),
                'count': int(res['brute_force']['count'])
            },
            'hnsw': {
                'results': [
                    {
                        'index': int(item['index']),
                        'similarity': float(item['similarity'])  # Convert float32 to float
                    } for item in res['hnsw']['results']
                ],
                'time': float(res['hnsw']['time']),
                'count': int(res['hnsw']['count'])
            },
            'comparison': {
                'speedup': float(res['comparison']['speedup']),
                'accuracy': float(res['comparison']['accuracy']),
                'common_results': int(res['comparison']['common_results'])
            }
        }
        results[query] = query_result
    
        return results

    def interactive_search(self):
        while True:
            query = input("\nNhập câu truy vấn (hoặc 'exit' để thoát): ").strip()
            if query.lower() in ['exit', 'quit']:
                break
            filter_source = input("Lọc theo nguồn (Enter để bỏ qua): ").strip() or None
            k = input("Số kết quả muốn hiển thị (default 5): ").strip()
            k = int(k) if k.isdigit() else 5
            res = self.search_with_comparison(query, k=k, filter_source=filter_source)
            self.display_search_results(res)


# Các hàm khác giữ nguyên...
def build_and_test_article_index():
    from crawl_articles import ArticleCrawler
    
    print("BENCHMARK: HNSW vs BRUTE FORCE")
    print("=" * 50)
    
    # Tải hoặc crawl dữ liệu mới
    crawler = ArticleCrawler()
    articles = crawler.load_articles()
    
    if not articles:
        print("Không có dữ liệu bài báo, đang crawl mới...")
        articles = crawler.crawl_vnexpress_rss(max_articles=1000, verbose=False)
        if articles:
            crawler.save_articles(articles)
        else:
            print("Không crawl được dữ liệu mới!")
            return
    
    print(f"Đã tải {len(articles)} bài báo")
    
    # Xây dựng index
    hnsw_mgr = ArticleHNSWManager()
    success = hnsw_mgr.build_index(articles)
    
    if not success:
        print("Xây dựng index thất bại!")
        return
    
    # Test với queries đa dạng
    test_queries = [
        "bóng đá Premier League",
        "chứng khoán thị trường",
        "công nghệ AI trí tuệ nhân tạo",
        "sức khỏe dinh dưỡng",
        "giáo dục đại học"
    ]
    
    print(f"\nBẮT ĐẦU TEST VỚI {len(test_queries)} QUERIES ĐA DẠNG")
    print("=" * 80)
    
    benchmark_results = hnsw_mgr.benchmark_multiple_queries(test_queries, k=10)
    
    # Lưu kết quả benchmark
    benchmark_file = os.path.join(hnsw_mgr.index_dir, 'benchmark_results.json')
    with open(benchmark_file, 'w', encoding='utf-8') as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nĐã lưu kết quả benchmark: {benchmark_file}")
    
    # Test tìm kiếm theo nguồn
    print("\nTEST TÌM KIẾM THEO NGUỒN BÁO")
    print("=" * 50)
    
    test_sources = ["Dân Trí", "VnExpress", "BBC", "Reuters"]
    for source in test_sources:
        try:
            result = hnsw_mgr.search_by_source(source, k=5)
            hnsw_mgr.display_search_results(result)
        except Exception as e:
            print(f"Lỗi khi tìm kiếm nguồn {source}: {e}")
    
    print("BENCHMARK HOÀN TẤT!")

def test_existing_index():
    """Test với index đã có sẵn"""
    hnsw_mgr = ArticleHNSWManager()
    
    try:
        hnsw_mgr.load_index()
        print(f"Đã tải index với {len(hnsw_mgr.articles)} bài báo")
        
        # Chạy chế độ tương tác
        hnsw_mgr.interactive_search()
        
    except Exception as e:
        print(f"Lỗi khi tải index: {e}")
        print("Cần xây dựng index mới...")
        build_and_test_article_index()

if __name__ == "__main__":
    print("HNSW MANAGER - HỆ THỐNG TÌM KIẾM BÀI BÁO")
    print("=" * 50)
    
    # Kiểm tra xem index đã tồn tại chưa
    index_exists = os.path.exists(os.path.join('article_index', 'metadata.json'))
    
    if index_exists:
        choice = input("Index đã tồn tại. Bạn muốn:\n1. Test index hiện có\n2. Xây dựng index mới\nChọn (1/2): ").strip()
        if choice == "1":
            test_existing_index()
        else:
            build_and_test_article_index()
    else:
        build_and_test_article_index()