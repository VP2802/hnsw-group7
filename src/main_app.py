import numpy as np
import time
import os
from hnsw_manager import ArticleHNSWManager

class ArticleSearchApp:
    def __init__(self):
        self.hnsw_mgr = ArticleHNSWManager('article_index')
        self.is_loaded = False
        
    def load_system(self):
        """Tải hệ thống từ data đã build"""
        print("ĐANG TẢI HỆ THỐNG...")
        print("=" * 50)
        
        try:
            if not os.path.exists('article_index/article_index.bin'):
                print("Chưa có data đã build! Cần chạy build trước.")
                print("Chạy: python article_hnsw_manager.py")
                return False
            
            self.hnsw_mgr.load_index()
            self.is_loaded = True
            
            info = self.hnsw_mgr.get_index_info()
            print("TẢI HỆ THỐNG THÀNH CÔNG!")
            print(f"Thông tin hệ thống:")
            print(f"  • Bài báo: {info['article_count']:,}")
            print(f"  • Vectors: {info['vector_count']:,}")
            print(f"  • Dimensions: {info['dim']}")
            
            return True
            
        except Exception as e:
            print(f"Lỗi tải hệ thống: {e}")
            return False
    
    def show_search_menu(self):
        """Hiển thị menu tìm kiếm"""
        print("\nMENU TÌM KIẾM")
        print("=" * 40)
        print("1. Tìm kiếm đơn giản (HNSW)")
        print("2. So sánh HNSW vs Brute Force") 
        print("3. Benchmark nhiều query")
        print("4. Thống kê hệ thống")
        print("5. Thoát")
        print("=" * 40)
    
    def simple_search(self):
        """Tìm kiếm nhanh chỉ dùng HNSW"""
        if not self.is_loaded:
            if not self.load_system():
                return
        
        print("\nTÌM KIẾM NHANH (HNSW)")
        print("=" * 30)
        
        query = input("Nhập từ khóa: ").strip()
        if not query:
            print("Vui lòng nhập từ khóa!")
            return
        
        try:
            k = int(input("Số kết quả (mặc định 10): ").strip() or "10")
        except:
            k = 10
        
        print(f"\nĐang tìm kiếm: '{query}'...")
        start_time = time.time()
        
        try:
            query_vector = self.hnsw_mgr.embedder.embed_query(query)
            labels, distances = self.hnsw_mgr.index.knn_query(query_vector, k=k)
            
            search_time = time.time() - start_time
            
            print(f"Tìm thấy {len(labels[0])} kết quả trong {search_time:.4f}s")
            print("\nKẾT QUẢ:")
            print("-" * 80)
            
            for i, (label, distance) in enumerate(zip(labels[0], distances[0])):
                article_idx = int(label)
                similarity = 1 - distance
                article = self.hnsw_mgr.articles[article_idx]
                
                print(f"\n#{i+1} | Similarity: {similarity:.3f}")
                print(f"Tiêu đề: {article['title']}")
                print(f"Chuyên mục: {article['category']} | Nguồn: {article['source']}")
                print(f"Thời gian: {article.get('published', 'N/A')}")
                if article.get('summary'):
                    print(f"Tóm tắt: {article['summary'][:150]}...")
                print("-" * 80)
                
        except Exception as e:
            print(f"Lỗi tìm kiếm: {e}")
    
    def comparison_search(self):
        """So sánh HNSW vs Brute Force"""
        if not self.is_loaded:
            if not self.load_system():
                return
        
        print("\nSO SÁNH HNSW vs BRUTE FORCE")
        print("=" * 40)
        
        query = input("Nhập từ khóa: ").strip()
        if not query:
            print("Vui lòng nhập từ khóa!")
            return
        
        try:
            k = int(input("Số kết quả (mặc định 10): ").strip() or "10")
        except:
            k = 10
        
        try:
            results = self.hnsw_mgr.search_with_comparison(query, k=k)
        except Exception as e:
            print(f"Lỗi: {e}")
            return
    
    def run_benchmark(self):
        """Chạy benchmark với nhiều query"""
        if not self.is_loaded:
            if not self.load_system():
                return
        
        print("\nBENCHMARK NHIỀU QUERY")
        print("=" * 40)
        
        default_queries = [
            "chứng khoán", "công nghệ", "giáo dục", 
            "thể thao", "sức khỏe", "xe hơi",
            "du lịch", "bất động sản", "âm nhạc"
        ]
        
        print("Các query mẫu:")
        for i, query in enumerate(default_queries, 1):
            print(f"  {i}. {query}")
        
        choice = input("\nChọn: [1] Dùng mẫu [2] Nhập query: ").strip()
        
        if choice == "1":
            queries = default_queries
        elif choice == "2":
            custom_queries = input("Nhập các query (cách nhau bằng dấu phẩy): ").strip()
            queries = [q.strip() for q in custom_queries.split(",") if q.strip()]
        else:
            queries = default_queries[:5]
        
        try:
            k = int(input("Số kết quả mỗi query (mặc định 10): ").strip() or "10")
        except:
            k = 10
        
        print(f"\nĐang chạy benchmark với {len(queries)} queries...")
        
        try:
            results = self.hnsw_mgr.benchmark_multiple_queries(queries, k=k)
            
            print(f"\n" + "="*60)
            print("KẾT QUẢ BENCHMARK TỔNG HỢP")
            print("="*60)
            
            for result in results:
                query = result['query']
                comparison = result['comparison']
                
                print(f"\n'{query}':")
                print(f"  Tốc độ tăng: {comparison['speedup']:.1f}x")
                print(f"  Độ chính xác: {comparison['accuracy']:.1%}")
                print(f"  Kết quả trùng: {comparison['common_results']}/{k}")
            
            avg_speedup = np.mean([r['comparison']['speedup'] for r in results])
            avg_accuracy = np.mean([r['comparison']['accuracy'] for r in results])
            
            print(f"\nTỔNG KẾT:")
            print(f"  • Tốc độ tăng trung bình: {avg_speedup:.1f}x")
            print(f"  • Độ chính xác trung bình: {avg_accuracy:.1%}")
            
        except Exception as e:
            print(f"Lỗi benchmark: {e}")
    
    def show_system_stats(self):
        """Hiển thị thống kê hệ thống"""
        if not self.is_loaded:
            if not self.load_system():
                return
        
        print("\nTHỐNG KÊ HỆ THỐNG")
        print("=" * 40)
        
        info = self.hnsw_mgr.get_index_info()
        
        print(f"Thông tin cơ bản:")
        print(f"  • Bài báo: {info['article_count']:,}")
        print(f"  • Vectors: {info['vector_count']:,}")
        print(f"  • Dimensions: {info['dim']}")
        
        articles = self.hnsw_mgr.articles
        
        sources = {}
        categories = {}
        
        for article in articles:
            src = article['source']
            cat = article['category']
            
            sources[src] = sources.get(src, 0) + 1
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"\nPHÂN BỐ NGUỒN BÁO:")
        for src, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(articles)) * 100
            print(f"  • {src}: {count} bài ({percentage:.1f}%)")
        
        print(f"\nTOP THỂ LOẠI:")
        top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]
        for cat, count in top_categories:
            percentage = (count / len(articles)) * 100
            print(f"  • {cat}: {count} bài ({percentage:.1f}%)")
        
        index_path = 'article_index/article_index.bin'
        if os.path.exists(index_path):
            size_mb = os.path.getsize(index_path) / (1024 * 1024)
            print(f"\nKích thước index: {size_mb:.2f} MB")
    
    def run(self):
        """Chạy ứng dụng chính"""
        print("HỆ THỐNG TÌM KIẾM BÀI BÁO THÔNG MINH")
        print("=" * 50)
        print("Sử dụng data đã build sẵn - Tốc độ cực nhanh!")
        
        if not self.load_system():
            return
        
        while True:
            self.show_search_menu()
            
            choice = input("Chọn chức năng (1-5): ").strip()
            
            if choice == '1':
                self.simple_search()
            elif choice == '2':
                self.comparison_search()
            elif choice == '3':
                self.run_benchmark()
            elif choice == '4':
                self.show_system_stats()
            elif choice == '5':
                print("Tạm biệt!")
                break
            else:
                print("Lựa chọn không hợp lệ!")
            
            input("\nNhấn Enter để tiếp tục...")

def main():
    try:
        app = ArticleSearchApp()
        app.run()
    except KeyboardInterrupt:
        print("\n\nThoát chương trình!")
    except Exception as e:
        print(f"\nLỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()