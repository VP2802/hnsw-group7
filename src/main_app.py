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
                print("CHƯA CÓ DATA ĐÃ BUILD!")
                print("Cần chạy build index trước:")
                print("   python hnsw_manager.py")
                print("\nHoặc nếu muốn build ngay bây giờ? (y/n): ", end="")
                choice = input().strip().lower()
                if choice == 'y':
                    self.build_index_now()
                return False
            
            self.hnsw_mgr.load_index()
            self.is_loaded = True
            
            info = self.hnsw_mgr.get_index_info()
            print("TẢI HỆ THỐNG THÀNH CÔNG!")
            print(f"Thông tin hệ thống:")
            print(f"   • Bài báo: {info['article_count']:,}")
            print(f"   • Vectors: {info['vector_count']:,}")
            print(f"   • Dimensions: {info['dim']}")
            
            return True
            
        except Exception as e:
            print(f"Lỗi tải hệ thống: {e}")
            return False
    
    def build_index_now(self):
        """Xây dựng index ngay lập tức"""
        print("\nĐANG XÂY DỰNG INDEX...")
        try:
            from crawl_articles import ArticleCrawler
            
            crawler = ArticleCrawler()
            articles = crawler.load_articles()
            
            if not articles:
                print("Đang crawl dữ liệu mới...")
                articles = crawler.crawl_vnexpress_rss(max_articles=1000, verbose=False)
                if articles:
                    crawler.save_articles(articles)
                else:
                    print("Không crawl được dữ liệu!")
                    return False
            
            print(f"Đã có {len(articles)} bài báo, đang build index...")
            success = self.hnsw_mgr.build_index(articles)
            
            if success:
                self.hnsw_mgr.load_index()
                self.is_loaded = True
                print("BUILD INDEX THÀNH CÔNG!")
                return True
            else:
                print("BUILD INDEX THẤT BẠI!")
                return False
                
        except Exception as e:
            print(f"Lỗi build index: {e}")
            return False
    
    def show_main_menu(self):
        """Hiển thị menu chính"""
        print("\n" + "="*60)
        print("HỆ THỐNG TÌM KIẾM BÀI BÁO THÔNG MINH")
        print("="*60)
        print("1. Tìm kiếm nhanh (gợi ý) - HNSW k=20")
        print("2. Tìm kiếm tùy chỉnh - HNSW")
        print("3. Tìm kiếm theo nguồn báo")
        print("4. So sánh HNSW vs Brute Force")
        print("5. Thống kê nguồn báo")
        print("6. Thoát")
        print("="*60)
    
    def quick_search_demo(self):
        """Tìm kiếm nhanh với từ khóa gợi ý"""
        if not self.is_loaded:
            print("Hệ thống chưa được tải!")
            return
        
        print("\nTÌM KIẾM NHANH - K=20")
        print("="*50)
        
        # Các từ khóa gợi ý
        suggested_queries = [
            "bóng đá Premier League",
            "công nghệ AI", 
            "chứng khoán thị trường",
            "du lịch Đà Nẵng",
            "sức khỏe dinh dưỡng"
        ]
        
        print("Từ khóa gợi ý:")
        for i, query in enumerate(suggested_queries, 1):
            print(f"   {i}. {query}")
        
        try:
            choice = int(input("\nChọn từ khóa (1-5): ").strip())
            if 1 <= choice <= 5:
                query = suggested_queries[choice-1]
            else:
                query = suggested_queries[0]
        except:
            query = suggested_queries[0]
        
        print(f"\nĐang tìm: '{query}' với k=20...")
        print("Đang xử lý...")
        
        start_time = time.time()
        
        try:
            query_vector = self.hnsw_mgr.embedder.embed_query(query)
            labels, distances = self.hnsw_mgr.index.knn_query(query_vector, k=20)
            
            search_time = time.time() - start_time
            
            print(f"Tìm thấy {len(labels[0])} kết quả trong {search_time:.4f}s")
            print("\nTOP 5 KẾT QUẢ:")
            print("="*100)
            
            for i, (label, distance) in enumerate(zip(labels[0][:5], distances[0][:5])):
                article_idx = int(label)
                similarity = 1 - distance
                article = self.hnsw_mgr.articles[article_idx]
                
                print(f"\n #{i+1} | Độ tương đồng: {similarity:.3f}")
                print(f" Tiêu đề: {article['title']}")
                print(f" Chuyên mục: {article['category']} | Nguồn: {article['source']}")
                print(f" Ngôn ngữ: {article['language']}")
                if article.get('summary'):
                    summary = article['summary']
                    if len(summary) > 200:
                        summary = summary[:200] + "..."
                    print(f" Tóm tắt: {summary}")
                print("-" * 100)
            
            # Hiển thị thêm kết quả
            if len(labels[0]) > 5:
                print(f"\nVà {len(labels[0]) - 5} kết quả khác...")
                
        except Exception as e:
            print(f"Lỗi tìm kiếm: {e}")
    
    def custom_search(self):
        """Tìm kiếm tùy chỉnh với HNSW"""
        if not self.is_loaded:
            print("Hệ thống chưa được tải!")
            return
        
        print("\nTÌM KIẾM TÙY CHỈNH - HNSW")
        print("="*50)
        
        query = input("Nhập từ khóa tìm kiếm: ").strip()
        if not query:
            print("Vui lòng nhập từ khóa!")
            return
        
        try:
            k = int(input(f"Số kết quả (mặc định 10): ").strip() or "10")
            k = min(k, 50)  # Giới hạn tối đa 50 kết quả
        except:
            k = 10
        
        print(f"\nĐang tìm: '{query}' với k={k}...")
        print("Đang xử lý...")
        
        start_time = time.time()
        
        try:
            query_vector = self.hnsw_mgr.embedder.embed_query(query)
            labels, distances = self.hnsw_mgr.index.knn_query(query_vector, k=k)
            
            search_time = time.time() - start_time
            
            print(f"Tìm thấy {len(labels[0])} kết quả trong {search_time:.4f}s")
            print("\nKẾT QUẢ:")
            print("="*100)
            
            for i, (label, distance) in enumerate(zip(labels[0], distances[0])):
                article_idx = int(label)
                similarity = 1 - distance
                article = self.hnsw_mgr.articles[article_idx]
                
                print(f"\n#{i+1} | Độ tương đồng: {similarity:.3f}")
                print(f" {article['title']}")
                print(f" {article['category']} | {article['source']} | {article['language']}")
                if article.get('summary'):
                    summary = article['summary']
                    if len(summary) > 150:
                        summary = summary[:150] + "..."
                    print(f" {summary}")
                print("-" * 100)
                
        except Exception as e:
            print(f"Lỗi tìm kiếm: {e}")
    
    def search_by_source(self):
        """Tìm kiếm bài báo theo nguồn báo"""
        if not self.is_loaded:
            print("Hệ thống chưa được tải!")
            return
        
        print("\nTÌM KIẾM THEO NGUỒN BÁO")
        print("="*50)
        
        # Hiển thị các nguồn báo có sẵn
        available_sources = self.hnsw_mgr.get_available_sources()
        print("Các nguồn báo có sẵn:")
        for i, source in enumerate(available_sources[:15], 1):  # Hiển thị 15 nguồn đầu
            print(f"   {i:2d}. {source}")
        if len(available_sources) > 15:
            print(f"   ... và {len(available_sources) - 15} nguồn khác")
        
        print("\nCách sử dụng:")
        print("   • Nhập tên đầy đủ: 'Dân Trí'")
        print("   • Nhập một phần: 'vnexpress' hoặc 'bbc'")
        print("   • Nhập '0' để quay lại")
        
        source_name = input("\nNhập tên nguồn báo: ").strip()
        
        if source_name == '0':
            return
        
        if not source_name:
            print("Vui lòng nhập tên nguồn báo!")
            return
        
        try:
            k = int(input(f"Số kết quả (mặc định 15): ").strip() or "15")
            k = min(k, 50)
        except:
            k = 15
        
        print(f"\nĐang tìm bài báo từ '{source_name}'...")
        print("Đang xử lý...")
        
        try:
            results = self.hnsw_mgr.search_by_source(source_name, k=k)
            
            print(f"\nKẾT QUẢ TỪ '{source_name}':")
            print(f"Tổng số bài báo tìm thấy: {results['count']}")
            print("="*100)
            
            if results['count'] == 0:
                print("Không tìm thấy bài báo nào từ nguồn này!")
                print("Gợi ý: Kiểm tra lại tên nguồn báo hoặc thử tìm với từ khóa ngắn hơn")
                return
            
            for i, result in enumerate(results['results'][:k], 1):
                article = result['article']
                
                print(f"\n #{i} | {article['source']}")
                print(f" Tiêu đề: {article['title']}")
                print(f" Chuyên mục: {article['category']} | Ngôn ngữ: {article['language']}")
                
                # Hiển thị thời gian nếu có
                if article.get('published'):
                    print(f" Thời gian: {article['published']}")
                
                if article.get('summary'):
                    summary = article['summary']
                    if len(summary) > 200:
                        summary = summary[:200] + "..."
                    print(f" Tóm tắt: {summary}")
                
                print(f" Link: {article['link']}")
                print("-" * 100)
            
            # Gợi ý tìm kiếm kết hợp
            if results['count'] > 0:
                print(f"\nGợi ý: Bạn có thể tìm kiếm kết hợp nội dung + nguồn báo")
                combine = input("   Muốn tìm thêm nội dung cụ thể? (y/n): ").strip().lower()
                if combine == 'y':
                    content_query = input("   Nhập nội dung tìm kiếm: ").strip()
                    if content_query:
                        print(f"\nĐang tìm '{content_query}' trong '{source_name}'...")
                        combined_results = self.hnsw_mgr.search_with_comparison(
                            content_query, k=10, filter_source=source_name
                        )
                        self.hnsw_mgr.display_search_results(combined_results)
                        
        except Exception as e:
            print(f"Lỗi tìm kiếm: {e}")
    
    def comparison_search(self):
        """So sánh HNSW vs Brute Force"""
        if not self.is_loaded:
            print("Hệ thống chưa được tải!")
            return
        
        print("\nSO SÁNH HNSW vs BRUTE FORCE")
        print("="*50)
        
        query = input("Nhập từ khóa: ").strip()
        if not query:
            print("Vui lòng nhập từ khóa!")
            return
        
        try:
            k = int(input(f"Số kết quả (mặc định 10): ").strip() or "10")
            k = min(k, 30)  # Giới hạn để brute force không quá chậm
        except:
            k = 10
        
        print(f"\nĐang so sánh với: '{query}', k={k}")
        print("Đang xử lý...")
        
        try:
            results = self.hnsw_mgr.search_with_comparison(query, k=k)
            
            # Hiển thị kết quả trực quan
            comparison = results['comparison']
            print(f"\nKẾT QUẢ SO SÁNH:")
            print("="*60)
            print(f" Tốc độ tăng: {comparison['speedup']:.1f}x")
            print(f" Độ chính xác: {comparison['accuracy']:.1%}")
            print(f" Kết quả trùng: {comparison['common_results']}/{k}")
            
            # Hiển thị top kết quả từ HNSW
            print(f"\nTOP 3 KẾT QUẢ HNSW:")
            print("="*80)
            
            hnsw_results = results['hnsw']['results'][:3]
            for i, result in enumerate(hnsw_results, 1):
                article_idx = result['index']
                similarity = result['similarity']
                article = self.hnsw_mgr.articles[article_idx]
                
                print(f"\n#{i} | Độ tương đồng: {similarity:.3f}")
                print(f" {article['title']}")
                print(f" {article['category']} | {article['source']}")
                print("-" * 80)
                
        except Exception as e:
            print(f"Lỗi so sánh: {e}")
    
    def show_statistics(self):
        """Hiển thị thống kê nguồn báo"""
        if not self.is_loaded:
            print("Hệ thống chưa được tải!")
            return
        
        print("\nTHỐNG KÊ NGUỒN BÁO")
        print("="*50)
        
        articles = self.hnsw_mgr.articles
        
        # Thống kê nguồn báo
        sources = {}
        categories = {}
        languages = {}
        
        for article in articles:
            src = article['source']
            cat = article['category']
            lang = article['language']
            
            sources[src] = sources.get(src, 0) + 1
            categories[cat] = categories.get(cat, 0) + 1
            languages[lang] = languages.get(lang, 0) + 1
        
        print(f"TỔNG QUAN:")
        print(f"   • Tổng bài báo: {len(articles):,}")
        print(f"   • Số nguồn báo: {len(sources)}")
        print(f"   • Số chuyên mục: {len(categories)}")
        
        print(f"\nTOP NGUỒN BÁO:")
        print("-" * 40)
        top_sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)[:10]
        for src, count in top_sources:
            percentage = (count / len(articles)) * 100
            print(f"   • {src:<20} {count:>4} bài ({percentage:5.1f}%)")
        
        print(f"\nTOP CHUYÊN MỤC:")
        print("-" * 40)
        top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:8]
        for cat, count in top_categories:
            percentage = (count / len(articles)) * 100
            print(f"   • {cat:<25} {count:>4} bài ({percentage:5.1f}%)")
        
        print(f"\nPHÂN BỐ NGÔN NGỮ:")
        print("-" * 40)
        for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(articles)) * 100
            print(f"   • {lang:<15} {count:>4} bài ({percentage:5.1f}%)")
        
        # Thông tin index
        index_path = 'article_index/article_index.bin'
        if os.path.exists(index_path):
            size_mb = os.path.getsize(index_path) / (1024 * 1024)
            print(f"\nKích thước index: {size_mb:.2f} MB")
    
    def run(self):
        """Chạy ứng dụng chính"""
        print("KHỞI ĐỘNG HỆ THỐNG TÌM KIẾM BÀI BÁO")
        print("Sử dụng HNSW - Tốc độ cực nhanh!")
        
        # Tự động load hệ thống
        if not self.load_system():
            print("\nKhông thể khởi động hệ thống!")
            return
        
        while True:
            self.show_main_menu()
            
            choice = input("Chọn chức năng (1-6): ").strip()
            
            if choice == '1':
                self.quick_search_demo()
            elif choice == '2':
                self.custom_search()
            elif choice == '3':
                self.search_by_source()
            elif choice == '4':
                self.comparison_search()
            elif choice == '5':
                self.show_statistics()
            elif choice == '6':
                print("\nTạm biệt! Hẹn gặp lại!")
                break
            else:
                print("Lựa chọn không hợp lệ! Vui lòng chọn 1-6")
            
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