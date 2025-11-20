import feedparser
import requests
import json
import time
from datetime import datetime
import os

class ArticleCrawler:
    def __init__(self, data_dir='article_data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def crawl_vnexpress_rss(self, max_articles=1000, verbose=False):
        print("ĐANG CRAWL BÀI BÁO TỪ CÁC TRANG BÁO...")
        print("=" * 50)
        
        rss_feeds = [
            # VIETNAMESE NEWS - TIẾNG VIỆT
            'https://vnexpress.net/rss/tin-moi-nhat.rss',
            'https://vnexpress.net/rss/thoi-su.rss',
            'https://vnexpress.net/rss/the-gioi.rss',
            'https://vnexpress.net/rss/kinh-doanh.rss',
            'https://vnexpress.net/rss/giai-tri.rss',
            'https://vnexpress.net/rss/phap-luat.rss',
            'https://vnexpress.net/rss/giao-duc.rss',
            'https://vnexpress.net/rss/suc-khoe.rss',
            'https://vnexpress.net/rss/doi-song.rss',
            'https://vnexpress.net/rss/du-lich.rss',
            'https://vnexpress.net/rss/khoa-hoc.rss',
            'https://vnexpress.net/rss/so-hoa.rss',
            'https://vnexpress.net/rss/oto-xe-may.rss',
            'https://vnexpress.net/rss/the-thao.rss',
            
            'https://dantri.com.vn/rss/thoi-su.rss',
            'https://dantri.com.vn/rss/the-gioi.rss',
            'https://dantri.com.vn/rss/kinh-doanh.rss',
            'https://dantri.com.vn/rss/giai-tri.rss',
            'https://dantri.com.vn/rss/the-thao.rss',
            'https://dantri.com.vn/rss/giao-duc.rss',
            'https://dantri.com.vn/rss/suc-khoe.rss',
            'https://dantri.com.vn/rss/du-lich.rss',
            'https://dantri.com.vn/rss/khoa-hoc.rss',
            'https://dantri.com.vn/rss/cong-nghe.rss',
            
            'https://thanhnien.vn/rss/the-thao.rss',
            'https://thanhnien.vn/rss/the-gioi.rss',
            'https://thanhnien.vn/rss/kinh-te.rss',
            'https://thanhnien.vn/rss/giao-duc.rss',
            'https://thanhnien.vn/rss/thoi-su.rss',
            'https://thanhnien.vn/rss/van-hoa.rss',
            'https://thanhnien.vn/rss/gioi-tre.rss',
            
            'https://tuoitre.vn/rss/the-thao.rss',
            'https://tuoitre.vn/rss/the-gioi.rss',
            'https://tuoitre.vn/rss/kinh-te.rss',
            'https://tuoitre.vn/rss/giao-duc.rss',
            'https://tuoitre.vn/rss/van-hoa.rss',
            'https://tuoitre.vn/rss/nhip-song-tre.rss',
            
            # THÊM CÁC BÁO TIẾNG VIỆT KHÁC - THỂ THAO ĐA DẠNG
            'https://laodong.vn/rss/the-thao.rss',
            'https://laodong.vn/rss/the-gioi.rss',
            'https://laodong.vn/rss/thoi-su.rss',
            'https://laodong.vn/rss/du-lich.rss',
            
            'https://vietnamnet.vn/rss/the-thao.rss',
            'https://vietnamnet.vn/rss/the-gioi.rss',
            'https://vietnamnet.vn/rss/thoi-su.rss',
            'https://vietnamnet.vn/rss/giai-tri.rss',
            'https://vietnamnet.vn/rss/du-lich.rss',
            
            'https://zingnews.vn/rss/the-thao.rss',
            'https://zingnews.vn/rss/the-gioi.rss',
            'https://zingnews.vn/rss/giai-tri.rss',
            'https://zingnews.vn/rss/du-lich.rss',
            
            # BÓNG ĐÁ QUỐC TẾ - TIẾNG VIỆT
            'https://www.24h.com.vn/bong-da-c48.rss',
            'https://www.24h.com.vn/bong-da-quoc-te-c269.rss',
            'https://bongdaplus.vn/rss/bong-da-quoc-te.rss',
            'https://webthethao.vn/rss/bong-da.rss',
            'https://thethao247.vn/bong-da.rss',
            'https://thethao247.vn/bong-da-quoc-te-c2.rss',
            
            # BÓNG ĐÁ QUỐC TẾ - TIẾNG ANH
            'https://www.espn.com/espn/rss/soccer/news',
            'https://www.skysports.com/rss/12040',
            'https://www.skysports.com/rss/12088',
            'https://www.goal.com/feeds/en/news',
            'https://www.bbc.co.uk/sport/football/rss.xml',
            
            # THỂ THAO KHÁC NGOÀI BÓNG ĐÁ - TIẾNG ANH
            'https://www.espn.com/espn/rss/news',
            'https://www.espn.com/espn/rss/nba/news',
            'https://www.espn.com/espn/rss/nfl/news',
            'https://www.espn.com/espn/rss/mlb/news',
            'https://www.espn.com/espn/rss/nhl/news',
            'https://www.espn.com/espn/rss/tennis/news',
            'https://www.bbc.co.uk/sport/rss.xml',
            'https://www.skysports.com/rss/12040',
            'https://www.skysports.com/rss/12059',  # Tennis
            
            # GIẢI TRÍ TIẾNG ANH
            'https://feeds.reuters.com/reuters/entertainment',
            'https://rss.cnn.com/rss/edition_entertainment.rss',
            'https://www.theguardian.com/uk/culture/rss',
            
            # DU LỊCH - TIẾNG ANH (THAY THẾ CHÍNH TRỊ)
            'https://feeds.reuters.com/reuters/lifestyle',
            'https://rss.cnn.com/rss/edition_travel.rss',
            'https://www.theguardian.com/uk/travel/rss',
            'https://www.nationalgeographic.com/travel/feed/',
            'https://www.lonelyplanet.com/news/feed/',
            'https://www.travelandleisure.com/rss/all.xml',
            
            # INTERNATIONAL NEWS TỔNG HỢP (CÁC NGUỒN DỄ TRUY CẬP)
            'https://feeds.reuters.com/reuters/topNews',
            'https://rss.cnn.com/rss/edition.rss',
            'https://www.aljazeera.com/xml/rss/all.xml',
            'https://www.theguardian.com/international/rss',
            'https://feeds.bbci.co.uk/news/rss.xml',
            'https://www.npr.org/rss/rss.php?id=1001'
        ]
        
        all_articles = []
        seen_links = set()
        
        print(f"Bắt đầu crawl từ {len(rss_feeds)} nguồn RSS...")
        
        # Định nghĩa số bài tối đa cho từng loại chuyên mục - TĂNG GIỚI HẠN
        category_limits = {
            # CÁC CHUYÊN MỤC ÍT QUAN TRỌNG: 8-10 bài/feed
            'Sức khỏe': 8,
            'Kinh doanh': 8, 
            'Đời sống': 8,
            'Giáo dục': 8,
            'Kinh tế': 8,
            'Văn hóa': 8,
            'Giới trẻ': 8,
            
            # CÁC CHUYÊN MỤC QUAN TRỌNG: 15-25 bài/feed
            'Thể thao': 20,
            'Bóng đá': 20,
            'Bóng đá quốc tế': 20,
            'Tin mới nhất': 20,
            'Thời sự': 20,
            'Thế giới': 20,
            'Công nghệ': 15,
            'Số hóa': 15,
            'Giải trí': 15,
            'Pháp luật': 15,
            'Khoa học': 15,
            'Xe': 15,
            'Du lịch': 20,  # TĂNG GIỚI HẠN CHO DU LỊCH
            
            # INTERNATIONAL NEWS: 15-20 bài/feed
            'International News': 20,
            'World News': 20,
            'Top Stories': 20,
            'Entertainment': 15,
            'Sports': 20,
            'Lifestyle': 15,
            'Travel': 15
        }
        
        # Crawl từng feed với giới hạn khác nhau
        for feed_url in rss_feeds:
            if len(all_articles) >= max_articles:
                break

            category = self._extract_category(feed_url)
            language = self._extract_language(feed_url)
            limit = category_limits.get(category, 12)  # Mặc định 12 bài
            
            print(f"Đang crawl {category} [{language}] (tối đa {limit} bài)...")

            try:
                feed = feedparser.parse(feed_url)
                articles_from_feed = 0
                
                for entry in feed.entries:
                    if len(all_articles) >= max_articles or articles_from_feed >= limit:
                        break
                    
                    if entry.link in seen_links:
                        continue
                    
                    seen_links.add(entry.link)
                    
                    article = {
                        'id': len(all_articles),
                        'title': entry.title,
                        'summary': entry.summary if hasattr(entry, 'summary') else '',
                        'link': entry.link,
                        'published': entry.published if hasattr(entry, 'published') else '',
                        'category': category,
                        'language': language,
                        'source': self._extract_source(feed_url),
                        'crawled_time': datetime.now().isoformat()
                    }
                    
                    all_articles.append(article)
                    articles_from_feed += 1
                    
                    if verbose:
                        print(f"  {len(all_articles)}. {entry.title[:60]}...")
                    else:
                        if len(all_articles) % 10 == 0:
                            print(f"Đã crawl {len(all_articles)}/{max_articles} bài báo...")
                    
                    time.sleep(0.03)  # Giảm thời gian chờ để crawl nhanh hơn
                        
            except Exception as e:
                print(f"Lỗi với feed {feed_url}: {e}")
                continue
        
        print(f"\nHOÀN THÀNH! Đã crawl được {len(all_articles)} bài báo")
        return all_articles
    
    def _extract_category(self, feed_url):
        categories = {
            # TIẾNG VIỆT
            'tin-moi-nhat': 'Tin mới nhất',
            'thoi-su': 'Thời sự', 
            'the-gioi': 'Thế giới',
            'kinh-doanh': 'Kinh doanh',
            'giai-tri': 'Giải trí',
            'phap-luat': 'Pháp luật',
            'giao-duc': 'Giáo dục',
            'suc-khoe': 'Sức khỏe',
            'doi-song': 'Đời sống',
            'du-lich': 'Du lịch',
            'khoa-hoc': 'Khoa học',
            'so-hoa': 'Số hóa',
            'oto-xe-may': 'Xe',
            'the-thao': 'Thể thao',
            'cong-nghe': 'Công nghệ',
            'xa-hoi': 'Xã hội',
            'kinh-te': 'Kinh tế',
            'bong-da': 'Bóng đá',
            'bong-da-quoc-te': 'Bóng đá quốc tế',
            'van-hoa': 'Văn hóa',
            'gioi-tre': 'Giới trẻ',
            'nhip-song-tre': 'Giới trẻ',
            
            # TIẾNG ANH - THỂ THAO
            'soccer': 'Bóng đá',
            'football': 'Bóng đá',
            'premierleague': 'Bóng đá',
            'laliga': 'Bóng đá',
            'bundesliga': 'Bóng đá',
            'seriea': 'Bóng đá',
            'ligue1': 'Bóng đá',
            'transfermarkt': 'Bóng đá',
            'sports': 'Thể thao',
            'nba': 'Thể thao',
            'nfl': 'Thể thao', 
            'mlb': 'Thể thao',
            'nhl': 'Thể thao',
            'tennis': 'Thể thao',
            'cricket': 'Thể thao',
            
            # TIẾNG ANH - GIẢI TRÍ & DU LỊCH
            'entertainment': 'Giải trí',
            'culture': 'Giải trí',
            'arts': 'Giải trí',
            'lifestyle': 'Lifestyle',
            'travel': 'Du lịch',
            
            # TIẾNG ANH - NEWS TỔNG HỢP
            'topnews': 'International News',
            'topstories': 'Top Stories',
            'edition': 'International News',
            'international': 'International News',
            'world': 'World News'
        }
        
        for key, value in categories.items():
            if key in feed_url.lower():
                return value
        
        # Mặc định cho các feed không xác định
        if 'news' in feed_url.lower():
            return 'International News'
        elif 'rss' in feed_url.lower():
            return 'Tin mới nhất'
        else:
            return 'Thế giới'
    
    def _extract_language(self, feed_url):
        """Phân loại ngôn ngữ"""
        vietnamese_domains = ['vnexpress', 'dantri', 'thanhnien', 'tuoitre', '24h', 'bongdaplus', 
                             'webthethao', 'thethao247', 'laodong', 'vietnamnet', 'zingnews']
        english_domains = ['espn', 'skysports', 'goal', 'bbc', 'theguardian', 
                          'reuters', 'cnn', 'aljazeera', 'npr', 'nationalgeographic',
                          'lonelyplanet', 'travelandleisure']
        
        if any(domain in feed_url for domain in vietnamese_domains):
            return 'Vietnamese'
        elif any(domain in feed_url for domain in english_domains):
            return 'English'
        else:
            return 'Other'
    
    def _extract_source(self, feed_url):
        sources = {
            'vnexpress': 'VnExpress',
            'dantri': 'Dân Trí',
            'thanhnien': 'Thanh Niên',
            'tuoitre': 'Tuổi Trẻ',
            '24h': '24h.com.vn',
            'bongdaplus': 'Bóng Đá Plus',
            'webthethao': 'Webthethao',
            'thethao247': 'Thể thao 247',
            'laodong': 'Lao Động',
            'vietnamnet': 'VietnamNet',
            'zingnews': 'ZingNews',
            
            'espn': 'ESPN',
            'skysports': 'Sky Sports',
            'goal': 'Goal.com',
            'bbc': 'BBC',
            'theguardian': 'The Guardian',
            'reuters': 'Reuters',
            'cnn': 'CNN',
            'aljazeera': 'Al Jazeera',
            'npr': 'NPR',
            'nationalgeographic': 'National Geographic',
            'lonelyplanet': 'Lonely Planet',
            'travelandleisure': 'Travel + Leisure'
        }
        
        for domain, source in sources.items():
            if domain in feed_url:
                return source
        return 'Other'
    
    def save_articles(self, articles, filename='vn_articles.json'):
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        
        print(f"Đã lưu {len(articles)} bài báo vào: {filepath}")
        
        # Tạo file thống kê với phân loại mới
        self._create_statistics_file(articles)
        
        return filepath
    
    def _create_statistics_file(self, articles):
        """Tạo file thống kê .txt với phân loại theo chủ đề và ngôn ngữ"""
        stats_file = os.path.join(self.data_dir, 'thong_ke_bai_bao.txt')
        
        categories = {}
        languages = {}
        sources = {}
        
        for article in articles:
            cat = article['category']
            lang = article['language']
            src = article['source']
            
            categories[cat] = categories.get(cat, 0) + 1
            languages[lang] = languages.get(lang, 0) + 1
            sources[src] = sources.get(src, 0) + 1
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("THỐNG KÊ BÀI BÁO - PHÂN LOẠI THEO CHỦ ĐỀ VÀ NGÔN NGỮ\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Tổng số bài báo: {len(articles)}\n")
            f.write(f"Thời gian thống kê: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("PHÂN BỐ THEO CHỦ ĐỀ:\n")
            f.write("-" * 40 + "\n")
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(articles)) * 100
                f.write(f"{cat:<25} {count:>4} bài ({percentage:5.1f}%)\n")
            
            f.write("\nPHÂN BỐ THEO NGÔN NGỮ:\n")
            f.write("-" * 40 + "\n")
            for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(articles)) * 100
                f.write(f"{lang:<15} {count:>4} bài ({percentage:5.1f}%)\n")
            
            f.write("\nPHÂN BỐ THEO NGUỒN BÁO:\n")
            f.write("-" * 40 + "\n")
            for src, count in sorted(sources.items(), key=lambda x: x[1], reverse=True)[:15]:
                percentage = (count / len(articles)) * 100
                f.write(f"{src:<20} {count:>4} bài ({percentage:5.1f}%)\n")
        
        print(f"Đã tạo file thống kê: {stats_file}")
    
    def load_articles(self, filename='vn_articles.json'):
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"File {filepath} không tồn tại!")
            return []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        print(f"Đã load {len(articles)} bài báo từ: {filepath}")
        return articles

def main():
    print("ARTICLE CRAWLER - PHÂN LOẠI THEO CHỦ ĐỀ & NGÔN NGỮ")
    print("=" * 60)
    
    crawler = ArticleCrawler()
    
    try:
        max_articles = int(input("Số bài báo muốn crawl (mặc định 1000): ").strip() or "1000")
    except:
        max_articles = 1000
    
    articles = crawler.crawl_vnexpress_rss(max_articles=max_articles, verbose=False)
    
    if articles:
        crawler.save_articles(articles)
        
        # Hiển thị thống kê nhanh
        categories = {}
        languages = {}
        
        for article in articles:
            cat = article['category']
            lang = article['language']
            categories[cat] = categories.get(cat, 0) + 1
            languages[lang] = languages.get(lang, 0) + 1
        
        print(f"\nTHỐNG KÊ NHANH:")
        print(f"Tổng số bài báo: {len(articles)}")
        print("\nTop chủ đề:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {cat}: {count} bài")
        
        print("\nPhân bố ngôn ngữ:")
        for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
            print(f"  {lang}: {count} bài")
        
        print(f"\nCRAWL THÀNH CÔNG!")
        print(f"Dữ liệu được lưu trong thư mục: {crawler.data_dir}")
    else:
        print("Không crawl được bài báo nào!")

if __name__ == "__main__":
    main()