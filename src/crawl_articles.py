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
    
    def crawl_vnexpress_rss(self, max_articles=5000, verbose=False):
        print("ĐANG CRAWL BÀI BÁO TỪ CÁC TRANG BÁO...")
        print("=" * 50)
        
        rss_feeds = [
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
            'https://vnexpress.net/rss/y-kien.rss',
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
            'https://dantri.com.vn/rss/xa-hoi.rss'
        ]
        
        all_articles = []
        seen_links = set()
        
        for feed_url in rss_feeds:
            if len(all_articles) >= max_articles:
                break

            print(f" Đang crawl: {self._extract_category(feed_url)}...")

            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    if len(all_articles) >= max_articles:
                        break
                    
                    if entry.link in seen_links:
                        continue
                    
                    seen_links.add(entry.link)
                    
                    article = {
                        'id': len(all_articles),
                        'title': entry.title,
                        'summary': entry.summary,
                        'link': entry.link,
                        'published': entry.published if hasattr(entry, 'published') else '',
                        'category': self._extract_category(feed_url),
                        'source': 'VnExpress' if 'vnexpress' in feed_url else 'Dân Trí',
                        'crawled_time': datetime.now().isoformat()
                    }
                    
                    all_articles.append(article)
                    
                    if verbose:
                        print(f"     {len(all_articles)}. {entry.title[:60]}...")
                    else:
                        print(f" Đã crawl {len(all_articles)} bài báo...", end='\r')
                    
                    time.sleep(0.1)
                        
            except Exception as e:
                print(f" Lỗi với feed {feed_url}: {e}")
                continue
        
        print(f"\nHOÀN THÀNH! Đã crawl được {len(all_articles)} bài báo")
        return all_articles
    
    def _extract_category(self, feed_url):
        categories = {
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
            'y-kien': 'Ý kiến',
            'the-thao': 'Thể thao'
        }
        
        for key, value in categories.items():
            if key in feed_url:
                return value
        return 'Khác'
    
    def save_articles(self, articles, filename='vn_articles.json'):
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        
        print(f"Đã lưu {len(articles)} bài báo vào: {filepath}")
        return filepath
    
    def load_articles(self, filename='vn_articles.json'):
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"File {filepath} không tồn tại!")
            return []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        print(f" Đã load {len(articles)} bài báo từ: {filepath}")
        return articles

def main():
    print("ARTICLE CRAWLER")
    print("=" * 40)
    
    crawler = ArticleCrawler()
    
    articles = crawler.crawl_vnexpress_rss(max_articles=500)
    
    if articles:
        crawler.save_articles(articles)
        
        categories = {}
        for article in articles:
            cat = article['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"\n THỐNG KÊ:")
        for cat, count in categories.items():
            print(f"   • {cat}: {count} bài")
        
        print(f"\nCRAWL THÀNH CÔNG!")
        print(f"Dữ liệu được lưu trong thư mục: {crawler.data_dir}")
    else:
        print("Không crawl được bài báo nào!")

if __name__ == "__main__":
    main()