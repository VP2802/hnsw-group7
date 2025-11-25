import feedparser
import requests
import json
import time
from datetime import datetime
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class ArticleCrawler:
    def __init__(self, data_dir='article_data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def crawl_vnexpress_rss(self, max_articles=10000, verbose=False):
        print("ĐANG CRAWL BÀI BÁO TỪ CÁC TRANG BÁO...")
        print("=" * 50)
        
        rss_feeds = [
            # === BÁO VIỆT NAM (HOẠT ĐỘNG TỐT) ===
            'https://vnexpress.net/rss/tin-moi-nhat.rss',
            'https://vnexpress.net/rss/thoi-su.rss',
            'https://vnexpress.net/rss/the-gioi.rss',
            'https://vnexpress.net/rss/kinh-doanh.rss',
            'https://vnexpress.net/rss/giai-tri.rss',
            'https://vnexpress.net/rss/the-thao.rss',
            'https://vnexpress.net/rss/phap-luat.rss',
            'https://vnexpress.net/rss/giao-duc.rss',
            'https://vnexpress.net/rss/suc-khoe.rss',
            'https://vnexpress.net/rss/doi-song.rss',
            'https://vnexpress.net/rss/du-lich.rss',
            'https://vnexpress.net/rss/khoa-hoc.rss',
            'https://vnexpress.net/rss/so-hoa.rss',
            'https://vnexpress.net/rss/oto-xe-may.rss',
            
            'https://dantri.com.vn/rss/thoi-su.rss',
            'https://dantri.com.vn/rss/the-gioi.rss',
            'https://dantri.com.vn/rss/kinh-doanh.rss',
            'https://dantri.com.vn/rss/giai-tri.rss',
            'https://dantri.com.vn/rss/the-thao.rss',
            'https://dantri.com.vn/rss/giao-duc.rss',
            'https://dantri.com.vn/rss/suc-khoe.rss',
            'https://dantri.com.vn/rss/du-lich.rss',
            
            'https://tuoitre.vn/rss/thoi-su.rss',
            'https://tuoitre.vn/rss/the-gioi.rss',
            'https://tuoitre.vn/rss/kinh-doanh.rss',
            'https://tuoitre.vn/rss/giai-tri.rss',
            'https://tuoitre.vn/rss/the-thao.rss',
            'https://tuoitre.vn/rss/giao-duc.rss',
            'https://tuoitre.vn/rss/suc-khoe.rss',
            
            'https://thanhnien.vn/rss/thoi-su.rss',
            'https://thanhnien.vn/rss/the-gioi.rss',
            'https://thanhnien.vn/rss/kinh-doanh.rss',
            'https://thanhnien.vn/rss/giai-tri.rss',
            'https://thanhnien.vn/rss/the-thao.rss',
            'https://thanhnien.vn/rss/giao-duc.rss',
            'https://thanhnien.vn/rss/suc-khoe.rss',
            
            'https://vietnamnet.vn/rss/thoi-su.rss',
            'https://vietnamnet.vn/rss/the-gioi.rss',
            'https://vietnamnet.vn/rss/kinh-doanh.rss',
            'https://vietnamnet.vn/rss/giai-tri.rss',
            'https://vietnamnet.vn/rss/the-thao.rss',
            'https://vietnamnet.vn/rss/giao-duc.rss',
            'https://vietnamnet.vn/rss/suc-khoe.rss',
            
            'https://zingnews.vn/rss/thoi-su.rss',
            'https://zingnews.vn/rss/the-gioi.rss',
            'https://zingnews.vn/rss/kinh-doanh.rss',
            'https://zingnews.vn/rss/giai-tri.rss',
            'https://zingnews.vn/rss/the-thao.rss',
            
            'https://laodong.vn/rss/thoi-su.rss',
            'https://laodong.vn/rss/the-gioi.rss',
            'https://laodong.vn/rss/kinh-doanh.rss',
            'https://laodong.vn/rss/the-thao.rss',
            
            'https://cafef.vn/rss/thoi-su.rss',
            'https://cafef.vn/rss/the-gioi.rss',
            'https://cafef.vn/rss/kinh-doanh.rss',
            'https://cafef.vn/rss/thi-truong.rss',
            
            'https://vtv.vn/rss/thoi-su.rss',
            'https://vtv.vn/rss/the-gioi.rss',
            'https://vtv.vn/rss/kinh-te.rss',
            
            # === BÁO THỂ THAO VIỆT NAM ===
            'https://www.24h.com.vn/rss/tin-bong-da.rss',
            'https://bongdaplus.vn/rss/bong-da-viet-nam.rss',
            'https://bongdaplus.vn/rss/bong-da-quoc-te.rss',
            'https://thethao247.vn/rss/tin-bong-da.rss',
            'https://webthethao.vn/rss/bong-da.rss',
            
            # === BÁO QUỐC TẾ HOẠT ĐỘNG TỐT ===
            'https://feeds.bbci.co.uk/news/rss.xml',
            'https://feeds.bbci.co.uk/news/world/rss.xml',
            'https://feeds.bbci.co.uk/news/business.rss.xml',
            'https://feeds.bbci.co.uk/news/technology.rss.xml',
            'https://feeds.bbci.co.uk/news/science_and_environment.rss.xml',
            
            'https://www.theguardian.com/international/rss',
            'https://www.theguardian.com/world.rss',
            'https://www.theguardian.com/business.rss',
            'https://www.theguardian.com/technology.rss',
            'https://www.theguardian.com/science.rss',
            
            'https://apnews.com/apf-topnews?format=xml',
            'https://apnews.com/apf-worldnews?format=xml',
            'https://apnews.com/apf-business?format=xml',
            'https://apnews.com/apf-technology?format=xml',
            
            # === THỂ THAO QUỐC TẾ HOẠT ĐỘNG TỐT ===
            'https://www.espn.com/espn/rss/soccer/news',
            'https://www.goal.com/feeds/en/news',
            'https://www.espn.com/espn/rss/nba/news',
            'https://www.espn.com/espn/rss/nfl/news', 
            'https://www.espn.com/espn/rss/mlb/news',
            'https://www.espn.com/espn/rss/nhl/news',
            'https://www.espn.com/espn/rss/tennis/news',
            'https://www.espn.com/espn/rss/golf/news',
            'https://www.espn.com/espn/rss/racing/news',

            'https://www.skysports.com/rss/12040',  # Football
            'https://www.skysports.com/rss/12036',  # Cricket
            'https://www.skysports.com/rss/12148',  # F1
            'https://www.skysports.com/rss/12150',  # Rugby
            'https://www.skysports.com/rss/12158',  # Golf
            'https://www.skysports.com/rss/12154',  # Tennis
            'https://www.skysports.com/rss/12156',  # Boxing
            'https://www.eurosport.com/rss.xml',

            # === THỂ THAO CHUYÊN NGÀNH ===
            'https://www.atptour.com/en/-/media/rss-feeds/feed-news.aspx',  # Tennis
            'https://www.wtatennis.com/rss/news',  # Tennis nữ
            'https://www.pgatour.com/rss/news.rss',  # Golf
            'https://www.formula1.com/content/fom-website/en/latest/all.rss',  # F1
            'https://www.fiba.basketball/rss',  # Basketball
            'https://www.icc-cricket.com/rss/news',  # Cricket
            
            # === CÔNG NGHỆ & KHOA HỌC HOẠT ĐỘNG TỐT ===
            'https://techcrunch.com/feed/',
            'https://www.theverge.com/rss/index.xml',
            'https://www.wired.com/feed/rss',
            'https://feeds.arstechnica.com/arstechnica/index',
            'https://www.nature.com/subjects/news.rss',

            # === BÓNG ĐÁ QUỐC TẾ ===
            'https://www.premierleague.com/rss',
            'https://www.laliga.com/rss',
            'https://www.bundesliga.com/rss',
            'https://www.legaseriea.it/en/rss',
            'https://www.ligue1.com/rss',

            # === WEBSITE BÓNG ĐÁ ===
            'https://www.transfermarkt.com/rss/news',
            'https://www.90min.com/feeds/rss',
            'https://www.fourfourtwo.com/news.rss',
            'https://www.squawka.com/news/feed/',
            'https://www.planetfootball.com/feed/',
            'https://www.football365.com/rss',

            # === NEWSPAPERS ANH ===
            'https://www.dailymail.co.uk/sport/football/index.rss',
            'https://www.mirror.co.uk/sport/football/rss.xml',
            'https://www.thesun.co.uk/sport/football/feed/',
            'https://www.independent.co.uk/sport/football/rss',

            # === TIN CHUYỂN NHƯỢNG ===
            'https://www.football-italia.net/feed',
            'https://www.getfootballnewsgermany.com/feed/',
            'https://www.football-espana.net/feed',

            # === KHOA HỌC & THIÊN NHIÊN HOẠT ĐỘNG TỐT ===
            'https://feeds.bbci.co.uk/news/science_and_environment.rss.xml',
            'https://www.theguardian.com/science.rss',
            'https://apnews.com/apf-science?format=xml',
            'https://www.earthtouchnews.com/feed/',
            'https://www.worldwildlife.org/rss',
            'https://www.nationalgeographic.com/rss/animals',
            'https://www.science.org/rss/news_current.xml',
            'https://www.newscientist.com/section/news/feed/',
            'https://phys.org/rss-feed/',
            'https://www.conservation.org/rss',
            'https://www.greenpeace.org/international/rss/',
            'https://www.space.com/feeds/all',
            'https://www.nasa.gov/rss/dyn/breaking_news.rss',
            'https://www.esa.int/rssfeed/Our_Activities',
            'https://www.skyandtelescope.com/feed/',
            'https://www.sciencedaily.com/rss/top/science.xml',
            'https://www.livescience.com/feeds/all',
            'https://www.sciencenews.org/feed',
            'https://www.discovermagazine.com/rss'
        ]
        
        all_articles = []
        seen_links = set()

        print(f"Bắt đầu crawl từ {len(rss_feeds)} nguồn RSS HOẠT ĐỘNG...")

        # TĂNG GIỚI HẠN MỖI FEED
        category_limits = {
            'Tin mới nhất': 200,
            'Thời sự': 150, 
            'Thế giới': 150,
            'Kinh doanh': 100,
            'Thể thao': 150,
            'Bóng đá': 120,
            'Bóng đá quốc tế': 120,
            'Bóng đá Việt Nam': 100,
            'Giải trí': 100,
            'Công nghệ': 100,
            'Sức khỏe': 100,
            'Giáo dục': 100,
            'Đời sống': 100,
            'Du lịch': 80,
            'Pháp luật': 60,
            'Khoa học': 80,
            'Xe': 80,
            'Số hóa': 80,
            'International News': 150,
            'World News': 150, 
            'Business News': 120,
            'Technology News': 100,
            'Science News': 100,
            'Sports': 150,
            'Soccer': 150,
        }
        
        default_limit = 100

        # TẠO SESSION VỚI RETRY
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        session.mount("http://", HTTPAdapter(max_retries=retry_strategy))
        session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
        
        # CRAWL TỪNG FEED
        successful_feeds = 0
        failed_feeds = 0
        
        for i, feed_url in enumerate(rss_feeds):
            if len(all_articles) >= max_articles:
                print(f"✅ ĐÃ ĐẠT GIỚI HẠN {max_articles} BÀI BÁO!")
                break

            category = self._extract_category(feed_url)
            language = self._extract_language(feed_url)
            limit = category_limits.get(category, default_limit)
            
            if i % 10 == 0:
                print(f"[{i+1}/{len(rss_feeds)}] Đang crawl {category}... ({len(all_articles)}/{max_articles})")

            try:
                response = session.get(feed_url, timeout=15)
                feed = feedparser.parse(response.content)
                articles_from_feed = 0
                
                if not feed.entries:
                    print(f"  ⚠️ Feed trống: {feed_url}")
                    failed_feeds += 1
                    continue
                
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
                    
                    time.sleep(0.001)
                
                successful_feeds += 1
                print(f"  ✅ {category}: +{articles_from_feed} bài")
                            
            except Exception as e:
                failed_feeds += 1
                print(f"  ❌ Lỗi với feed {feed_url}: {str(e)[:100]}...")
                continue
        
        print(f"\n🎯 HOÀN THÀNH CRAWL!")
        print(f"   ✅ Feeds thành công: {successful_feeds}")
        print(f"   ❌ Feeds thất bại: {failed_feeds}")
        print(f"   📚 Tổng bài báo: {len(all_articles)}")
        
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
            'thi-truong': 'Thị trường',
            
            # BÓNG ĐÁ VIỆT NAM
            'bong-da': 'Bóng đá',
            'bong-da-viet-nam': 'Bóng đá Việt Nam',
            'bong-da-quoc-te': 'Bóng đá quốc tế',
            'tin-bong-da': 'Bóng đá',
            
            # BÓNG ĐÁ QUỐC TẾ - GIẢI ĐẤU
            'premierleague': 'Premier League',
            'laliga': 'La Liga',
            'bundesliga': 'Bundesliga', 
            'seriea': 'Serie A',
            'ligue1': 'Ligue 1',
            'championsleague': 'Champions League',
            'europaleague': 'Europa League',
            'worldcup': 'World Cup',
            
            # BÓNG ĐÁ QUỐC TẾ - WEBSITE
            'transfermarkt': 'Chuyển nhượng',
            '90min': 'Bóng đá',
            'fourfourtwo': 'Bóng đá',
            'squawka': 'Bóng đá',
            'planetfootball': 'Bóng đá',
            'football365': 'Bóng đá',
            'football-italia': 'Serie A',
            'football-espana': 'La Liga',
            'getfootballnewsgermany': 'Bundesliga',
            
            # THỂ THAO QUỐC TẾ - TỔNG HỢP
            'soccer': 'Bóng đá',
            'football': 'Bóng đá',
            'sportsnews': 'Thể thao',
            'sport': 'Thể thao',
            'sports': 'Thể thao',
            
            # THỂ THAO QUỐC TẾ - MÔN THỂ THAO
            'tennis': 'Tennis',
            'golf': 'Golf',
            'racing': 'Đua xe',
            'formula-1': 'Formula 1',
            'f1': 'Formula 1',
            'cricket': 'Cricket',
            'rugby': 'Bóng bầu dục',
            'boxing': 'Quyền anh',
            'cycling': 'Đua xe đạp',
            'basketball': 'Bóng rổ',
            'fiba': 'Bóng rổ',
            'wtatennis': 'Tennis',
            'atptour': 'Tennis',
            'pgatour': 'Golf',
            'icc-cricket': 'Cricket',
            
            # CÔNG NGHỆ & KHOA HỌC
            'tech': 'Công nghệ',
            'technology': 'Công nghệ',
            'science': 'Khoa học',
            'science_and_environment': 'Khoa học',
            'breaking_news': 'Tin mới nhất',
            
            # TIẾNG ANH - NEWS TỔNG HỢP
            'topnews': 'Tin nổi bật',
            'worldnews': 'Thế giới',
            'businessnews': 'Kinh doanh',
            'technologynews': 'Công nghệ',
            'sciencenews': 'Khoa học',
            'edition': 'Tin quốc tế',
            'edition_world': 'Thế giới',
            'edition_business': 'Kinh doanh',
            'edition_technology': 'Công nghệ',
            'edition_sport': 'Thể thao',
            'international': 'Tin quốc tế',
            'world': 'Thế giới',
            'business': 'Kinh doanh',
            'technology': 'Công nghệ',
            'science': 'Khoa học'
        }
        
        for key, value in categories.items():
            if key in feed_url.lower():
                return value
        
        # Mặc định cho các feed không xác định
        if 'news' in feed_url.lower():
            return 'Tin quốc tế'
        elif 'rss' in feed_url.lower():
            return 'Tin mới nhất'
        else:
            return 'Thế giới'

    def _extract_language(self, feed_url):
        """Phân loại ngôn ngữ"""
        vietnamese_domains = ['vnexpress', 'dantri', 'thanhnien', 'tuoitre', '24h', 'bongdaplus', 
                            'webthethao', 'thethao247', 'laodong', 'vietnamnet', 'zingnews',
                            'cafef', 'vtv']
        english_domains = ['espn', 'skysports', 'goal', 'eurosport', 'bbc', 'theguardian', 
                        'reuters', 'cnn', 'apnews', 'techcrunch', 'theverge', 'wired',
                        'arstechnica', 'nasa', 'nature', 'premierleague', 'laliga', 'bundesliga',
                        'legaseriea', 'ligue1', 'transfermarkt', '90min', 'fourfourtwo', 'squawka',
                        'planetfootball', 'football365', 'dailymail', 'mirror', 'thesun', 'independent',
                        'fifa', 'uefa', 'nba', 'nfl', 'mlb', 'nhl', 'atptour', 'wtatennis', 'pgatour',
                        'formula1', 'fiba', 'icc-cricket', 'football-italia', 'getfootballnewsgermany',
                        'football-espana', 'science.org', 'newscientist', 'phys.org', 'space.com',
                        'esa.int', 'skyandtelescope', 'sciencedaily', 'livescience', 'sciencenews',
                        'discovermagazine', 'nationalgeographic', 'earthtouchnews', 'worldwildlife',
                        'conservation', 'greenpeace']
        
        if any(domain in feed_url for domain in vietnamese_domains):
            return 'Vietnamese'
        elif any(domain in feed_url for domain in english_domains):
            return 'English'
        else:
            return 'Other'

    def _extract_source(self, feed_url):
        sources = {
            # BÁO VIỆT NAM
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
            'cafef': 'Cafef',
            'vtv': 'VTV',
            
            # THỂ THAO QUỐC TẾ - TỔNG HỢP
            'espn': 'ESPN',
            'skysports': 'Sky Sports',
            'goal': 'Goal.com',
            'eurosport': 'Eurosport',
            'reuters': 'Reuters',
            'cnn': 'CNN',
            'bbc': 'BBC',
            'theguardian': 'The Guardian',
            'apnews': 'Associated Press',
            
            # GIẢI ĐẤU BÓNG ĐÁ CHÍNH THỐNG
            'premierleague': 'Premier League',
            'laliga': 'La Liga',
            'bundesliga': 'Bundesliga',
            'legaseriea': 'Serie A', 
            'ligue1': 'Ligue 1',
            
            # WEBSITE BÓNG ĐÁ
            'transfermarkt': 'Transfermarkt',
            '90min': '90min',
            'fourfourtwo': 'FourFourTwo',
            'squawka': 'Squawka',
            'planetfootball': 'Planet Football',
            'football365': 'Football365',
            'football-italia': 'Football Italia',
            'getfootballnewsgermany': 'Get German Football News',
            'football-espana': 'Football Espana',
            
            # NEWSPAPERS ANH
            'dailymail': 'Daily Mail',
            'mirror': 'Daily Mirror',
            'thesun': 'The Sun',
            'independent': 'The Independent',
            
            # THỂ THAO CHUYÊN NGÀNH
            'atptour': 'ATP Tour',
            'wtatennis': 'WTA Tennis',
            'pgatour': 'PGA Tour',
            'formula1': 'Formula 1',
            'fiba': 'FIBA Basketball',
            'icc-cricket': 'ICC Cricket',
            
            # CÔNG NGHỆ & KHOA HỌC
            'techcrunch': 'TechCrunch',
            'theverge': 'The Verge',
            'wired': 'Wired',
            'arstechnica': 'Ars Technica',
            'nasa': 'NASA',
            'nature': 'Nature',

            # THẾ GIỚI ĐỘNG VẬT & THIÊN NHIÊN
            'nationalgeographic': 'National Geographic',
            'bbcearth': 'BBC Earth',
            'earthtouchnews': 'Earth Touch News',
            'worldwildlife': 'WWF',
            
            # KHOA HỌC
            'science.org': 'Science Magazine',
            'newscientist': 'New Scientist',
            'phys.org': 'Phys.org',
            'sciencedaily': 'Science Daily',
            
            # BẢO TỒN & MÔI TRƯỜNG
            'conservation': 'Conservation International',
            'greenpeace': 'Greenpeace',
            
            # KHOA HỌC VŨ TRỤ
            'space.com': 'Space.com',
            'nasa': 'NASA',
            'esa.int': 'European Space Agency',
            'skyandtelescope': 'Sky & Telescope',
            
            # KHOA HỌC ĐỜI SỐNG
            'livescience': 'Live Science',
            'sciencenews': 'Science News',
            'discovermagazine': 'Discover Magazine'
        }
        
        # Xử lý các domain phức tạp
        if 'feeds.reuters.com' in feed_url:
            return 'Reuters'
        elif 'feeds.bbci.co.uk' in feed_url:
            return 'BBC'
        elif 'rss.cnn.com' in feed_url:
            return 'CNN'
        elif 'apf-science' in feed_url:
            return 'Associated Press'
        elif 'science_and_environment' in feed_url:
            return 'BBC Science'
        
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
        max_articles = int(input("Số bài báo muốn crawl (mặc định 10000): ").strip() or "10000")
    except:
        max_articles = 10000
    
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