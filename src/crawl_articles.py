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
            # Báo Tieng Viet
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
            
            # Bao The Thao
            'https://www.24h.com.vn/rss/tin-bong-da.rss',
            'https://bongdaplus.vn/rss/bong-da-viet-nam.rss',
            'https://bongdaplus.vn/rss/bong-da-quoc-te.rss',
            'https://thethao247.vn/rss/tin-bong-da.rss',
            'https://webthethao.vn/rss/bong-da.rss',
            
            # Bao Quoc Te
            'https://feeds.reuters.com/reuters/topNews',
            'https://feeds.reuters.com/reuters/worldNews',
            'https://feeds.reuters.com/reuters/businessNews',
            'https://feeds.reuters.com/reuters/technologyNews',
            'https://feeds.reuters.com/reuters/scienceNews',
            
            'https://rss.cnn.com/rss/edition.rss',
            'https://rss.cnn.com/rss/edition_world.rss',
            'https://rss.cnn.com/rss/edition_business.rss',
            'https://rss.cnn.com/rss/edition_technology.rss',
            
            'https://feeds.bbci.co.uk/news/rss.xml',
            'https://feeds.bbci.co.uk/news/world/rss.xml',
            'https://feeds.bbci.co.uk/news/business/rss.xml',
            'https://feeds.bbci.co.uk/news/technology/rss.xml',
            'https://feeds.bbci.co.uk/news/science_and_environment.rss.xml',
            
            'https://www.theguardian.com/international/rss',
            'https://www.theguardian.com/world/rss',
            'https://www.theguardian.com/business.rss',
            'https://www.theguardian.com/technology.rss',
            'https://www.theguardian.com/science.rss',
            
            'https://apnews.com/apf-topnews?format=xml',
            'https://apnews.com/apf-worldnews?format=xml',
            'https://apnews.com/apf-business?format=xml',
            'https://apnews.com/apf-technology?format=xml',
            
            # The Thao Quoc Te
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

            # Thể thao chuyên ngành
            'https://www.atptour.com/en/-/media/rss-feeds/feed-news.aspx',  # Tennis
            'https://www.wtatennis.com/rss/news',  # Tennis nữ
            'https://www.pgatour.com/rss/news.rss',  # Golf
            'https://www.formula1.com/content/fom-website/en/latest/all.rss',  # F1
            'https://www.fiba.basketball/rss',  # Basketball
            'https://www.icc-cricket.com/rss/news',  # Cricket
            
            # Cong Nghe & Khoa Hoc
            'https://techcrunch.com/feed/',
            'https://www.theverge.com/rss/index.xml',
            'https://www.wired.com/feed/rss',
            'https://feeds.arstechnica.com/arstechnica/index',
            'https://www.nature.com/subjects/news.rss',

            # Bong da Quoc Te - Chuyen sau
            'https://www.premierleague.com/rss',  # Premier League chinh thong
            'https://www.laliga.com/rss',  # La Liga chinh thong
            'https://www.bundesliga.com/rss',  # Bundesliga chinh thong
            'https://www.legaseriea.it/en/rss',  # Serie A chinh thong
            'https://www.ligue1.com/rss',  # Ligue 1 chinh thong

            # Cac website bong da noi tieng
            'https://www.transfermarkt.com/rss/news',
            'https://www.90min.com/feeds/rss',
            'https://www.fourfourtwo.com/news/rss',
            'https://www.squawka.com/news/feed/',
            'https://www.planetfootball.com/feed/',
            'https://www.football365.com/rss',

            # Tin tuc bong da tu cac newspaper lon
            'https://www.dailymail.co.uk/sport/football/index.rss',
            'https://www.mirror.co.uk/sport/football/rss.xml',
            'https://www.thesun.co.uk/sport/football/feed/',
            'https://www.independent.co.uk/sport/football/rss',

            # Tin chuyen nhuong
            'https://www.football-italia.net/feed',
            'https://www.getfootballnewsgermany.com/feed/',
            'https://www.football-espana.net/feed'

            # The gioi dong vat & Khoa hoc Tu nhien
            'https://feeds.bbci.co.uk/news/science_and_environment/rss.xml',  # BBC Science
            'https://www.theguardian.com/science/rss',  # Guardian Science
            'https://apnews.com/apf-science?format=xml',  # AP Science

            # Dong vat & Thien nhien
            'https://www.bbc.com/earth/rss.xml',  # BBC Earth
            'https://www.earthtouchnews.com/feed/',  # Earth Touch News
            'https://www.worldwildlife.org/rss',  # WWF (thêm vào cho đa dạng)
            'https://www.nationalgeographic.com/rss/animals',  # National Geographic Animals (thêm)

            # Khoa hoc & Khám phá
            'https://www.science.org/rss/news_current.xml',  # Science Magazine
            'https://www.newscientist.com/section/news/feed/',  # New Scientist
            'https://phys.org/rss-feed/',  # Phys.org

            # Bao ton & Moi truong
            'https://www.conservation.org/rss',  # Conservation International
            'https://www.greenpeace.org/international/rss/',  # Greenpeace

            # Khoa hoc Vu tru & Thien van
            'https://www.space.com/feeds/all',  # Space.com
            'https://www.nasa.gov/rss/dyn/breaking_news.rss',  # NASA News
            'https://www.esa.int/rssfeed/Our_Activities',  # European Space Agency
            'https://www.skyandtelescope.com/feed/',  # Sky & Telescope

            # Khoa hoc Doi song
            'https://www.sciencedaily.com/rss/top/science.xml',  # Science Daily Top
            'https://www.livescience.com/feeds/all',  # Live Science
            'https://www.sciencenews.org/feed',  # Science News
            'https://www.discovermagazine.com/rss',  # Discover Magazine
        ]
        
        all_articles = []
        seen_links = set()
        
        print(f"Bắt đầu crawl từ {len(rss_feeds)} nguồn RSS...")
        
        # Định nghĩa số bài tối đa cho từng loại chuyên mục
        category_limits = {
            # CÁC CHUYÊN MỤC ÍT QUAN TRỌNG: 30-50 bài/feed
            'Sức khỏe': 50,
            'Kinh doanh': 50, 
            'Đời sống': 50,
            'Giáo dục': 50,
            'Kinh tế': 50,
            'Văn hóa': 50,
            'Giới trẻ': 50,
            'Pháp luật': 30,
            'Khoa học': 50,
            'Số hóa': 50,
            'Xe': 50,
            
            # CÁC CHUYÊN MỤC QUAN TRỌNG: 70-100 bài/feed
            'Thể thao': 100,
            'Bóng đá': 100,
            'Bóng đá quốc tế': 100,
            'Bóng đá Việt Nam': 80,
            'Tin mới nhất': 80,
            'Thời sự': 100,
            'Thế giới': 100,
            'Công nghệ': 70,
            'Giải trí': 70,
            'Du lịch': 70,
            'Thị trường': 70,

            # BÓNG ĐÁ QUỐC TẾ - GIẢI ĐẤU: 80-120 bài/feed
            'Premier League': 120,
            'La Liga': 100,
            'Bundesliga': 100,
            'Serie A': 100,
            'Ligue 1': 100,
            'Champions League': 80,
            'Europa League': 80,
            'World Cup': 80,
            'Chuyển nhượng': 60,

            # THỂ THAO QUỐC TẾ - MÔN THỂ THAO: 60-100 bài/feed
            'Bóng rổ': 80,
            'Bóng bầu dục Mỹ': 60,
            'Bóng chày': 60,
            'Khúc côn cầu': 60,
            'Tennis': 80,
            'Golf': 60,
            'Đua xe': 60,
            'Formula 1': 70,
            'Cricket': 60,
            'Bóng bầu dục': 60,
            'Quyền anh': 50,
            'Đua xe đạp': 50,
            
            # INTERNATIONAL NEWS: 70-120 bài/feed
            'International News': 120,
            'World News': 120,
            'Top Stories': 100,
            'Business News': 100,
            'Technology News': 100,
            'Science News': 100,
            'Sports': 120,
            'Soccer': 120,
            'Football': 120,

            # TIN QUỐC TẾ TỔNG HỢP: 80-100 bài/feed
            'Tin quốc tế': 100,
            'Tin nổi bật': 100,

            # THIÊN NHIÊN & KHOA HỌC: 30-70 bài/feed
            'Động vật': 50,
            'Thiên nhiên': 40,
            'Môi trường': 35,
            'Khoa học & Môi trường': 45,
            'Khoa học': 70,
            'Bảo tồn': 30,
            'Khám phá vũ trụ': 40,
            'Thiên văn': 30,
            'Khoa học đời sống': 50,
            'Tin khoa học': 60,
            'Khám phá': 40
        }
        
        # Crawl từng feed với giới hạn khác nhau
        for feed_url in rss_feeds:
            if len(all_articles) >= max_articles:
                break

            category = self._extract_category(feed_url)
            language = self._extract_language(feed_url)
            limit = category_limits.get(category, 50)
            
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
                    
                    time.sleep(0.03)
                        
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
        max_articles = int(input("Số bài báo muốn crawl (mặc định 5000): ").strip() or "10000")
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