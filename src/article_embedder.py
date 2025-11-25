import numpy as np
from sentence_transformers import SentenceTransformer
import re

class ArticleEmbedder:
    def __init__(self):
        print("Đang tải model embedding tiếng Việt...")
        self.model = SentenceTransformer('keepitreal/vietnamese-sbert')
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded: {self.dim} dimensions")
    
    def preprocess_text(self, text):
        """Làm sạch và chuẩn hóa văn bản"""
        if not text:
            return ""
        
        # Loại bỏ HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Loại bỏ URLs
        text = re.sub(r'http\S+', '', text)
        # Giữ lại ký tự tiếng Việt và dấu câu cơ bản
        text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ_,.!?;:()\-]', ' ', text)
        # Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def is_source_query(self, query):
        """Kiểm tra xem query có phải là tìm kiếm theo nguồn báo không"""
        source_keywords = [
            'báo', 'trang', 'nguồn', 'từ', 'của', 'trên',
            'vnexpress', 'dân trí', 'thanh niên', 'tuổi trẻ', 
            'lao động', 'vietnamnet', 'zingnews', '24h',
            'bbc', 'reuters', 'cnn', 'the guardian', 'espn'
        ]
        
        query_lower = query.lower() 
        
        # Kiểm tra nếu query chứa từ khóa chỉ nguồn báo
        for keyword in source_keywords:
            if keyword in query_lower:
                return True
        
        # Kiểm tra nếu query có dạng "báo [tên báo]" hoặc "[tên báo]"
        common_sources = ['vnexpress', 'dantri', 'thanhnien', 'tuoitre', 'laodong', 
                         'vietnamnet', 'zingnews', '24h', 'bbc', 'reuters', 'cnn']
        
        for source in common_sources:
            if source in query_lower:
                return True
        
        return False
    
    def extract_source_from_query(self, query):
        """Trích xuất tên nguồn báo từ query"""
        query_lower = query.lower()
        
        # Map từ viết tắt/thông dụng sang tên đầy đủ
        source_mapping = {
            'vnexpress': 'VnExpress',
            'dantri': 'Dân Trí',
            'dân trí': 'Dân Trí',
            'thanhnien': 'Thanh Niên',
            'thanh niên': 'Thanh Niên',
            'tuoitre': 'Tuổi Trẻ',
            'tuổi trẻ': 'Tuổi Trẻ',
            'laodong': 'Lao Động',
            'lao động': 'Lao Động',
            'vietnamnet': 'VietnamNet',
            'zingnews': 'ZingNews',
            'zing news': 'ZingNews',
            '24h': '24h.com.vn',
            '24h.com.vn': '24h.com.vn',
            'bbc': 'BBC',
            'reuters': 'Reuters',
            'cnn': 'CNN',
            'the guardian': 'The Guardian',
            'guardian': 'The Guardian',
            'espn': 'ESPN',
            'sky sports': 'Sky Sports',
            'skysports': 'Sky Sports',
            'goal': 'Goal.com',
            'goal.com': 'Goal.com'
        }
        
        # Tìm tên nguồn trong query
        for source_key, source_name in source_mapping.items():
            if source_key in query_lower:
                return source_name
        
        # Nếu không tìm thấy, thử trích xuất sau từ "báo"
        if 'báo' in query_lower:
            parts = query_lower.split('báo')
            if len(parts) > 1:
                potential_source = parts[1].strip()
                for source_key, source_name in source_mapping.items():
                    if source_key in potential_source:
                        return source_name
        
        return None
    
    def prepare_article_text(self, article):
        """Chuẩn bị văn bản để embed từ thông tin bài báo"""
        title = self.preprocess_text(article.get('title', ''))
        summary = self.preprocess_text(article.get('summary', ''))
        
        # Ưu tiên: title + summary (nếu có)
        if title and summary:
            text_to_embed = f"{title}. {summary}"
        elif title:
            text_to_embed = title
        elif summary:
            text_to_embed = summary
        else:
            return ""
        
        # Giới hạn độ dài để tránh quá dài
        if len(text_to_embed) > 2000:
            text_to_embed = text_to_embed[:2000]
        
        return text_to_embed
    
    def embed_articles(self, articles):
        """Embed danh sách bài báo"""
        print(f"Đang embed {len(articles)} bài báo...")
        
        texts = []
        valid_articles = []
        
        for i, article in enumerate(articles):
            text = self.prepare_article_text(article)
            if text and len(text) > 10:  # Đảm bảo text có ý nghĩa
                texts.append(text)
                valid_articles.append(article)
        
        print(f"  Số bài báo hợp lệ: {len(valid_articles)}/{len(articles)}")
        
        if not texts:
            print("  Không có văn bản hợp lệ để embed!")
            return [], np.array([])
        
        print("  Đang tạo embeddings...")
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=True, 
            batch_size=32,
            normalize_embeddings=True  # Chuẩn hóa để tính cosine similarity chính xác
        )
        
        print(f"Embedding hoàn thành: {len(embeddings)} vectors")
        return valid_articles, embeddings.astype(np.float32)
    
    def embed_query(self, query):
        """Embed câu query tìm kiếm"""
        processed_query = self.preprocess_text(query)
        print(f"Query: '{query}' -> '{processed_query}'")
        
        embedding = self.model.encode(
            [processed_query],
            normalize_embeddings=True
        )[0].astype(np.float32)
        
        return embedding.reshape(1, -1)
    
    def analyze_query(self, query):
        """
        Phân tích query để xác định loại tìm kiếm
        Trả về: {'type': 'content'|'source'|'mixed', 'source_name': str hoặc None, 'content_query': str}
        """
        if self.is_source_query(query):
            source_name = self.extract_source_from_query(query)
            if source_name:
                # Nếu query chỉ có tên báo, không có nội dung khác
                query_without_source = query.lower()
                source_keywords = ['báo', 'trang', 'nguồn', 'từ', 'của']
                
                for keyword in source_keywords + [source_name.lower()]:
                    query_without_source = query_without_source.replace(keyword, '')
                
                query_without_source = query_without_source.strip()
                
                if not query_without_source or len(query_without_source) < 3:
                    return {
                        'type': 'source',
                        'source_name': source_name,
                        'content_query': None
                    }
                else:
                    return {
                        'type': 'mixed',
                        'source_name': source_name,
                        'content_query': query_without_source.strip()
                    }
        
        return {
            'type': 'content',
            'source_name': None,
            'content_query': query
        }