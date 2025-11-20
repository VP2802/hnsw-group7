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