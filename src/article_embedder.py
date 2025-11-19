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
        if not text:
            return ""
        
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ_,.!?;:()\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def prepare_article_text(self, article):
        title = self.preprocess_text(article.get('title', ''))
        summary = self.preprocess_text(article.get('summary', ''))
        content = self.preprocess_text(article.get('content', ''))
        
        if content and len(content) > 100:
            main_text = content
        elif summary:
            main_text = summary
        else:
            main_text = title
        
        if title and main_text != title:
            text_to_embed = f"{title}. {main_text}"
        else:
            text_to_embed = main_text
        
        if len(text_to_embed) > 2000:
            text_to_embed = text_to_embed[:2000]
        
        return text_to_embed
    
    def embed_articles(self, articles):
        print(f"Đang embed {len(articles)} bài báo...")
        
        texts = []
        valid_articles = []
        
        for i, article in enumerate(articles):
            text = self.prepare_article_text(article)
            if text and len(text) > 10:
                texts.append(text)
                valid_articles.append(article)
        
        print(f"  Số bài báo hợp lệ: {len(valid_articles)}/{len(articles)}")
        
        print("  Đang tạo embeddings...")
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        
        print(f"Embedding hoàn thành: {len(embeddings)} vectors")
        return valid_articles, embeddings.astype(np.float32)
    
    def embed_query(self, query):
        processed_query = self.preprocess_text(query)
        print(f"Query: '{query}' -> '{processed_query}'")
        
        embedding = self.model.encode([processed_query])[0].astype(np.float32)
        return embedding.reshape(1, -1)