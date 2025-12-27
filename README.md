# Hệ thống Tìm kiếm Bài báo Semantic Search với HNSW

Dự án triển khai hệ thống tìm kiếm bài báo thông minh dựa trên ngữ nghĩa (Semantic Search), sử dụng thuật toán **HNSW (Hierarchical Navigable Small World)** để tối ưu hóa tốc độ truy vấn trên không gian vector cao chiều.

---

## Cấu trúc thư mục (Project Structure)

```text
project/
├── src/
│   ├── crawl_articles.py       # Thu thập dữ liệu từ hơn 30+ nguồn RSS
│   ├── article_embedder.py     # Sinh Embedding bằng Vietnamese-SBERT (768 dims)
│   ├── hnsw_manager.py         # Xây dựng và quản lý chỉ mục HNSW
│   ├── article_search_system.py # Xử lý logic tìm kiếm (Semantic/Keyword/Hybrid)
│   ├── server.py               # Backend FastAPI
│   ├── merge_article_index.py  # Công cụ gộp các chỉ mục dữ liệu
│   ├── update_summary_data.py  # Cập nhật metadata và thống kê hệ thống
│   └── graph.py                # Trực quan hóa cấu trúc đồ thị HNSW
├── templates/
│   └── index.html              # Giao diện người dùng (Frontend)
├── article_index/              # Lưu trữ dữ liệu chỉ mục (.bin, .npy, .json)
├── article_data/               # Lưu trữ nội dung bài báo thô (.json)
├── visualization.py            # Phân tích và hiển thị biểu đồ kết quả
├── requirements.txt            # Danh sách thư viện cần thiết
└── README.md                   # Tài liệu hướng dẫn

```

# Chức năng chính
Crawl dữ liệu tự động: Hỗ trợ hơn 30 nguồn báo uy tín, tự động phân loại ngôn ngữ (Tiếng Việt ~80%, Tiếng Anh ~20%).

Xử lý Ngôn ngữ Tự nhiên (NLP): Sử dụng mô hình Vietnamese-SBERT để chuyển đổi văn bản thành vector đặc trưng.

Tìm kiếm xấp xỉ (ANN): Sử dụng HNSW giúp tìm kiếm hàng nghìn bài báo trong thời gian mili giây.

Đa chế độ tìm kiếm:

Semantic Search: Tìm theo ý nghĩa nội dung.

Keyword Search: Tìm theo từ khóa truyền thống.

Hybrid Search: Kết hợp cả hai phương pháp để tăng độ chính xác.

Giao diện trực quan: Web interface thân thiện, phản hồi nhanh (~135ms).

# Cài đặt

Yêu cầu Python 3.8+

```text
pip install feedparser==6.0.10 requests==2.32.4
pip install huggingface_hub>=0.24.0 sentence-transformers>=3.0.0
pip install hnswlib==0.7
pip install fastapi==0.115.2 uvicorn==0.34.0
pip install python-multipart
```

# Hướng dẫn cài đặt

### 1. Cài đặt thư viện phụ thuộc
Yêu cầu **Python 3.8+**. Chạy lệnh sau để cài đặt các package cần thiết:

```bash
pip install feedparser==6.0.10 requests==2.32.4
pip install huggingface_hub>=0.24.0 sentence-transformers>=3.0.0
pip install hnswlib==0.7
pip install fastapi==0.115.2 uvicorn==0.34.0
pip install python-multipart
```
2. Xây dựng dữ liệu (Pipeline)
Nếu bạn chạy dự án từ đầu, thực hiện theo thứ tự:
Thu thập dữ liệu:
```Bash
python src/crawl_articles.py
```
Sinh Embedding và xây dựng chỉ mục HNSW:
```Bash
python src/hnsw_manager.py
```
3. Khởi chạy hệ thống
Chạy lệnh sau để khởi động Web Server:
```Bash
python src/server.py
```
Sau đó truy cập địa chỉ: http://localhost:8000


# Thông tin Dataset
Quy mô: 8,661 bài báo.

Ngôn ngữ:

Tiếng Việt: 6,884 bài (79.5%)

Tiếng Anh: 1,777 bài (20.5%)

Chủ đề: Thế giới, Thể thao, Công nghệ, Giáo dục, Kinh doanh, Sức khỏe...

# Liên kết tham khảo
# Bản Demo Web: [GitHub Pages](https://vp2802.github.io/hnsw-group7/)

# Thử nghiệm trực tuyến: [Google Colab](https://l.facebook.com/l.php?u=https%3A%2F%2Fcolab.research.google.com%2Fdrive%2F1iWQEyGi5aBXxDRD09-qgvT7lF-CNjnDB%3Fusp%3Dsharing%26fbclid%3DIwZXh0bgNhZW0CMTAAYnJpZBExb2d4NlprVGY0bFlXM1pIanNydGMGYXBwX2lkEDIyMjAzOTE3ODgyMDA4OTIAAR57GBEBcIr5eqmNaGZ0yr-tdFr0StTeA1Ab339Tf0_wLxij6P-zCQRt4VuQTQ_aem_YYHyPRrL4B8OY6AQb1Tymg&h=AT2l7o3dErmF_vDnALnVQ4JcWzVvYseKj07JoUDR4jZpuBHHq9P2gt7FIDIPDdoB1mINVb00IH3oBIUSXLFwWqCeaUTubxyfLkvwgyDoai_LkI_uM18QArTd9eBksZXsRHPW3RH8bzhIYL52Ax28jQ)

# Ghi chú
Khi thực hiện cập nhật dữ liệu mới (Crawl thêm bài báo), bạn bắt buộc phải chạy lại script hnsw_manager.py để cập nhật lại không gian vector và cấu trúc đồ thị HNSW, đảm bảo kết quả tìm kiếm luôn mới nhất.
