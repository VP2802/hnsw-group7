🔍 Hệ thống Tìm kiếm Bài báo bằng HNSW

Dự án này triển khai một công cụ tìm kiếm bài báo theo ngữ nghĩa sử dụng HNSW (Hierarchical Navigable Small World) để thực hiện truy vấn tương đồng nhanh và hiệu quả.

Hệ thống bao gồm các thành phần:

Crawl (thu thập) bài báo từ nhiều nguồn RSS

Sinh embedding văn bản bằng mô hình ngôn ngữ

Xây dựng chỉ mục HNSW cho tìm kiếm xấp xỉ

Cung cấp giao diện web hỗ trợ nhiều chế độ tìm kiếm

📁 Cấu trúc thư mục
project/
├── src/
│   ├── crawl_articles.py
│   ├── article_embedder.py
│   ├── hnsw_manager.py
│   ├── article_search_system.py
│   ├── server.py
│   ├── merge_article_index.py
│   ├── update_summary_data.py
│   └── graph.py
├── templates/
│   └── index.html
├── article_index/
│   ├── article_index.bin
│   ├── embeddings.npy
│   ├── metadata.json
│   └── benchmark_results.json
├── article_data/
│   ├── articles.json
│   └── summary.txt
├── index.html
├── visualization.py
├── requirements.txt
└── README.md

🚀 Chức năng chính

Crawl tin tức
Thu thập bài báo từ 30+ nguồn báo online, hỗ trợ đa ngôn ngữ

Tiếng Việt: 79.5%

Tiếng Anh: 20.5%

Sinh embedding
Sử dụng Vietnamese-SBERT (768 chiều) để biểu diễn ngữ nghĩa bài báo

Chỉ mục HNSW
Xây dựng chỉ mục HNSW cho bài toán Approximate Nearest Neighbor Search

Giao diện web
Hỗ trợ 3 chế độ tìm kiếm:

Semantic Search (ngữ nghĩa)

Keyword Search (từ khóa)

Hybrid Search (kết hợp)

Backend
FastAPI (Python) với thời gian phản hồi trung bình ~135ms

⭐ Kết quả nổi bật

8,661 bài báo từ 30+ nguồn RSS

20 chủ đề đa dạng: Thế giới, Thể thao, Công nghệ, Giáo dục, …

Hiệu năng vượt trội:

HNSW nhanh hơn 56× so với Brute Force

Thời gian tìm kiếm chỉ tăng 17.9% khi dữ liệu tăng 5 lần

Độ chính xác cao:

3 truy vấn đạt 100% similarity

7/10 truy vấn có similarity > 0.85

⚙️ Cài đặt
1. Cài đặt thư viện
pip install feedparser==6.0.10 requests==2.32.4
pip install huggingface_hub>=0.24.0 sentence-transformers>=3.0.0
pip install hnswlib==0.7
pip install fastapi==0.115.2 uvicorn==0.34.0
pip install python-multipart


Yêu cầu: Python 3.8+

2. Đảm bảo cấu trúc thư mục

Cấu trúc thư mục cần giống như mục Cấu trúc thư mục ở trên.

🏗️ Xây dựng dữ liệu

Nếu chưa có article_data/ và article_index/, cần thực hiện các bước sau.

Bước 1: Crawl bài báo
python src/crawl_articles.py


Kết quả:

Thư mục article_data/ chứa nội dung bài báo

File metadata phục vụ bước embedding

Bước 2: Sinh embedding & build HNSW
python src/hnsw_manager.py


Kết quả:

Thư mục article_index/ chứa chỉ mục HNSW

File embeddings.npy chứa vector embedding

🌐 Chạy Web Server
python src/server.py


Mở trình duyệt và truy cập:

http://localhost:8000

🧑‍💻 Cách sử dụng

Nhập truy vấn tìm kiếm vào ô tìm kiếm

Chọn chế độ:

Semantic: tìm kiếm theo ngữ nghĩa

Keyword: tìm kiếm theo từ khóa

Hybrid: kết hợp cả hai

Chọn số lượng kết quả (TopK) và cách sắp xếp

Hệ thống sinh embedding cho truy vấn và tìm các bài báo tương tự

Trả về danh sách bài báo kèm độ tương đồng

📄 Mô tả các file chính

src/crawl_articles.py
Crawl bài báo từ RSS feeds và lưu dữ liệu

src/article_embedder.py
Sinh embedding văn bản bằng Vietnamese-SBERT

src/article_search_system.py
Tìm kiếm bài báo với 3 chế độ: Semantic / Keyword / Hybrid

src/hnsw_manager.py
Xây dựng và quản lý chỉ mục HNSW

src/server.py
Backend FastAPI

src/merge_article_index.py
Gộp và cập nhật chỉ mục

src/update_summary_data.py
Cập nhật metadata và thống kê dữ liệu

src/graph.py
Trực quan hóa cấu trúc đồ thị HNSW

visualization.py
Phân tích kết quả và hiển thị biểu đồ

templates/index.html
Giao diện web phía người dùng

📊 Dataset thống kê

Tổng số bài báo: 8,661

Tiếng Việt: 6,884 bài (79.5%)

Tiếng Anh: 1,777 bài (20.5%)

Nguồn báo: 30+ nguồn uy tín

Chủ đề: 20 chủ đề đa dạng

⚡ Hiệu suất hệ thống

Thời gian tìm kiếm: ~135 ms

Recall: > 97.5%

Cải thiện tốc độ: nhanh hơn 56× so với Brute Force

🔗 Live Demo
🌐 GitHub Pages

🖥️ Google Colab (chạy sẵn)

📝 Ghi chú

Khi cập nhật danh sách bài báo, cần build lại embedding và HNSW index

Hệ thống hỗ trợ tìm kiếm đa ngôn ngữ, đa chủ đề

Có thể mở rộng cho các hệ thống tìm kiếm quy mô lớn
