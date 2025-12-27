🔍 HỆ THỐNG TÌM KIẾM BÀI BÁO SỬ DỤNG HNSW

(Hierarchical Navigable Small World Graph)

1. Giới thiệu

Trong bối cảnh số lượng bài báo trực tuyến ngày càng tăng nhanh, việc tìm kiếm thông tin liên quan một cách hiệu quả là một bài toán quan trọng. Dự án này triển khai một hệ thống tìm kiếm bài báo dựa trên độ tương đồng ngữ nghĩa, sử dụng mô hình embedding văn bản kết hợp với HNSW (Hierarchical Navigable Small World) để thực hiện truy vấn gần đúng (Approximate Nearest Neighbor – ANN) với tốc độ cao.

Hệ thống cho phép:

Thu thập (crawl) dữ liệu bài báo từ các nguồn tin tức

Biểu diễn nội dung bài báo dưới dạng vector embedding

Xây dựng chỉ mục HNSW để tìm kiếm nhanh

Cung cấp giao diện web cho người dùng truy vấn

2. Kiến trúc tổng thể hệ thống

Hệ thống gồm 4 thành phần chính:

Thu thập dữ liệu (Crawler)
Crawl các bài báo từ nguồn online và lưu trữ nội dung cùng metadata.

Sinh embedding văn bản
Sử dụng mô hình Sentence Transformer để ánh xạ bài báo sang không gian vector.

Xây dựng & quản lý chỉ mục HNSW
Áp dụng thuật toán HNSW để lưu trữ và truy vấn vector embedding hiệu quả.

Web Application
Backend viết bằng Python (FastAPI), frontend HTML/JS cho phép người dùng tìm kiếm bài báo theo truy vấn tự nhiên.

3. Cấu trúc thư mục
project/
├── templates/
│   └── index.html
│   └── script.js
├── article_embedder.py
├── article_search_system.py
├── crawl_articles.py
├── hnsw_manager.py
├── graph.py
├── server.py
├── merge_article_index.py
├── update_summary_data.py
└── README.md

Mô tả ngắn gọn các file

crawl_articles.py: Thu thập và lưu trữ dữ liệu bài báo

article_embedder.py: Sinh embedding cho văn bản

hnsw_manager.py: Xây dựng và quản lý chỉ mục HNSW

article_search_system.py: Thực hiện truy vấn tìm kiếm

server.py: Backend FastAPI

graph.py: Mô phỏng cấu trúc đồ thị HNSW

merge_article_index.py: Gộp và cập nhật chỉ mục

update_summary_data.py: Cập nhật metadata và thống kê

templates/index.html: Giao diện web

4. Công nghệ sử dụng

Python 3

Sentence-Transformers

HNSWlib

FastAPI

Uvicorn

HTML / JavaScript

5. Cài đặt môi trường

Cài đặt các thư viện cần thiết:

pip install feedparser==6.0.10
pip install requests==2.32.4
pip install huggingface_hub>=0.24.0
pip install sentence-transformers>=3.0.0
pip install hnswlib==0.7
pip install "fastapi>=0.115.2,<1.0"
pip install "starlette>=0.49.1,<1.0"
pip install "anyio>=4.9.0,<5.0"
pip install "uvicorn>=0.34.0,<1.0"
pip install python-multipart>=0.0.18

6. Xây dựng dữ liệu và chỉ mục
Bước 1: Crawl bài báo
python crawl_articles.py


Kết quả:

Thư mục article.data/: chứa nội dung bài báo

File metadata phục vụ embedding

Bước 2: Sinh embedding & xây dựng HNSW
python hnsw_manager.py


Kết quả:

Thư mục article.index/: chỉ mục HNSW

File embeddings.npy: vector embedding

7. Chạy hệ thống web
python server.py


Truy cập trên trình duyệt:

http://localhost:8000/

8. Hướng dẫn sử dụng

Người dùng nhập truy vấn tìm kiếm

Hệ thống sinh embedding cho truy vấn

Chỉ mục HNSW tìm các vector gần nhất

Trả về danh sách bài báo liên quan theo độ tương đồng

9. Demo & Tài nguyên
🌐 Live Demo (GitHub Pages)

👉 https://vp2802.github.io/hnsw-group7/

📘 Google Colab (chạy sẵn)

👉 https://colab.research.google.com/drive/1iWQEyGi5aBXxDRD09-qgvT7lF-CNjnDB?usp=sharing

Notebook Colab đã được cấu hình sẵn, có thể chạy trực tiếp không cần cài môi trường.

10. Ghi chú

Khi cập nhật dữ liệu bài báo, cần rebuild embedding và HNSW index

HNSW cho phép đánh đổi chính xác – tốc độ thông qua các tham số (M, ef)

Phù hợp cho hệ thống tìm kiếm quy mô lớn

11. Kết luận

Dự án đã chứng minh hiệu quả của HNSW trong bài toán tìm kiếm ngữ nghĩa với dữ liệu văn bản. Giải pháp có khả năng mở rộng tốt, tốc độ truy vấn nhanh và dễ tích hợp vào các hệ thống tìm kiếm thực tế.
