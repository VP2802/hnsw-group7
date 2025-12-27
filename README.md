HỆ THỐNG TÌM KIẾM BÀI BÁO SỬ DỤNG HNSW

(Hierarchical Navigable Small World Graph)

1. Giới thiệu

Trong bối cảnh số lượng bài báo và tin tức trực tuyến ngày càng gia tăng, nhu cầu tìm kiếm thông tin nhanh và chính xác trở nên vô cùng quan trọng. Tuy nhiên, các phương pháp tìm kiếm tuyến tính trên không gian vector có chi phí tính toán lớn khi dữ liệu tăng mạnh.

Dự án này xây dựng hệ thống tìm kiếm bài báo dựa trên HNSW (Hierarchical Navigable Small World) – một cấu trúc đồ thị hiệu quả cho Approximate Nearest Neighbor Search (ANNS). Hệ thống cho phép truy vấn các bài báo tương đồng ngữ nghĩa với tốc độ cao, độ chính xác tốt và khả năng mở rộng lớn.

2. Mục tiêu của dự án

Thu thập (crawl) dữ liệu bài báo từ các nguồn tin tức trực tuyến

Biểu diễn nội dung bài báo dưới dạng vector embedding

Xây dựng HNSW index cho tìm kiếm tương đồng ngữ nghĩa

Triển khai backend Python phục vụ truy vấn

Xây dựng giao diện web cho người dùng cuối

Đánh giá khả năng ứng dụng HNSW trong hệ thống tìm kiếm văn bản

3. Công nghệ sử dụng
3.1. Ngôn ngữ & Framework

Python 3

FastAPI – xây dựng RESTful API

Uvicorn – ASGI server

3.2. Thư viện chính

sentence-transformers – sinh embedding văn bản

hnswlib – xây dựng chỉ mục HNSW

feedparser, requests – crawl dữ liệu

numpy – xử lý vector

starlette, anyio – backend hỗ trợ

4. Kiến trúc hệ thống
Luồng xử lý tổng quát:

Crawl bài báo

Tiền xử lý & lưu metadata

Sinh embedding cho từng bài

Xây dựng chỉ mục HNSW

Nhận truy vấn người dùng

Sinh embedding truy vấn

Tìm kiếm ANN bằng HNSW

Trả về danh sách bài báo liên quan

5. Cấu trúc thư mục
project/
├── src/
│   ├── article_embedder.py
│   ├── article_search_system.py
│   ├── crawl_articles.py
│   ├── hnsw_manager.py
│   ├── graph.py
│   ├── merge_article_index.py
│   ├── update_summary_data.py
│   └── server.py
├── templates/
│   └── index.html
├── visualization.py
├── README.md
└── .gitignore

6. Mô tả các thành phần chính
🔹 crawl_articles.py

Thu thập bài báo từ các nguồn RSS / website và lưu nội dung vào bộ nhớ cục bộ.

🔹 article_embedder.py

Sử dụng mô hình Sentence Transformer để chuyển văn bản thành vector embedding.

🔹 hnsw_manager.py

Khởi tạo và xây dựng HNSW graph

Lưu / load chỉ mục từ ổ đĩa

Quản lý quá trình thêm vector

🔹 article_search_system.py

Thực hiện truy vấn tìm kiếm dựa trên embedding và HNSW index.

🔹 server.py

Backend FastAPI:

Nhận truy vấn từ frontend

Gọi hệ thống tìm kiếm

Trả kết quả về client

🔹 templates/index.html

Giao diện web cho người dùng tìm kiếm bài báo.

7. Hướng dẫn cài đặt
7.1. Cài đặt thư viện
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

8. Xây dựng dữ liệu & chỉ mục
Bước 1: Crawl bài báo
python crawl_articles.py


Kết quả:

article.data/ – nội dung bài báo

File metadata

Bước 2: Sinh embedding & build HNSW
python hnsw_manager.py


Kết quả:

article.index/ – HNSW index

embeddings.npy – vector embedding

9. Chạy hệ thống
Chạy backend
python server.py


Truy cập:

http://localhost:8000

10. Demo & triển khai
🔴 Live Demo (GitHub Pages)

👉 https://vp2802.github.io/hnsw-group7/

🟢 Google Colab (đã chạy sẵn)

👉 https://colab.research.google.com/drive/1iWQEyGi5aBXxDRD09-qgvT7lF-CNjnDB?usp=sharing

(Colab cho phép chạy thử toàn bộ pipeline mà không cần cài đặt môi trường cục bộ)

11. Đánh giá & nhận xét

HNSW cho tốc độ truy vấn rất nhanh so với tìm kiếm tuyến tính

Độ chính xác cao với dữ liệu văn bản lớn

Phù hợp cho các hệ thống tìm kiếm, recommendation, semantic search

Có thể mở rộng thêm:

Cập nhật index động

Đánh giá Recall / Latency

So sánh với FAISS, IVF, Flat index

12. Kết luận

Dự án đã triển khai thành công một hệ thống tìm kiếm bài báo dựa trên HNSW, kết hợp embedding ngữ nghĩa và đồ thị ANN. Kết quả cho thấy HNSW là giải pháp hiệu quả cho bài toán tìm kiếm tương đồng trên không gian vector lớn, có tiềm năng ứng dụng thực tế cao.
