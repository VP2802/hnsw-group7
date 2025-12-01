# Hệ thống Tìm kiếm Bài báo bằng HNSW

Dự án này triển khai một công cụ tìm kiếm bài báo sử dụng HNSW (Hierarchical Navigable Small World) để thực hiện truy vấn tương đồng nhanh. Hệ thống bao gồm các thành phần để crawl (thu thập) bài báo, sinh embedding văn bản, xây dựng chỉ mục HNSW và cung cấp giao diện web để tìm kiếm.

## Cấu trúc thư mục

```
project/
├── templates/
│   └── index.html
|   └── sript.js
├── article_embedder.py
├── article_search_system.py
├── crawl_articles.py
├── hnsw_manager.py
├── graph.py
├── server.py
├── requirements.txt
└── README.md
```

## Chức năng chính

* Crawl tin tức từ các nguồn báo online, lưu vào thư mục dữ liệu.
* Sinh embedding cho từng bài báo bằng mô hình câu.
* Xây dựng chỉ mục HNSW cho tìm kiếm xấp xỉ theo độ tương đồng.
* Cung cấp giao diện web để người dùng nhập truy vấn tìm bài báo.
* Backend viết bằng Python.

## Cài đặt

1. Cài các thư viện cần thiết:

```
pip install -r requirements.txt
```

2. Đảm bảo cấu trúc thư mục giống như ở trên.

## Xây dựng dữ liệu

Nếu chưa có thư mục `article.data` và `article.index`, bạn cần chạy quá trình crawl và build chỉ mục.

### Bước 1: Crawl bài báo

Crawl các bài báo từ nguồn đã định sẵn và lưu dữ liệu vào ổ đĩa.

```
python crawl_articles.py
```

Kết quả:

* Thư mục `article.data/` chứa nội dung bài báo.
* File metadata phục vụ cho bước embed.

### Bước 2: Sinh embedding và build HNSW

Chạy script build chỉ mục HNSW:

```
python hnsw_manager.py
```

Kết quả:

* Thư mục `article.index/` chứa chỉ mục HNSW.
* File `embeddings.npy` chứa vector embedding của các bài báo.

## Chạy web server

Chạy server:

```
python server.py
```

Sau đó mở trình duyệt và truy cập:

```
http://localhost:8000/
```

## Cách sử dụng

1. Nhập truy vấn tìm kiếm vào ô tìm kiếm.
2. Hệ thống sẽ sinh embedding cho truy vấn.
3. Chỉ mục HNSW được dùng để tìm các bài tương tự nhất.
4. Trả về danh sách bài báo liên quan.

## Mô tả file

* **crawl_articles.py**
  Crawl bài báo và lưu dữ liệu.

* **article_embedder.py**
  Sinh embedding từ văn bản.

* **article_search_system.py**
  Xử lý tìm kiếm bài báo bằng HNSW.

* **hnsw_manager.py**
  Xây dựng và quản lý HNSW index.

* **server.py**
  Chạy web server backend.

* **script.js**
  Gửi truy vấn từ frontend và nhận kết quả.

* **templates/index.html**
  Giao diện web phía người dùng.

## Ghi chú

* Nếu cập nhật danh sách bài báo, cần chạy lại bước embed và build chỉ mục.
* Cần dùng phiên bản Python tương thích với thư viện trong `requirements.txt`.
