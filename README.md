HNSW-GROUP7/
├── 📁 src/                    # Source code chính
│   ├── crawl_articles.py          # Thu thập bài báo
│   ├── article_embedder.py        # Tạo embedding
│   ├── hnsw_manager.py            # Quản lý HNSW index
│   ├── article_search_system.py   # Hệ thống tìm kiếm
│   ├── server.py                  # FastAPI backend
│   ├── merge_article_index.py     # Gộp chỉ mục
│   ├── update_summary_data.py     # Cập nhật thống kê
│   └── graph.py                   # Visualize đồ thị của data
│
├── 📁 templates/                 # Frontend HTML
│   └── index.html                # Giao diện web
│
├── 📄 index.html                 # Trang chủ chính (redirect)
├── 📄 visualization.py           # Visualize HNSW graph
├── 📄 requirements.txt           # Dependencies
├── 📄 .gitignore
└── 📄 README.md

Mô tả chi tiết:
📁 src/ - Core source code
crawl_articles.py: Crawl RSS feeds (30+ nguồn, 8,661 bài báo)

article_embedder.py: Tạo embedding bằng Vietnamese-SBERT (768D)

hnsw_manager.py: Xây dựng và query HNSW index

article_search_system.py: Search engine với 3 chế độ (semantic/keyword/hybrid)

server.py: FastAPI backend (port 8000)

merge_article_index.py: Merge multiple indices

update_summary_data.py: Update metadata và thống kê

graph.py: Visualize Đồ thị biểu diễn data đã crawl

📁 templates/ - Frontend
index.html: Single-page web app với tìm kiếm

📁 article_index/ - Index files
article_index.bin: Binary HNSW index

embeddings.npy: Vector embeddings

metadata.json: Article metadata

benchmark_results.json: Performance data

📁 article_data/ - Raw data
articles.json: All crawled articles

summary.txt: Dataset statistics

📄 Root files
index.html: Main landing page

visualization.py: Visualize HNSW structure

requirements.txt: Python dependencies

README.md: Project documentation

Quy trình chạy:
bash
# 1. Cài dependencies
pip install -r requirements.txt

# 2. Crawl dữ liệu
python src/crawl_articles.py

# 3. Build HNSW index
python src/hnsw_manager.py

# 4. Chạy server
python src/server.py

# 5. Truy cập web
# http://localhost:8000
Demo:
Live Demo: https://vp2802.github.io/hnsw-group7/

Colab: [https://colab.research.google.com/drive/1iWQEyGi5aBXxDRD09-qgvT7lF-CNjnDB](https://l.facebook.com/l.php?u=https%3A%2F%2Fcolab.research.google.com%2Fdrive%2F1iWQEyGi5aBXxDRD09-qgvT7lF-CNjnDB%3Fusp%3Dsharing%26fbclid%3DIwZXh0bgNhZW0CMTAAYnJpZBExb2d4NlprVGY0bFlXM1pIanNydGMGYXBwX2lkEDIyMjAzOTE3ODgyMDA4OTIAAR5hCqw5G8QsAbLqop9shEsidhxlttNVSxy5WAlGG91isPYU5_rCoyPw7LRfXg_aem_6kbgyS0K9tb_aujNYIvBRQ&h=AT2l7o3dErmF_vDnALnVQ4JcWzVvYseKj07JoUDR4jZpuBHHq9P2gt7FIDIPDdoB1mINVb00IH3oBIUSXLFwWqCeaUTubxyfLkvwgyDoai_LkI_uM18QArTd9eBksZXsRHPW3RH8bzhIYL52Ax28jQ)
