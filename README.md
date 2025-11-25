# hnsw-group7
requirements.txt: chứa các packages cần install
Các bước build web:
1.Tải các file article_embedder.py, artical_search_system.py, crawl_articles.py, hnsw_manager.py, script.js, server.py và bỏ vào 1 folder (Ví dụ: D:\hnsw-group7)
2. Tải file index.html về và bỏ vào trong folder templates (Ví dụ: D:\hnsw-group7\templates)
3. Mở Command Prompt của máy tính và chuyển sang folder chứa các file đã nói (Ví dụ câu lệnh: D: cd hnsw-group7)
4. Nếu chưa có folder article.data và article.index thì chạy câu lệnh python crawl_articles.py để build data
5. Tiếp tục chạy lệnh python hnsw_manager.py để build hnsw 
6. Gõ python server.py để tạo web
7. Lên GG Chrome và gõ http://localhost:8000/ để đến web
