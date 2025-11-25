from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from article_search_system import ArticleSearchApp
import os
import numpy as np

app = FastAPI()

# Cho phép CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tạo thư mục templates nếu chưa tồn tại
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Templates + Static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

class SearchRequest(BaseModel):
    query: str
    topk: int = 10

# Load hệ thống
try:
    search_app = ArticleSearchApp()
    search_app.load_system()
    print("✅ Hệ thống tìm kiếm đã được load thành công!")
except Exception as e:
    print(f"❌ Lỗi khi load hệ thống: {e}")
    search_app = None

# Giao diện chính - trả về HTML tích hợp CSS
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    html_content = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Article Search Engine</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
        }

        body {
            background: linear-gradient(135deg, #1a73e8 0%, #4285f4 100%);
            color: #333;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
            color: white;
        }

        .logo {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            margin-bottom: 10px;
        }

        .logo-icon {
            font-size: 2.5rem;
            color: white;
        }

        .logo-text {
            font-size: 2.5rem;
            font-weight: 500;
            color: white;
        }

        .tagline {
            font-size: 1.2rem;
            opacity: 0.9;
            margin-bottom: 10px;
        }

        .search-container {
            background: white;
            border-radius: 24px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 5px;
            margin-bottom: 30px;
            display: flex;
            align-items: center;
            transition: box-shadow 0.3s;
        }

        .search-container:focus-within {
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
        }

        .search-icon {
            padding: 0 15px;
            color: #5f6368;
        }

        #query {
            flex: 1;
            border: none;
            outline: none;
            padding: 15px 0;
            font-size: 1.1rem;
            color: #333;
        }

        .search-btn {
            background: #1a73e8;
            color: white;
            border: none;
            border-radius: 20px;
            padding: 12px 24px;
            margin: 5px;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.3s;
        }

        .search-btn:hover {
            background: #0d62d9;
        }

        .results-container {
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 20px;
            min-height: 200px;
        }

        .results-header {
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e8eaed;
            color: #1a73e8;
            font-weight: 500;
        }

        .card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 4px solid #1a73e8;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        .card h2 {
            color: #1a73e8;
            margin-bottom: 10px;
            font-size: 1.3rem;
            line-height: 1.4;
        }

        .card p {
            margin-bottom: 8px;
            color: #555;
            line-height: 1.5;
        }

        .meta-info {
            display: flex;
            gap: 15px;
            margin-bottom: 10px;
            font-size: 0.9rem;
            color: #5f6368;
        }

        .meta-info span {
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .similarity {
            display: inline-block;
            background: #e8f0fe;
            color: #1a73e8;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 500;
            margin-top: 10px;
        }

        .footer {
            text-align: center;
            margin-top: 40px;
            color: white;
            font-size: 0.9rem;
            opacity: 0.8;
        }

        .empty-state {
            text-align: center;
            padding: 40px 20px;
            color: #5f6368;
        }

        .empty-state i {
            font-size: 3rem;
            margin-bottom: 15px;
            color: #dadce0;
        }

        .error-state {
            text-align: center;
            padding: 20px;
            background: #ffeaa7;
            border-radius: 8px;
            margin-bottom: 20px;
            color: #e17055;
        }

        @media (max-width: 600px) {
            .container {
                padding: 10px;
            }
            
            .logo-text {
                font-size: 2rem;
            }
            
            .search-container {
                flex-direction: column;
                border-radius: 12px;
                padding: 15px;
            }
            
            #query {
                width: 100%;
                margin-bottom: 10px;
            }
            
            .search-btn {
                width: 100%;
                border-radius: 8px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">
                <i class="fas fa-search logo-icon"></i>
                <h1 class="logo-text">Article Search</h1>
            </div>
            <p class="tagline">Tìm kiếm bài viết một cách thông minh</p>
        </div>

        <div class="search-container">
            <div class="search-icon">
                <i class="fas fa-search"></i>
            </div>
            <input id="query" type="text" placeholder="Nhập từ khoá tìm kiếm...">
            <button class="search-btn" onclick="doSearch()">Tìm kiếm</button>
        </div>

        <div class="results-container">
            <h3 class="results-header">Kết quả tìm kiếm</h3>
            <div id="results">
                <div class="empty-state">
                    <i class="fas fa-newspaper"></i>
                    <p>Nhập từ khoá và nhấn Tìm kiếm để xem kết quả</p>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>© 2025 Article Search Engine.</p>
        </div>
    </div>

    <script>
        async function doSearch() {
            const q = document.getElementById("query").value.trim();
            
            if (!q) {
                alert("Vui lòng nhập từ khoá tìm kiếm!");
                return;
            }
            
            // Hiển thị trạng thái đang tải
            document.getElementById("results").innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-spinner fa-spin"></i>
                    <p>Đang tìm kiếm...</p>
                </div>
            `;
            
            try {
                const response = await fetch("/search", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({ query: q, topk: 10 })
                });

                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`HTTP error! status: ${response.status}, details: ${errorText}`);
                }

                const data = await response.json();
                
                let html = "";
                
                if (data.error) {
                    html = `
                        <div class="error-state">
                            <i class="fas fa-exclamation-triangle"></i>
                            <p><strong>Lỗi:</strong> ${data.error}</p>
                            <p>${data.details || ''}</p>
                        </div>
                    `;
                } else if (data.results && data.results.length > 0) {
                    data.results.forEach(r => {
                        html += `
                            <div class="card">
                                <h2>${r.title}</h2>
                                <div class="meta-info">
                                    <span><i class="fas fa-newspaper"></i> ${r.source}</span>
                                    <span><i class="fas fa-tag"></i> ${r.category}</span>
                                </div>
                                <p>${r.summary}</p>
                                <span class="similarity">Độ tương đồng: ${r.similarity.toFixed(4)}</span>
                            </div>
                        `;
                    });
                } else {
                    html = `
                        <div class="empty-state">
                            <i class="fas fa-search"></i>
                            <p>Không tìm thấy kết quả nào cho "${q}"</p>
                            <p>Hãy thử với từ khoá khác hoặc kiểm tra chính tả</p>
                        </div>
                    `;
                }
                
                document.getElementById("results").innerHTML = html;
            } catch (error) {
                document.getElementById("results").innerHTML = `
                    <div class="error-state">
                        <i class="fas fa-exclamation-triangle"></i>
                        <p><strong>Lỗi kết nối:</strong> ${error.message}</p>
                        <p>Vui lòng thử lại sau hoặc kiểm tra kết nối mạng</p>
                    </div>
                `;
                console.error("Search error:", error);
            }
        }
        
        // Cho phép tìm kiếm bằng phím Enter
        document.getElementById("query").addEventListener("keypress", function(event) {
            if (event.key === "Enter") {
                doSearch();
            }
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

# Hàm chuyển đổi numpy types sang Python native types
def convert_numpy_types(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# API tìm kiếm - ĐÃ FIX LỖI SERIALIZATION
@app.post("/search")
async def search(req: SearchRequest):
    try:
        if search_app is None:
            return {"error": "Hệ thống tìm kiếm chưa được khởi tạo", "details": "Vui lòng kiểm tra lại"}
        
        # Debug: in thông tin query
        print(f"Search query: {req.query}, topk: {req.topk}")
        
        query_vector = search_app.hnsw_mgr.embedder.embed_query(req.query)
        print(f"Query vector shape: {len(query_vector)}")
        
        labels, distances = search_app.hnsw_mgr.index.knn_query(query_vector, k=req.topk)
        print(f"Found {len(labels[0])} results, distances: {distances[0][:5]}")  # In 5 distances đầu tiên
        
        # Ngưỡng tương đồng tối thiểu 30%
        MIN_SIMILARITY_THRESHOLD = 0.3
        
        results = []
        for i, (label, dist) in enumerate(zip(labels[0], distances[0])):
            print(f"Result {i}: label={label}, dist={dist}")
            
            # Tính similarity an toàn
            similarity = 1 / (1 + float(dist))  # Chuyển dist sang float trước
            
            print(f"  Similarity calculated: {similarity}")
            
            # Chỉ thêm kết quả nếu đạt ngưỡng tối thiểu 30%
            if similarity >= MIN_SIMILARITY_THRESHOLD:
                try:
                    article = search_app.hnsw_mgr.articles[int(label)]
                    
                    # Đảm bảo tất cả giá trị đều là Python native types
                    result_item = {
                        "title": str(article["title"]),
                        "source": str(article["source"]),
                        "category": str(article["category"]),
                        "summary": str(article.get("summary", "")[:200]),
                        "similarity": float(round(similarity, 4))  # Chuyển sang float
                    }
                    
                    results.append(result_item)
                    print(f"  Added article: {article['title'][:50]}...")
                except (IndexError, KeyError, ValueError) as e:
                    print(f"  Error getting article {label}: {e}")
                    continue
        
        print(f"Returning {len(results)} results")
        
        # Chuyển đổi tất cả numpy types trước khi trả về
        final_results = convert_numpy_types(results)
        return {"results": final_results}
        
    except Exception as e:
        print(f"Search error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "error": "Lỗi khi tìm kiếm",
            "details": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)