from __future__ import annotations

import os
import re
import time
import html as html_lib
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from article_search_system import ArticleSearchApp

app = FastAPI()

# Cho phép CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tạo thư mục templates/static nếu chưa tồn tại
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Templates + Static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


class SearchRequest(BaseModel):
    query: str
    topk: int = Field(default=10, ge=1, le=50)
    mode: str = Field(default="hybrid", description="semantic|keyword|hybrid")
    sort: str = Field(default="relevance", description="relevance|newest")


# -----------------------
# Text/url sanitize (BACKEND)
# -----------------------
_TAG_RE = re.compile(r"<[^>]+>")


def strip_html_tags(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    s = html_lib.unescape(s)
    s = _TAG_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_text(text: Any, max_len: Optional[int] = None) -> str:
    s = strip_html_tags(text)
    if max_len is not None and len(s) > max_len:
        return s[:max_len]
    return s


def safe_url(u: Any) -> str:
    u = ("" if u is None else str(u)).strip()
    if u.lower().startswith(("http://", "https://")):
        return u
    return ""


# -----------------------
# Date parsing
# -----------------------
_DATE_KEYS = (
    "published",
    "pubDate",
    "date",
    "datetime",
    "time",
    "timestamp",
    "created_at",
    "updated_at",
    "createdAt",
    "updatedAt",
)


def _try_parse_datetime(val: str) -> Optional[datetime]:
    v = (val or "").strip()
    if not v:
        return None

    # numeric epoch seconds/ms
    try:
        if re.fullmatch(r"\d{10,13}", v):
            n = int(v)
            if len(v) == 13:
                n = n // 1000
            return datetime.fromtimestamp(n)
    except Exception:
        pass

    # common formats
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%a, %d %b %Y %H:%M:%S %z",  # RSS
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S.%f",
    ]
    for f in fmts:
        try:
            dt = datetime.strptime(v, f)
            # convert aware -> naive local
            if dt.tzinfo is not None:
                dt = dt.astimezone().replace(tzinfo=None)
            return dt
        except Exception:
            pass

    return None


def extract_article_datetime(article: Dict[str, Any]) -> Optional[datetime]:
    for k in _DATE_KEYS:
        if k in article and article[k]:
            dt = _try_parse_datetime(str(article[k]))
            if dt:
                return dt
    return None


def format_date_vi(dt: Optional[datetime]) -> str:
    if not dt:
        return ""
    return dt.strftime("%d/%m/%Y %H:%M")


# -----------------------
# Keyword index (BM25-lite)
# -----------------------
_TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)


def tokenize(text: str) -> List[str]:
    toks = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    return [t for t in toks if len(t) >= 2]


# Global keyword structures
KW_POSTINGS: Dict[str, List[Tuple[int, int]]] = {}
KW_DF: Dict[str, int] = {}
DOC_LEN: List[int] = []
AVG_DL: float = 0.0
N_DOCS: int = 0

# Date cache
DOC_DATE: List[Optional[datetime]] = []


def build_keyword_index(articles: List[Dict[str, Any]]) -> None:
    """Build a simple postings index: token -> list[(doc_id, tf)]."""
    global KW_POSTINGS, KW_DF, DOC_LEN, AVG_DL, N_DOCS, DOC_DATE

    postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    df: Dict[str, int] = defaultdict(int)
    doc_len: List[int] = []
    doc_date: List[Optional[datetime]] = []

    for i, a in enumerate(articles):
        title = safe_text(a.get("title", ""))
        summary = safe_text(a.get("summary", ""))
        toks = tokenize(f"{title} {summary}")
        doc_len.append(len(toks))

        doc_date.append(extract_article_datetime(a))

        if not toks:
            continue

        tf = Counter(toks)
        for tok, cnt in tf.items():
            postings[tok].append((i, cnt))
            df[tok] += 1

    KW_POSTINGS = dict(postings)
    KW_DF = dict(df)
    DOC_LEN = doc_len
    DOC_DATE = doc_date
    N_DOCS = len(articles)
    AVG_DL = (sum(doc_len) / max(1, len(doc_len))) if doc_len else 0.0


def bm25_lite_scores(
    query: str, *, k1: float = 1.2, b: float = 0.75, max_docs: int = 2000
) -> Dict[int, float]:
    """Compute BM25-ish scores for docs matching query tokens."""
    if N_DOCS == 0:
        return {}

    q_toks = tokenize(query)
    if not q_toks:
        return {}

    q_tf = Counter(q_toks)
    scores: Dict[int, float] = defaultdict(float)

    for tok, qcnt in q_tf.items():
        postings = KW_POSTINGS.get(tok)
        if not postings:
            continue

        df = KW_DF.get(tok, 0)
        idf = np.log((N_DOCS - df + 0.5) / (df + 0.5) + 1.0)

        for doc_id, tf in postings:
            dl = DOC_LEN[doc_id] if doc_id < len(DOC_LEN) else 0
            denom = tf + k1 * (1 - b + b * (dl / (AVG_DL + 1e-9)))
            score = float(idf) * (tf * (k1 + 1) / (denom + 1e-9))
            scores[doc_id] += score * (1.0 + 0.1 * (qcnt - 1))

    if len(scores) > max_docs:
        top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_docs]
        scores = defaultdict(float, top_items)

    return dict(scores)


def normalize_scores(d: Dict[int, float]) -> Dict[int, float]:
    if not d:
        return {}
    vals = list(d.values())
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-9:
        return {k: 0.0 for k in d}
    return {k: (v - mn) / (mx - mn) for k, v in d.items()}


# -----------------------
# Load hệ thống
# -----------------------
try:
    search_app = ArticleSearchApp()
    search_app.load_system()
    print("Hệ thống tìm kiếm đã được load thành công!")

    if (
        getattr(search_app, "hnsw_mgr", None) is not None
        and getattr(search_app.hnsw_mgr, "articles", None) is not None
    ):
        build_keyword_index(search_app.hnsw_mgr.articles)
        print(f"Keyword index built: {N_DOCS} docs, {len(KW_POSTINGS)} terms")
    else:
        print("Không build keyword index vì thiếu articles")
except Exception as e:
    print(f"Lỗi khi load hệ thống: {e}")
    search_app = None


# -----------------------
# UI
# -----------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    html_content = r"""
<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Article Search Engine</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    * { margin:0; padding:0; box-sizing:border-box; font-family: 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif; }
    body { background: linear-gradient(135deg, #1a73e8 0%, #4285f4 100%); color:#333; min-height:100vh; padding: 20px; }
    .container { max-width: 920px; margin: 0 auto; padding: 20px; }

    .header { text-align:center; margin-bottom: 26px; color:white; }
    .logo { display:flex; align-items:center; justify-content:center; gap: 14px; margin-bottom: 6px; }
    .logo-icon { font-size: 2.4rem; color:white; }
    .logo-text { font-size: 2.4rem; font-weight: 600; color:white; }
    .tagline { font-size: 1.08rem; opacity: 0.92; }

    .search-panel { background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.18); border-radius: 18px; padding: 14px; margin-bottom: 18px; backdrop-filter: blur(6px); }
    .search-row { display:flex; gap: 12px; align-items:center; }
    .search-box {
      flex: 1; background:white; border-radius: 999px; padding: 10px 14px; display:flex; align-items:center; gap: 10px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.10);
    }
    .search-icon { color:#5f6368; }
    #query { flex:1; border:none; outline:none; font-size: 1.08rem; padding: 6px 0; color:#333; }
    .search-btn {
      background:#1a73e8; color:white; border:none; border-radius: 999px;
      padding: 12px 20px; font-size: 1rem; font-weight: 700; cursor:pointer;
      box-shadow: 0 2px 10px rgba(0,0,0,0.10);
    }
    .search-btn:hover { background:#0d62d9; }

    .controls-row { display:flex; gap: 12px; margin-top: 12px; flex-wrap: wrap; }
    .control {
      background: rgba(255,255,255,0.92);
      border: 1px solid rgba(255,255,255,0.55);
      border-radius: 12px;
      padding: 10px 12px;
      display:flex; align-items:center; gap: 10px;
      box-shadow: 0 1px 6px rgba(0,0,0,0.08);
    }
    .control i { color:#1a73e8; }
    .control span { color:#3c4043; font-weight: 700; font-size: 0.92rem; }
    .control select {
      border: none; outline: none; background: transparent; font-weight: 700; color:#1a73e8;
      padding: 4px 6px; cursor:pointer;
    }

    .history {
      background: rgba(255,255,255,0.92);
      border: 1px solid rgba(255,255,255,0.55);
      border-radius: 12px;
      padding: 10px 12px;
      box-shadow: 0 1px 6px rgba(0,0,0,0.08);
      margin-top: 10px;
      display: none;
    }
    .history-header { display:flex; justify-content: space-between; align-items:center; margin: 8px 2px; color:#5f6368; font-size: 0.9rem; }
    .history-clear { cursor:pointer; color:#1a73e8; user-select:none; }
    .history-items { display:flex; flex-wrap: wrap; gap: 8px; }
    .chip {
      background:#e8f0fe; color:#1a73e8; border: 1px solid rgba(26,115,232,0.25);
      padding: 6px 10px; border-radius: 999px; font-size: 0.92rem;
      cursor:pointer;
    }
    .chip:hover { filter: brightness(0.98); }

    .results-container { background: white; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); padding: 20px; min-height: 220px; }
    .results-header { margin-bottom: 14px; padding-bottom: 10px; border-bottom: 1px solid #e8eaed; color:#1a73e8; font-weight: 700; display:flex; justify-content: space-between; align-items:center; }
    .results-sub { color:#5f6368; font-weight: 500; font-size: 0.92rem; }

    .card { background:#f8f9fa; border-radius: 10px; padding: 18px; margin-bottom: 16px; border-left: 5px solid #1a73e8; transition: transform 0.15s, box-shadow 0.15s; }
    .card:hover { transform: translateY(-1px); box-shadow: 0 4px 10px rgba(0,0,0,0.08); }
    .card h2 { color:#1a73e8; margin-bottom: 8px; font-size: 1.25rem; line-height: 1.35; }
    .card p { margin-top: 8px; margin-bottom: 0; color:#555; line-height: 1.6; }

    .meta-info { display:flex; flex-wrap: wrap; gap: 14px; margin-top: 4px; color:#5f6368; font-size: 0.92rem; }
    .meta-info span { display:flex; align-items:center; gap: 6px; }
    .meta-info a { color:#1a73e8; text-decoration:none; font-weight: 600; }
    .meta-info a:hover { text-decoration: underline; }

    .badge { display:inline-block; background:#e8f0fe; color:#1a73e8; padding: 5px 10px; border-radius: 999px; font-size: 0.85rem; font-weight: 700; margin-top: 12px; }

    .empty-state { text-align:center; padding: 38px 18px; color:#5f6368; }
    .empty-state i { font-size: 3rem; margin-bottom: 12px; color:#dadce0; }
    .error-state { text-align:center; padding: 18px; background:#ffeaa7; border-radius: 10px; margin-bottom: 16px; color:#e17055; }

    .footer { text-align:center; margin-top: 26px; color:white; font-size: 0.9rem; opacity: 0.85; }

    @media (max-width: 700px) {
      .container { padding: 10px; }
      .search-row { flex-direction: column; align-items: stretch; }
      .search-btn { width: 100%; border-radius: 14px; }
      .search-box { border-radius: 14px; }
      .controls-row { gap: 8px; }
      .control { width: 100%; justify-content: space-between; }
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
      <p class="tagline">Semantic • Keyword • Hybrid</p>
    </div>

    <div class="search-panel">
      <div class="search-row">
        <div class="search-box">
          <i class="fas fa-search search-icon"></i>
          <input id="query" type="text" placeholder="Nhập từ khoá tìm kiếm..." autocomplete="off">
        </div>
        <button class="search-btn" onclick="doSearch()">Tìm kiếm</button>
      </div>

      <div class="controls-row">
        <div class="control" title="Chế độ tìm kiếm">
          <i class="fas fa-sliders"></i>
          <span>Mode</span>
          <select id="mode">
            <option value="hybrid" selected>Hybrid</option>
            <option value="semantic">Semantic</option>
            <option value="keyword">Keyword</option>
          </select>
        </div>

        <div class="control" title="Sắp xếp kết quả">
          <i class="fas fa-sort"></i>
          <span>Sort</span>
          <select id="sort">
            <option value="relevance" selected>Liên quan nhất</option>
            <option value="newest">Mới nhất</option>
          </select>
        </div>

        <div class="control" title="Số kết quả">
          <i class="fas fa-list"></i>
          <span>TopK</span>
          <select id="topk">
            <option value="5">5</option>
            <option value="10" selected>10</option>
            <option value="20">20</option>
            <option value="50">50</option>
          </select>
        </div>
      </div>

      <div class="history" id="history">
        <div class="history-header">
          <span><i class="fas fa-clock"></i> Lịch sử tìm kiếm</span>
          <span class="history-clear" onclick="clearHistory()">Xoá</span>
        </div>
        <div class="history-items" id="historyItems"></div>
      </div>
    </div>

    <div class="results-container">
      <div class="results-header">
        <span>Kết quả tìm kiếm</span>
        <span class="results-sub" id="resultsSub"></span>
      </div>
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
    // ---------- util ----------
    function escapeHtml(s) {
      if (s === null || s === undefined) return "";
      return String(s)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function isHttpUrl(u) {
      return /^https?:\/\//i.test(String(u || ""));
    }

    // ---------- history ----------
    const HISTORY_KEY = "article_search_history_v1";

    function getHistory() {
      try {
        const raw = localStorage.getItem(HISTORY_KEY);
        const arr = raw ? JSON.parse(raw) : [];
        return Array.isArray(arr) ? arr : [];
      } catch {
        return [];
      }
    }

    function setHistory(arr) {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(arr.slice(0, 10)));
    }

    function addToHistory(q) {
      const query = String(q || "").trim();
      if (!query) return;
      let h = getHistory();
      h = h.filter(x => x !== query);
      h.unshift(query);
      setHistory(h);
      renderHistory();
    }

    function clearHistory() {
      localStorage.removeItem(HISTORY_KEY);
      renderHistory();
    }

    function renderHistory() {
      const h = getHistory();
      const box = document.getElementById("history");
      const items = document.getElementById("historyItems");
      if (!h.length) {
        box.style.display = "none";
        items.innerHTML = "";
        return;
      }
      box.style.display = "block";
      items.innerHTML = h.map(q => `<span class="chip" onclick="useHistory(${JSON.stringify(q)})">${escapeHtml(q)}</span>`).join("");
    }

    function useHistory(q) {
      document.getElementById("query").value = q;
      doSearch();
    }

    // ---------- search ----------
    let currentAbort = null;

    async function doSearch() {
      const q = document.getElementById("query").value.trim();
      const mode = document.getElementById("mode").value;
      const sort = document.getElementById("sort").value;
      const topk = Number(document.getElementById("topk").value || 10);

      if (!q) {
        alert("Vui lòng nhập từ khoá tìm kiếm!");
        return;
      }

      addToHistory(q);

      // Abort previous request
      if (currentAbort) currentAbort.abort();
      currentAbort = new AbortController();

      document.getElementById("resultsSub").textContent =
        `Mode: ${mode} • Sort: ${sort} • TopK: ${topk} • Thời gian: ...`;

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
          body: JSON.stringify({ query: q, topk: topk, mode: mode, sort: sort }),
          signal: currentAbort.signal
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`HTTP error! status: ${response.status}, details: ${errorText}`);
        }

        const data = await response.json();

        const count = (data.results && Array.isArray(data.results)) ? data.results.length : 0;
        const serverMs = (data.took_ms !== undefined && data.took_ms !== null) ? Number(data.took_ms) : null;

        // Hiển thị thời gian xử lý search ở backend
        document.getElementById("resultsSub").textContent =
          `Mode: ${mode} • Sort: ${sort} • TopK: ${topk} • ${count} kết quả` +
          (serverMs !== null ? ` • Thời gian: ${serverMs} ms` : "");

        let html = "";
        if (data.error) {
          html = `
            <div class="error-state">
              <i class="fas fa-exclamation-triangle"></i>
              <p><strong>Lỗi:</strong> ${escapeHtml(data.error)}</p>
              <p>${escapeHtml(data.details || "")}</p>
            </div>
          `;
        } else if (data.results && data.results.length > 0) {
          data.results.forEach(r => {
            const title = escapeHtml(r.title);
            const source = escapeHtml(r.source);
            const category = escapeHtml(r.category);
            const summary = escapeHtml(r.summary);
            const dateText = escapeHtml(r.published || "");
            const linkRaw = r.link || "";
            const link = isHttpUrl(linkRaw) ? linkRaw : "";
            const scoreLabel = (mode === "keyword") ? "Điểm" : "Độ tương đồng";

            html += `
              <div class="card">
                <h2>${title}</h2>
                <div class="meta-info">
                  <span><i class="fas fa-newspaper"></i> ${source || "(không rõ nguồn)"}</span>
                  <span><i class="fas fa-tag"></i> ${category || "(không rõ)"}</span>
                  ${dateText ? `<span><i class="fas fa-calendar"></i> ${dateText}</span>` : ""}
                  ${link ? `<span><i class="fas fa-link"></i> <a href="${escapeHtml(link)}" target="_blank" rel="noopener noreferrer">Mở bài gốc</a></span>` : ""}
                </div>
                <p>${summary}</p>
                <span class="badge">${scoreLabel}: ${Number(r.score).toFixed(4)}</span>
              </div>
            `;
          });
        } else {
          html = `
            <div class="empty-state">
              <i class="fas fa-search"></i>
              <p>Không tìm thấy kết quả nào cho "${escapeHtml(q)}"</p>
              <p>Hãy thử từ khoá khác hoặc đổi mode (Hybrid/Semantic/Keyword)</p>
            </div>
          `;
        }

        document.getElementById("results").innerHTML = html;

      } catch (error) {
        if (error.name === "AbortError") return;

        document.getElementById("resultsSub").textContent =
          `Mode: ${mode} • Sort: ${sort} • TopK: ${topk} • Thời gian: -`;

        document.getElementById("results").innerHTML = `
          <div class="error-state">
            <i class="fas fa-exclamation-triangle"></i>
            <p><strong>Lỗi kết nối:</strong> ${escapeHtml(error.message)}</p>
            <p>Vui lòng thử lại sau hoặc kiểm tra kết nối mạng</p>
          </div>
        `;
        console.error("Search error:", error);
      }
    }

    // Enter để search
    document.getElementById("query").addEventListener("keypress", function(event) {
      if (event.key === "Enter") doSearch();
    });

    // Render history khi load trang
    renderHistory();
  </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


# -----------------------
# Numpy conversion
# -----------------------
def convert_numpy_types(obj: Any) -> Any:
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(x) for x in obj]
    return obj


# -----------------------
# Search endpoint
# -----------------------
@app.post("/search")
async def search(req: SearchRequest):
    t0 = time.perf_counter()
    try:
        if search_app is None:
            took_ms = int((time.perf_counter() - t0) * 1000)
            return {
                "error": "Hệ thống tìm kiếm chưa được khởi tạo",
                "details": "Vui lòng kiểm tra lại",
                "took_ms": took_ms,
            }

        query = (req.query or "").strip()
        if not query:
            took_ms = int((time.perf_counter() - t0) * 1000)
            return {"results": [], "took_ms": took_ms}

        mode = (req.mode or "hybrid").lower()
        sort = (req.sort or "relevance").lower()
        topk = int(req.topk or 10)

        # Semantic candidates
        semantic_scores: Dict[int, float] = {}
        if mode in ("semantic", "hybrid"):
            k_sem = max(topk * 6, 60)
            query_vector = search_app.hnsw_mgr.embedder.embed_query(query)
            labels, distances = search_app.hnsw_mgr.index.knn_query(query_vector, k=k_sem)

            for label, dist in zip(labels[0], distances[0]):
                doc_id = int(label)
                sim = 1.0 / (1.0 + float(dist))
                semantic_scores[doc_id] = float(sim)

        # Keyword candidates
        keyword_scores: Dict[int, float] = {}
        if mode in ("keyword", "hybrid"):
            keyword_scores = bm25_lite_scores(query)

        # Combine
        combined: Dict[int, float] = {}
        if mode == "semantic":
            combined = semantic_scores
        elif mode == "keyword":
            combined = keyword_scores
        else:
            sem_n = normalize_scores(semantic_scores)
            kw_n = normalize_scores(keyword_scores)
            w_sem, w_kw = 0.55, 0.45
            all_ids = set(sem_n.keys()) | set(kw_n.keys())
            for doc_id in all_ids:
                combined[doc_id] = w_sem * sem_n.get(doc_id, 0.0) + w_kw * kw_n.get(doc_id, 0.0)

        if not combined:
            took_ms = int((time.perf_counter() - t0) * 1000)
            return {"results": [], "took_ms": took_ms}

        if mode in ("semantic", "hybrid"):
            MIN_SIM = 0.35
            combined = {k: v for k, v in combined.items() if v >= MIN_SIM}
            if not combined:
                took_ms = int((time.perf_counter() - t0) * 1000)
                return {"results": [], "took_ms": took_ms}

        articles = search_app.hnsw_mgr.articles

        def get_dt(doc_id: int) -> Optional[datetime]:
            if 0 <= doc_id < len(DOC_DATE):
                return DOC_DATE[doc_id]
            try:
                return extract_article_datetime(articles[doc_id])
            except Exception:
                return None

        items: List[Tuple[int, float, Optional[datetime]]] = []
        for doc_id, score in combined.items():
            if 0 <= doc_id < len(articles):
                items.append((doc_id, float(score), get_dt(doc_id)))

        if sort == "newest":
            items.sort(key=lambda x: (x[2] is not None, x[2] or datetime.min, x[1]), reverse=True)
        else:
            items.sort(key=lambda x: x[1], reverse=True)

        items = items[:topk]

        results = []
        for doc_id, score, dt in items:
            a = articles[doc_id]
            results.append(
                {
                    "title": safe_text(a.get("title", "")),
                    "source": safe_text(a.get("source", "")),
                    "category": safe_text(a.get("category", "")),
                    "summary": safe_text(a.get("summary", ""), max_len=240),
                    "link": safe_url(a.get("link", "")),
                    "published": format_date_vi(dt),
                    "score": float(round(score, 4)),
                }
            )

        took_ms = int((time.perf_counter() - t0) * 1000)
        return {"results": convert_numpy_types(results), "took_ms": took_ms}

    except Exception as e:
        import traceback
        traceback.print_exc()
        took_ms = int((time.perf_counter() - t0) * 1000)
        return {"error": "Lỗi khi tìm kiếm", "details": str(e), "took_ms": took_ms}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
