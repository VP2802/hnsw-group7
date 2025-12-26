#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_article_index.py

Gộp dữ liệu bài báo mới vào "article_index" hiện có.

Hỗ trợ 2 kiểu input JSON phổ biến:
1) File dạng LIST  -> chính là list[article] (vd: article_data/vn_articles.json do crawl_articles.py tạo)
2) File dạng DICT  -> có key "articles" (vd: article_index/metadata.json do hnsw_manager.py lưu)

Cơ chế gộp:
- Dedup ưu tiên theo "link" (case-sensitive). Nếu thiếu link thì fallback theo (title, source, published).
- Nếu trùng: giữ id cũ, và "làm giàu" record (fill các field đang thiếu) + cập nhật crawled_time nếu mới hơn.
- Nếu mới: gán id tăng dần (max_id + 1).

Cập nhật index:
- Mặc định chạy incremental: load index hiện tại, embed bài mới, add_items vào HNSW, vstack embeddings
- Nếu bạn muốn chính xác tuyệt đối (khi bạn update title/summary của bài cũ và muốn re-embed), dùng --rebuild để build lại toàn bộ.

Ví dụ:
  # 1) Add bài mới từ crawler output
  python merge_article_index.py --index-dir article_index --new-json article_data/vn_articles.json

  # 2) Gộp metadata từ một index khác (đã extract zip) vào index hiện tại
  python merge_article_index.py --index-dir article_index --new-json other_index/metadata.json --rebuild

  # 3) Nhiều file input
  python merge_article_index.py --index-dir article_index --new-json a.json b.json c.json

Yêu cầu:
- project có các file: hnsw_manager.py, article_embedder.py (và model/requirements tương ứng)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import project modules
from hnsw_manager import ArticleHNSWManager  # type: ignore


REQUIRED_KEYS = ["title", "link", "category", "language", "source"]


def _now_iso() -> str:
    return datetime.now().isoformat()


def _safe_get(d: Dict[str, Any], k: str, default=None):
    v = d.get(k, default)
    return default if v is None else v


def _normalize_article(a: Dict[str, Any]) -> Dict[str, Any]:
    """
    Đảm bảo article có đủ key cơ bản để pipeline chạy ổn.
    Không cố "decode HTML entities" vì bạn đang lưu raw RSS summary.
    """
    a = dict(a)

    # Key chuẩn
    a.setdefault("title", "")
    a.setdefault("summary", "")
    a.setdefault("link", "")
    a.setdefault("published", "")
    a.setdefault("category", "Unknown")
    a.setdefault("language", "Unknown")
    a.setdefault("source", "Unknown")
    a.setdefault("crawled_time", _now_iso())

    # id: có thể thiếu nếu là dữ liệu mới
    if "id" in a:
        # ép kiểu nếu có thể
        try:
            a["id"] = int(a["id"])
        except Exception:
            # giữ nguyên nếu không cast được
            pass

    return a


def load_articles_any_json(path: str) -> List[Dict[str, Any]]:
    """
    Load JSON theo 2 format:
    - list[article]
    - {"articles": list[article], ...}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        articles = data
    elif isinstance(data, dict) and "articles" in data and isinstance(data["articles"], list):
        articles = data["articles"]
    else:
        raise ValueError(
            f"Format JSON không hỗ trợ ở {path}. Cần là list[...] hoặc dict có key 'articles'."
        )

    normed = []
    for a in articles:
        if not isinstance(a, dict):
            continue
        normed.append(_normalize_article(a))

    return normed


def article_key(a: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """
    Key fallback khi link rỗng: (title, source, published, category)
    """
    return (
        _safe_get(a, "title", "").strip(),
        _safe_get(a, "source", "").strip(),
        _safe_get(a, "published", "").strip(),
        _safe_get(a, "category", "").strip(),
    )


def merge_two_articles(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge record trùng nhau:
    - ưu tiên giữ id cũ (nếu có)
    - nếu old thiếu field mà new có -> fill
    - crawled_time: lấy cái "mới hơn" theo ISO nếu parse được, còn không thì lấy new
    """
    merged = dict(old)

    # Giữ id cũ
    if "id" in old:
        merged["id"] = old["id"]
    elif "id" in new:
        merged["id"] = new["id"]

    # Fill các field quan trọng nếu đang trống
    for k in ["title", "summary", "link", "published", "category", "language", "source"]:
        ov = _safe_get(old, k, "")
        nv = _safe_get(new, k, "")
        if (ov is None) or (isinstance(ov, str) and ov.strip() == ""):
            if nv is not None and (not isinstance(nv, str) or nv.strip() != ""):
                merged[k] = nv

    # Nếu summary mới dài hơn nhiều, có thể ưu tiên new (giúp data "hoàn thiện")
    try:
        if isinstance(old.get("summary", ""), str) and isinstance(new.get("summary", ""), str):
            if len(new["summary"]) > len(old["summary"]) * 1.2:
                merged["summary"] = new["summary"]
    except Exception:
        pass

    # crawled_time: ưu tiên new nếu "mới hơn"
    old_ct = _safe_get(old, "crawled_time", "")
    new_ct = _safe_get(new, "crawled_time", "")
    merged["crawled_time"] = choose_newer_time(old_ct, new_ct)

    return merged


def choose_newer_time(old_ct: str, new_ct: str) -> str:
    def _parse_iso(s: str) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None

    od = _parse_iso(old_ct) if isinstance(old_ct, str) else None
    nd = _parse_iso(new_ct) if isinstance(new_ct, str) else None

    if od and nd:
        return new_ct if nd >= od else old_ct
    if nd:
        return new_ct
    if od:
        return old_ct
    return new_ct or old_ct or _now_iso()


@dataclass
class MergeResult:
    merged_articles: List[Dict[str, Any]]
    added_new: int
    merged_duplicates: int


def merge_articles(existing: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> MergeResult:
    """
    Merge theo link, fallback theo article_key.
    """
    # Index existing
    by_link: Dict[str, int] = {}
    by_fallback: Dict[Tuple[str, str, str, str], int] = {}

    merged = list(existing)

    for idx, a in enumerate(merged):
        link = _safe_get(a, "link", "")
        if link:
            by_link[link] = idx
        else:
            by_fallback[article_key(a)] = idx

    # max id
    max_id = -1
    for a in merged:
        if isinstance(a.get("id"), int):
            max_id = max(max_id, a["id"])

    added = 0
    dups = 0

    for a in incoming:
        a = _normalize_article(a)
        link = _safe_get(a, "link", "")

        hit_idx = None
        if link and link in by_link:
            hit_idx = by_link[link]
        elif not link:
            fk = article_key(a)
            if fk in by_fallback:
                hit_idx = by_fallback[fk]

        if hit_idx is not None:
            # merge duplicate
            merged[hit_idx] = merge_two_articles(merged[hit_idx], a)
            dups += 1
        else:
            # new
            max_id += 1
            if "id" not in a or not isinstance(a.get("id"), int):
                a["id"] = max_id
            merged.append(a)
            new_idx = len(merged) - 1
            if link:
                by_link[link] = new_idx
            else:
                by_fallback[article_key(a)] = new_idx
            added += 1

    return MergeResult(merged_articles=merged, added_new=added, merged_duplicates=dups)


def save_metadata(index_dir: str, dim: int, articles: List[Dict[str, Any]]) -> str:
    metadata = {
        "dim": dim,
        "total_articles": len(articles),
        "articles": articles,
        "build_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(index_dir, exist_ok=True)
    metadata_path = os.path.join(index_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata_path


def _try_resize(index, new_max: int) -> bool:
    """
    HNSWlib python binding thường có resize_index().
    Có bản có get_max_elements().
    """
    try:
        # if have get_max_elements
        current_max = None
        try:
            current_max = index.get_max_elements()
        except Exception:
            current_max = None

        if current_max is not None and new_max <= current_max:
            return True

        if hasattr(index, "resize_index"):
            index.resize_index(new_max)
            return True
    except Exception:
        return False
    return False


def incremental_update_index(
    index_dir: str,
    merged_articles: List[Dict[str, Any]],
    existing_count: int,
    new_articles: List[Dict[str, Any]],
    ef: int = 100,
) -> None:
    """
    Load index hiện có, embed new_articles, add_items, vstack embeddings, save.
    Lưu ý: chỉ incremental đối với bài "mới" (không re-embed bài cũ).
    """
    mgr = ArticleHNSWManager(index_dir=index_dir)
    mgr.load_index()  # loads mgr.articles + mgr.all_embeddings + mgr.index

    if mgr.index is None or mgr.all_embeddings is None:
        raise RuntimeError("Index/embeddings chưa được load đúng. Kiểm tra article_index/embeddings.npy và article_index.bin")

    if len(mgr.articles) != existing_count:
        # người dùng merge metadata có thể đổi thứ tự; vẫn cho chạy nhưng cảnh báo.
        print(
            f"[WARN] existing_count theo metadata ({existing_count}) khác count trong index ({len(mgr.articles)}). "
            f"Mình sẽ dùng count trong index để gán label."
        )
        existing_count = len(mgr.articles)

    # Dedup new_articles lại theo link so với mgr.articles
    existing_links = {a.get("link", "") for a in mgr.articles if a.get("link")}
    filtered_new = [a for a in new_articles if a.get("link", "") and a.get("link") not in existing_links]
    if not filtered_new:
        print("Không có bài mới để add vào index (toàn bộ bị trùng link). Chỉ cập nhật metadata.")
        mgr.articles = merged_articles
        save_metadata(index_dir, mgr.dim, mgr.articles)
        return

    valid_new, new_emb = mgr.embedder.embed_articles(filtered_new)
    if len(valid_new) == 0 or new_emb is None or len(new_emb) == 0:
        print("Không embed được bài mới. Chỉ cập nhật metadata.")
        mgr.articles = merged_articles
        save_metadata(index_dir, mgr.dim, mgr.articles)
        return

    # Capacity
    total_needed = existing_count + len(new_emb)
    _try_resize(mgr.index, total_needed + 256)  # buffer

    # Add to index
    new_labels = np.arange(existing_count, existing_count + len(new_emb))
    mgr.index.add_items(new_emb, new_labels)

    # Update in-memory
    mgr.all_embeddings = np.vstack([mgr.all_embeddings, new_emb])
    mgr.articles = merged_articles

    # Save artifacts
    save_metadata(index_dir, mgr.dim, mgr.articles)

    emb_path = os.path.join(index_dir, "embeddings.npy")
    np.save(emb_path, mgr.all_embeddings)

    idx_path = os.path.join(index_dir, "article_index.bin")
    mgr.index.save_index(idx_path)

    # set ef
    try:
        mgr.index.set_ef(ef)
    except Exception:
        pass

    print(f"✅ Incremental update OK: +{len(new_emb)} vectors. Total vectors: {mgr.index.get_current_count()}")


def rebuild_index(index_dir: str, articles: List[Dict[str, Any]], max_elements: Optional[int] = None) -> None:
    mgr = ArticleHNSWManager(index_dir=index_dir)
    if max_elements is None:
        # max_elements ít nhất bằng số bài hiện có, cộng buffer
        max_elements = max(len(articles) + 256, 1024)
    ok = mgr.build_index(articles, max_elements=max_elements)
    if not ok:
        raise RuntimeError("Rebuild index thất bại. Xem log ở build_index().")
    print(f"✅ Rebuild OK. Total articles: {len(mgr.articles)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default="article_index", help="Thư mục article_index hiện có (có metadata.json, embeddings.npy, article_index.bin)")
    ap.add_argument("--new-json", nargs="+", required=True, help="1 hoặc nhiều file JSON bài báo mới (vn_articles.json hoặc metadata.json của index khác)")
    ap.add_argument("--out-dir", default=None, help="Nếu muốn output sang thư mục khác (copy index mới). Mặc định ghi đè vào --index-dir")
    ap.add_argument("--rebuild", action="store_true", help="Build lại toàn bộ index (chậm hơn nhưng chính xác nhất)")
    ap.add_argument("--max-elements", type=int, default=None, help="Chỉ dùng khi --rebuild. max_elements cho HNSW")
    args = ap.parse_args()

    index_dir = args.index_dir
    out_dir = args.out_dir or index_dir

    # Load existing metadata (nếu có)
    existing_metadata_path = os.path.join(index_dir, "metadata.json")
    existing_articles: List[Dict[str, Any]] = []
    if os.path.exists(existing_metadata_path):
        existing_articles = load_articles_any_json(existing_metadata_path)
        print(f"Đã load existing metadata: {existing_metadata_path} -> {len(existing_articles)} bài")
    else:
        print(f"[WARN] Không thấy {existing_metadata_path}. Sẽ coi như index rỗng (chỉ phù hợp khi --rebuild).")

    # Load incoming
    incoming_all: List[Dict[str, Any]] = []
    for p in args.new_json:
        inc = load_articles_any_json(p)
        incoming_all.extend(inc)
        print(f"Đã load incoming: {p} -> {len(inc)} bài")

    # Merge
    mr = merge_articles(existing_articles, incoming_all)
    merged_articles = mr.merged_articles
    print(f"=== MERGE RESULT ===")
    print(f"Existing: {len(existing_articles)} | Incoming: {len(incoming_all)}")
    print(f"Added new: {mr.added_new} | Merged duplicates: {mr.merged_duplicates}")
    print(f"Total after merge: {len(merged_articles)}")

    # Nếu out_dir khác index_dir: copy artifacts? Ở đây simple: rebuild hoặc save metadata + rebuild
    if out_dir != index_dir:
        os.makedirs(out_dir, exist_ok=True)
        print(f"[INFO] out-dir khác index-dir: {out_dir}. Khuyến nghị dùng --rebuild để tạo index đồng bộ trong out-dir.")

    # Update index
    if args.rebuild:
        rebuild_index(out_dir, merged_articles, max_elements=args.max_elements)
    else:
        # incremental: cần index artifacts tồn tại
        idx_path = os.path.join(index_dir, "article_index.bin")
        emb_path = os.path.join(index_dir, "embeddings.npy")
        if not (os.path.exists(idx_path) and os.path.exists(emb_path) and os.path.exists(existing_metadata_path)):
            print("[ERROR] Thiếu file index để incremental update. Bạn cần --rebuild (hoặc đảm bảo article_index đủ 3 file: metadata.json, embeddings.npy, article_index.bin).")
            sys.exit(2)

        # new articles chính là phần "added" (không trùng). Để lấy chính xác, lọc theo link so với existing_articles.
        existing_links = {a.get("link", "") for a in existing_articles if a.get("link")}
        new_only = [a for a in merged_articles if a.get("link", "") and a.get("link") not in existing_links]

        incremental_update_index(
            index_dir=index_dir,
            merged_articles=merged_articles,
            existing_count=len(existing_articles),
            new_articles=new_only,
        )

    # Always save metadata to out_dir (if rebuild already did it, it's fine)
    # In incremental, we saved already inside incremental_update_index.
    if args.rebuild:
        # build_index already saved metadata; but to be safe keep consistent with merged_articles ordering
        # (build_index filters duplicates by link again inside ArticleHNSWManager.build_index)
        pass

    print("DONE.")


if __name__ == "__main__":
    main()
