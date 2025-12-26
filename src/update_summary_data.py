#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_summary_data.py

Tạo/cập nhật file thống kê bài báo (.txt) theo format giống mẫu "thong_ke_bai_bao.txt",
dựa trên dữ liệu JSON sau khi bạn đã gộp.

Hỗ trợ 2 kiểu input:
1) LIST  -> list[article] (vd: article_data/vn_articles.json)
2) DICT  -> {"articles": list[article], ...} (vd: article_index/metadata.json)

Mỗi article dự kiến có các field (thiếu thì sẽ fallback):
- category
- language
- source

Cách chạy:
  python update_summary_data.py --in article_index/metadata.json --out thong_ke_bai_bao.txt
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Tuple


def load_articles_any_json(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        articles = data
    elif isinstance(data, dict) and isinstance(data.get("articles"), list):
        articles = data["articles"]
    else:
        raise ValueError("Input JSON phải là list[...] hoặc dict có key 'articles'.")

    out: List[Dict[str, Any]] = []
    for a in articles:
        if isinstance(a, dict):
            out.append(a)
    return out


def _safe_str(v: Any, default: str = "Unknown") -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _fmt_line(name: str, count: int, total: int, name_width: int = 25) -> str:
    pct = (count / total * 100.0) if total else 0.0
    return f"{name:<{name_width}}{count:>5} bài ({pct:5.1f}%)"


def _sorted(counter: Counter) -> List[Tuple[str, int]]:
    return sorted(counter.items(), key=lambda x: (-x[1], str(x[0])))


def build_report_text(articles: List[Dict[str, Any]]) -> str:
    total = len(articles)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cat = Counter()
    lang = Counter()
    src = Counter()

    for a in articles:
        cat[_safe_str(a.get("category"))] += 1
        lang[_safe_str(a.get("language"))] += 1
        src[_safe_str(a.get("source"))] += 1

    lines: List[str] = []
    lines.append("THỐNG KÊ BÀI BÁO - PHÂN LOẠI THEO CHỦ ĐỀ VÀ NGÔN NGỮ")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Tổng số bài báo: {total}")
    lines.append(f"Thời gian thống kê: {ts}")
    lines.append("")
    lines.append("PHÂN BỐ THEO CHỦ ĐỀ:")
    lines.append("-" * 40)
    for name, cnt in _sorted(cat):
        lines.append(_fmt_line(str(name), int(cnt), total))
    lines.append("")
    lines.append("PHÂN BỐ THEO NGÔN NGỮ:")
    lines.append("-" * 40)
    for name, cnt in _sorted(lang):
        lines.append(_fmt_line(str(name), int(cnt), total))
    lines.append("")
    lines.append("PHÂN BỐ THEO NGUỒN BÁO:")
    lines.append("-" * 40)
    for name, cnt in _sorted(src):
        lines.append(_fmt_line(str(name), int(cnt), total))
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Đường dẫn JSON input (metadata.json hoặc list json)")
    ap.add_argument("--out", dest="out", default="thong_ke_bai_bao.txt", help="Đường dẫn file txt output")
    args = ap.parse_args()

    articles = load_articles_any_json(args.inp)
    txt = build_report_text(articles)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(txt)

    print(f"✅ Wrote report: {args.out} (total={len(articles)})")


if __name__ == "__main__":
    main()
