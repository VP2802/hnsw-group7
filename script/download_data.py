#!/usr/bin/env python3
"""
Download pre-built data từ GitHub
"""

import os
import requests
import zipfile

def main():
    print("TẢI PRE-BUILT DATA")
    print("=" * 30)
    
    DOWNLOAD_URL = "https://github.com/VP2802/hnsw-group7/releases/download/v1.0.1/article_index.zip"
    
    print("Đang tải data từ GitHub...")
    
    try:
        response = requests.get(DOWNLOAD_URL)
        response.raise_for_status()
        
        with open("article_index.zip", "wb") as f:
            f.write(response.content)
        
        print("Download xong!")
        
        print("Đang giải nén...")
        with zipfile.ZipFile("article_index.zip", "r") as zip_ref:
            zip_ref.extractall("article_index")
        
        os.remove("prebuilt_data.zip")
        
        print("HOÀN TẤT! Data đã sẵn sàng trong folder article_index/")
        print("Chạy: python src/main_app.py")
        
    except Exception as e:
        print(f"Lỗi: {e}")
        print("Kiểm tra internet và thử lại")

if __name__ == "__main__":
    main()