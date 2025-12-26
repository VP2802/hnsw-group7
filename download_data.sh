%%bash
cd /content/hnsw-group7

cat > download_data.sh << 'EOF'
#!/usr/bin/env bash
set -e

# Link tải trực tiếp từ GitHub Release v2.0
URL="https://github.com/VP2802/hnsw-group7/releases/download/v2.0/article_index.zip"

ZIP="article_index.zip"
OUTDIR="article_index"

echo "==> [1/3] Downloading $ZIP ..."
rm -f "$ZIP"
wget -O "$ZIP" "$URL"

echo "==> [2/3] Unzipping to ./$OUTDIR ..."
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"
unzip -o "$ZIP" -d "$OUTDIR" >/dev/null

echo "==> [3/3] Done. Listing output:"
ls -la "$OUTDIR" | head
EOF

chmod +x download_data.sh
