#!/usr/bin/env bash
# 把一张 1024x1024 的 PNG 转成 macOS 图标 build/icon.icns。
# 用法：build/make-icns.sh path/to/icon-1024.png
set -euo pipefail
SRC="${1:?用法: make-icns.sh icon-1024.png}"
OUT="$(cd "$(dirname "$0")" && pwd)/icon.icns"
TMP="$(mktemp -d)/icon.iconset"; mkdir -p "$TMP"

for s in 16 32 64 128 256 512; do
  sips -z $s $s     "$SRC" --out "$TMP/icon_${s}x${s}.png"        >/dev/null
  sips -z $((s*2)) $((s*2)) "$SRC" --out "$TMP/icon_${s}x${s}@2x.png" >/dev/null
done
cp "$SRC" "$TMP/icon_512x512@2x.png"
iconutil -c icns "$TMP" -o "$OUT"
echo "wrote $OUT"
