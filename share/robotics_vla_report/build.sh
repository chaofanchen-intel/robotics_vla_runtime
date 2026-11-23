#!/usr/bin/env bash
# 编译两份 PDF：main.pdf（结果汇报稿，16:9）与 main_framework.pdf（三框架机制与对比，同为 16:9）。
# 优先用 Tectonic（可通过 TECTONIC 环境变量指定路径），否则用 latexmk + XeLaTeX。
# 用法：bash build.sh          编译两份
#       bash build.sh main     只编译结果稿
#       bash build.sh framework 只编译框架文档
set -euo pipefail
cd "$(dirname "$0")"

TECTONIC_BIN="${TECTONIC:-$(command -v tectonic || true)}"
if [[ -z "$TECTONIC_BIN" && -x "$HOME/.local/bin/tectonic" ]]; then
  TECTONIC_BIN="$HOME/.local/bin/tectonic"
fi

build() {
  local src="$1"
  if [[ -n "$TECTONIC_BIN" ]]; then
    "$TECTONIC_BIN" "$src"
  elif command -v latexmk >/dev/null 2>&1; then
    latexmk -xelatex -interaction=nonstopmode -halt-on-error "$src"
  else
    printf '需要 Tectonic 或 XeLaTeX + latexmk；见 README.md。\n' >&2
    exit 127
  fi
}

case "${1:-all}" in
  main)      build main.tex ;;
  framework) build main_framework.tex ;;
  all)       build main.tex; build main_framework.tex ;;
  *)         printf '未知参数：%s（可用 main / framework / 不带参数）\n' "$1" >&2; exit 2 ;;
esac
