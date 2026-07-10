#!/usr/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Usage: build.sh [OPTIONS]

Build TileLang from source.

Options:
  --wheel, -w          Build wheel package (Release, strip enabled)
  --no-deps, -n        Skip installing requirements; only pip install -e .
  -h, --help           Show this help

Default: install requirements-dev.txt + requirements.txt, then editable install.
EOF
}

install_deps() {
  pip install -r requirements-dev.txt
  pip install -r requirements.txt
}

build_editable() {
  pip install -e . -v --no-build-isolation \
    --config-settings=install.strip=false \
    --config-settings=cmake.build-type=Debug
}

build_wheel() {
  python -m build --wheel --no-isolation \
    --config-setting=cmake.build-type=Release \
    --config-setting=install.strip=true
}

WHEEL=0
NO_DEPS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wheel|-w)
      WHEEL=1
      shift
      ;;
    --no-deps|-n)
      NO_DEPS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$NO_DEPS" -eq 0 ]]; then
  install_deps
fi

if [[ "$WHEEL" -eq 1 ]]; then
  build_wheel
else
  build_editable
fi
