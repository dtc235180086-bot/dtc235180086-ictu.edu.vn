#!/bin/sh
# Linux/macOS: tạo môi trường, cài thư viện và chạy lại toàn bộ thực nghiệm
set -e
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { python3 -m venv .venv && .venv/bin/pip install -r requirements.txt; }
.venv/bin/python src/chapter1_pipeline.py data/insurance.csv
.venv/bin/python src/chapter2_models.py
.venv/bin/python src/make_html_report.py
