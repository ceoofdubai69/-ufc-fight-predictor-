#!/bin/bash
PYTHONPATH=/Users/ceoofmacs/Library/Python/3.9/lib/python/site-packages \
  /Users/ceoofmacs/Library/Python/3.9/bin/streamlit run \
  "$(dirname "$0")/app.py" --server.port 8501
