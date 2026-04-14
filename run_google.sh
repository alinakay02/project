#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source venv/Scripts/activate
python -u scripts/run_all_tests.py --dataset google "$@"
