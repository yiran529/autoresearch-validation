#!/usr/bin/env bash
set -euo pipefail
cd /home/wyr/CORAL/validation/greedylore-validation
export PYTHONDONTWRITEBYTECODE=1
exec python3 campaigns/powersgd-graph-timing-20261118/controller.py --execute >> campaigns/powersgd-graph-timing-20261118/controller.log 2>&1
