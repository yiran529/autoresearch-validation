#!/usr/bin/env bash
set -euo pipefail
printf "%s\n" "Historical campaign launcher archived. Use python3 -m glval plan EXPERIMENT.json, then glval run RUN --gpus GPUS --execute after authorization." >&2
exit 2
