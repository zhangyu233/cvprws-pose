#!/usr/bin/env bash
set -euo pipefail

# Offline cached trust-head training + W&B.
# Usage:
#   bash projects/self_verify_pose/run_offline_cached_trusthead_wandb.sh /path/to/pose_ckpt.pth
# Optional env vars:
#   WANDB_PROJECT=self_verify_pose
#   WANDB_NAME=offline_cached_trusthead
#   WANDB_MODE=online|offline|disabled
#   (plus all env vars supported by run_offline_cached_trusthead.sh)

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

: "${WANDB_PROJECT:=self_verify_pose}"
: "${WANDB_NAME:=offline_cached_trusthead_rtmpose_s}"
: "${WANDB_MODE:=online}"

cd "$REPO_ROOT"

python - <<'PY'
try:
    import wandb  # noqa: F401
except Exception as e:
    raise SystemExit(
        'wandb is not available. Please run: pip install -U wandb\n'
        'Then run: wandb login\n\n'
        f'Original import error: {e}'
    )
print('wandb import ok')
PY

export USE_WANDB=1
export WANDB_PROJECT
export WANDB_NAME
export WANDB_MODE

bash projects/self_verify_pose/run_offline_cached_trusthead.sh "$@"
