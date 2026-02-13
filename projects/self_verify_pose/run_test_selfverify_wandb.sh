#!/usr/bin/env bash
set -euo pipefail

# Self-Verify + W&B testing launcher
# Usage:
#   bash projects/self_verify_pose/run_test_selfverify_wandb.sh /path/to/ckpt.pth
# Defaults to: work_dirs/self_verify_pose/selfverify_cycle/latest.pth
# Optional env vars:
#   WANDB_PROJECT=self_verify_pose
#   WANDB_NAME=selfverify_cycle_rtmpose_s_test
#   WANDB_MODE=online|offline|disabled

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CONFIG="$REPO_ROOT/projects/self_verify_pose/configs/selfverify_cycle_rtmpose_s_1x32g_coco-256x192.py"
WORK_DIR_ROOT_DEFAULT="$REPO_ROOT/work_dirs/self_verify_pose/selfverify_cycle_test"

CKPT=${1:-"$REPO_ROOT/work_dirs/self_verify_pose/selfverify_cycle/latest.pth"}

: "${WANDB_PROJECT:=self_verify_pose}"
: "${WANDB_NAME:=selfverify_cycle_rtmpose_s_test}"
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

WORK_DIR_ROOT="${WORK_DIR:-$WORK_DIR_ROOT_DEFAULT}"

mim test mmpose "$CONFIG" --checkpoint "$CKPT" --work-dir "$WORK_DIR_ROOT"
