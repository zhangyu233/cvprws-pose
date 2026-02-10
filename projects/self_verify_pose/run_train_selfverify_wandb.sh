#!/usr/bin/env bash
set -euo pipefail

# Self-Verify + W&B training launcher
# Usage (from repo root):
#   bash projects/self_verify_pose/run_train_selfverify_wandb.sh
# Optional env vars:
#   WANDB_PROJECT=self_verify_pose
#   WANDB_NAME=selfverify_cycle_rtmpose_s
#   WANDB_MODE=online|offline|disabled
# Pass-through args example:
#   bash projects/self_verify_pose/run_train_selfverify_wandb.sh --cfg-options train_dataloader.batch_size=8

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CONFIG="$REPO_ROOT/projects/self_verify_pose/configs/selfverify_cycle_rtmpose_s_1x32g_coco-256x192.py"
WORK_DIR_DEFAULT="$REPO_ROOT/work_dirs/self_verify_pose/selfverify_cycle"

# Defaults (can be overridden by env)
: "${WANDB_PROJECT:=self_verify_pose}"
: "${WANDB_NAME:=selfverify_cycle_rtmpose_s}"
: "${WANDB_MODE:=online}"

cd "$REPO_ROOT"

# Basic dependency check
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

# Allow overriding work dir via env WORK_DIR
WORK_DIR="${WORK_DIR:-$WORK_DIR_DEFAULT}"

mim train mmpose "$CONFIG" --work-dir "$WORK_DIR" "$@"
