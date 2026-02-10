#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CONFIG="$REPO_ROOT/projects/self_verify_pose/configs/selfverify_cycle_rtmpose_s_1x32g_coco-256x192.py"
WORK_DIR="$REPO_ROOT/work_dirs/self_verify_pose/selfverify_cycle"

cd "$REPO_ROOT"

# You can pass extra mmengine args after this script, e.g.:
#   bash projects/self_verify_pose/run_train_selfverify.sh --cfg-options train_dataloader.batch_size=8
mim train mmpose "$CONFIG" --work-dir "$WORK_DIR" "$@"
