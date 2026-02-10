#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CONFIG="$REPO_ROOT/projects/self_verify_pose/configs/selfverify_cycle_rtmpose_s_1x32g_coco-256x192.py"
WORK_DIR="$REPO_ROOT/work_dirs/self_verify_pose/selfverify_cycle"

CKPT=${1:-"$WORK_DIR/latest.pth"}

cd "$REPO_ROOT"

# Usage:
#   bash projects/self_verify_pose/run_test_selfverify.sh /path/to/ckpt.pth
mim test mmpose "$CONFIG" --checkpoint "$CKPT" --work-dir "$WORK_DIR/test"
