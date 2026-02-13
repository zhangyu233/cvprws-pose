#!/usr/bin/env bash
set -euo pipefail

# FK-based offline cache + trust head training.
# This uses pytorch_kinematics to define an explicit kinematic chain and
# runs IK over joint angles to compute the solvability pseudo-trust.
#
# Usage:
#   bash projects/self_verify_pose/run_offline_cached_trusthead_fk.sh /path/to/pose_ckpt.pth
# Optional env vars:
#   CACHE_DIR=work_dirs/self_verify_pose/cache
#   CACHE_TRAIN=work_dirs/self_verify_pose/cache/train_sol_fk.npz
#   FORCE_CACHE=1
#   MAX_SAMPLES=1000
#   WORK_DIR=...

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
POSE_CKPT=${POSE_CKPT:-}

# If the first argument is not an option, treat it as checkpoint path.
if [[ ${#} -ge 1 && "${1}" != -* ]]; then
  POSE_CKPT="$1"
  shift || true
fi

if [[ -z "${POSE_CKPT}" ]]; then
  if [[ -f "$REPO_ROOT/checkpoints/rtmpose-s_simcc-coco_256x192.pth" ]]; then
    POSE_CKPT="$REPO_ROOT/checkpoints/rtmpose-s_simcc-coco_256x192.pth"
    echo "[INFO] Using default pose checkpoint: $POSE_CKPT" >&2
  fi
fi

if [[ -z "${POSE_CKPT}" ]]; then
  echo "ERROR: pose checkpoint path is required (no default found)." >&2
  echo "Usage: bash projects/self_verify_pose/run_offline_cached_trusthead_fk.sh /path/to/pose_ckpt.pth" >&2
  exit 1
fi

cd "$REPO_ROOT"

# Ensure pytorch_kinematics is available.
if ! python - <<'PY'
import pytorch_kinematics  # noqa: F401
print('pytorch_kinematics import ok')
PY
then
  echo "Installing pytorch-kinematics ..." >&2
  python -m pip install -U pytorch-kinematics
fi

CACHEGEN_CONFIG="$REPO_ROOT/projects/self_verify_pose/configs/cachegen_rtmpose_s_1x32g_coco-256x192.py"
TRAIN_CONFIG="$REPO_ROOT/projects/self_verify_pose/configs/solvability_cached_trusthead_rtmpose_s_1x32g_coco-256x192.py"

: "${CACHE_DIR:=$REPO_ROOT/work_dirs/self_verify_pose/cache}"
: "${FORCE_CACHE:=0}"
: "${MAX_SAMPLES:=-1}"

# Choose a safe default cache name.
# - Full run: train_sol_fk.npz
# - Smoke run (MAX_SAMPLES set): train_sol_fk_<N>.npz (avoid overwriting full cache)
if [[ -z "${CACHE_TRAIN:-}" ]]; then
  if [[ "$MAX_SAMPLES" != "-1" ]]; then
    CACHE_TRAIN="$CACHE_DIR/train_sol_fk_${MAX_SAMPLES}.npz"
  else
    CACHE_TRAIN="$CACHE_DIR/train_sol_fk.npz"
  fi
fi

# If MAX_SAMPLES is set for a smoke run, also restrict training/val/test
# datasets to the first N samples so ids always exist in the cache.
SMOKE_INDICES=""
if [[ "$MAX_SAMPLES" != "-1" ]]; then
  SMOKE_INDICES="$MAX_SAMPLES"
fi

mkdir -p "$CACHE_DIR"

# Guard: if we're doing a full run but the default cache is tiny,
# it is almost certainly a leftover smoke cache. Regenerate it.
if [[ "$MAX_SAMPLES" == "-1" && "$FORCE_CACHE" == "0" && -f "$CACHE_TRAIN" ]]; then
  if python - <<PY
import numpy as np
import sys
f = r"$CACHE_TRAIN"
try:
    n = int(np.load(f)["ids"].shape[0])
except Exception:
    sys.exit(0)
print(n)
PY
  then
    N=$(python - <<PY
import numpy as np
f = r"$CACHE_TRAIN"
print(int(np.load(f)["ids"].shape[0]))
PY
    )
    if [[ "$N" -lt 10000 ]]; then
      echo "[WARN] Cache looks too small for a full run (num_samples=$N): $CACHE_TRAIN" >&2
      echo "[WARN] Regenerating full cache (set FORCE_CACHE=0 to keep existing)." >&2
      FORCE_CACHE=1
    fi
  fi
fi

if [[ ! -f "$CACHE_TRAIN" || "$FORCE_CACHE" == "1" ]]; then
  echo "[1/2] Generating FK cache -> $CACHE_TRAIN"
  EXTRA=()
  if [[ "$MAX_SAMPLES" != "-1" ]]; then
    EXTRA+=("--max-samples" "$MAX_SAMPLES")
  fi
  python projects/self_verify_pose/tools/generate_solvability_cache.py \
    "$CACHEGEN_CONFIG" "$POSE_CKPT" \
    --split train \
    --out "$CACHE_TRAIN" \
    --teacher fk \
    --use-gt-visible \
    "${EXTRA[@]}"
else
  echo "[1/2] Cache exists, skip -> $CACHE_TRAIN"
fi

echo "[2/2] Training trust head (pose frozen)"
WORK_DIR_DEFAULT="$REPO_ROOT/work_dirs/self_verify_pose/solvability_cached_trusthead_fk"
WORK_DIR="${WORK_DIR:-$WORK_DIR_DEFAULT}"

PSEUDO_TRUST_CACHE="$CACHE_TRAIN" \
POSE_CKPT="$POSE_CKPT" \
SMOKE_INDICES="$SMOKE_INDICES" \
mim train mmpose "$TRAIN_CONFIG" --work-dir "$WORK_DIR" "$@"
