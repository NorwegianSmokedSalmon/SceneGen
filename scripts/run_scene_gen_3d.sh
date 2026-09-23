#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 DATA_DIR CHECKPOINT OUTPUT_DIR [INSTANCE_ID ...]" >&2
    exit 2
fi
SCENE_DATA="$(realpath "$1")"
SCENE_CKPT="$(realpath "$2")"
SCENE_OUTPUT="$(realpath -m "$3")"
shift 3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SCENE_3D_ENV:-scene_gen_3d}"
export LD_LIBRARY_PATH=
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
extra_args=()
if [[ $# -gt 0 ]]; then
    extra_args=(--instances "$@")
fi
python examples/segmentation/generate_sam3d_objects.py \
    --views "$SCENE_DATA/cluster_result/candidate_views" --ckpt "$SCENE_CKPT" \
    --output "$SCENE_OUTPUT" "${extra_args[@]}"
