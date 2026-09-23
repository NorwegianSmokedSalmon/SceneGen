#!/usr/bin/env bash
# Furniture-focused pipeline for the downloaded Mip-NeRF 360 room scene.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SCENE_ENV:-scene_gen}"
export LD_LIBRARY_PATH=
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
SCENE_DATA="$(realpath -m "${1:-data/mipnerf360_indoor/room}")"
SCENE_RESULT="$(realpath -m "${2:-results/scene_gen_room}")"
SCENE_STEPS="${3:-7000}"
[[ "$SCENE_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo 'STEPS must be positive' >&2; exit 2; }
if [[ ! -d "$SCENE_DATA/sparse/0" ]]; then
    echo 'Download data first: python scripts/download_indoor_scene.py --scenes room bonsai' >&2
    exit 1
fi
SCENE_CKPT="$SCENE_RESULT/ckpts/ckpt_$((SCENE_STEPS-1))_rank0.pt"
if [[ ! -f "$SCENE_CKPT" ]]; then
    python examples/extended_trainer.py --data-dir "$SCENE_DATA" --data-factor 1 \
        --result-dir "$SCENE_RESULT" --max-steps "$SCENE_STEPS" \
        --save-steps "$SCENE_STEPS" --ply-steps "$SCENE_STEPS" --eval-steps "$SCENE_STEPS" \
        --refine-stop-iter "$((SCENE_STEPS*11/14))" \
        --disable-viewer --disable-video --dataset-preload "${SCENE_PRELOAD:-cuda}" \
        --no-use-bilateral-grid --no-use-fused-bilagrid \
        --depth-dir-name None --normal-dir-name None \
        --depth-loss-weight 0 --render-normal-loss-weight 0 --surf-normal-loss-weight 0
fi
python examples/segmentation/segment_furniture_sam3.py --data-dir "$SCENE_DATA" --output-subdir furniture
if [[ -d "$SCENE_DATA/cluster_result" ]]; then
    mkdir -p "$SCENE_RESULT"
    mv "$SCENE_DATA/cluster_result" "$SCENE_RESULT/previous_cluster_result_$(date +%Y%m%d_%H%M%S_%N)"
fi
python examples/segmentation/instascene_gauscluster.py --data_dir "$SCENE_DATA" \
    --data_factor 1 --normalize_world_space --ckpt "$SCENE_CKPT" \
    --mask_subdir furniture --min_mask_pixels 300
# Verified room IDs. Different scenes must supply their inspected IDs.
read -r -a scene_ids <<< "${SCENE_INSTANCE_IDS:-0 1 2}"
python examples/segmentation/export_observed_instance_views.py --data-dir "$SCENE_DATA" \
    --ckpt "$SCENE_CKPT" --instances "${scene_ids[@]}"
python examples/segmentation/densify_instance_labels.py --data-dir "$SCENE_DATA" \
    --ckpt "$SCENE_CKPT" --instances "${scene_ids[@]}"
python examples/segmentation/assemble_multiview_templates.py \
    --candidate_views_dir "$SCENE_DATA/cluster_result/candidate_views" --mask_subdir samrefiner \
    --output_dir "$SCENE_DATA/cluster_result/inference_templates" --require_mask
if [[ "${SCENE_RUN_3D:-1}" == 0 ]]; then
    echo "Scene reconstruction and observed object views prepared: $SCENE_RESULT"
    exit 0
fi
conda activate "${SCENE_3D_ENV:-scene_gen_3d}"
export LD_LIBRARY_PATH=
python examples/segmentation/generate_sam3d_objects.py \
    --views "$SCENE_DATA/cluster_result/candidate_views" --ckpt "$SCENE_CKPT" \
    --labels "$SCENE_DATA/cluster_result/instance_labels_dense.npy" \
    --output "$SCENE_RESULT/sam3d_furniture" --instances "${scene_ids[@]}"
