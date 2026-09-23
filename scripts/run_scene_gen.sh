#!/usr/bin/env bash
# Reconstruct, segment, generate SAM 3D objects and compose a Gaussian scene.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 DATA_DIR [RESULT_DIR] [STEPS]" >&2
    exit 2
fi
SCENE_DATA="$(realpath "$1")"
SCENE_RESULT="$(realpath -m "${2:-results/scene_gen}")"
SCENE_STEPS="${3:-1000}"
[[ "$SCENE_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo 'STEPS must be a positive integer' >&2; exit 2; }
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SCENE_ENV:-scene_gen}"
export LD_LIBRARY_PATH=
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# Automatically use the verified local ModelScope checkpoint when available.
if [[ -z "${SAM3_CKPT+x}" && -f .cache/scene_gen/models/sam3/download_manifest.json ]]; then
    export SAM3_CKPT="$PWD/.cache/scene_gen/models/sam3/sam3.pt"
fi
if [[ -n "${SAM3_CKPT:-}" && ! -f "$SAM3_CKPT" ]]; then
    echo "SAM3_CKPT does not point to a file." >&2
    exit 1
fi
if [[ ! -d "$SCENE_DATA/sparse" ]]; then
    [[ -d "$SCENE_DATA/input" ]] || { echo 'Expected sparse/ + images/, or an input/ image folder.' >&2; exit 1; }
    [[ ! -e "$SCENE_DATA/distorted" ]] || { echo 'SfM working directory already exists; inspect it before rerunning SfM.' >&2; exit 1; }
    python examples/preprocess/run_hloc_sfm.py --input_image_dir "$SCENE_DATA/input" \
        --camera_model OPENCV --matching_method "${SCENE_MATCHING_METHOD:-exhaustive}" \
        --matcher_type superpoint+lightglue --gpu_ba off
fi
SCENE_CKPT="$SCENE_RESULT/ckpts/ckpt_$((SCENE_STEPS - 1))_rank0.pt"
if [[ ! -f "$SCENE_CKPT" ]]; then
    python examples/extended_trainer.py --data-dir "$SCENE_DATA" --data-factor 1 \
        --result-dir "$SCENE_RESULT" --max-steps "$SCENE_STEPS" \
        --save-steps "$SCENE_STEPS" --ply-steps "$SCENE_STEPS" --eval-steps "$SCENE_STEPS" \
        --disable-viewer --disable-video --dataset-preload "${SCENE_PRELOAD:-cuda}" \
        --no-use-bilateral-grid --no-use-fused-bilagrid \
        --depth-dir-name None --normal-dir-name None \
        --depth-loss-weight 0 --render-normal-loss-weight 0 --surf-normal-loss-weight 0
else
    echo "Using existing checkpoint: $SCENE_CKPT"
fi
python submodules/CropFormer/run_cropformer.py \
    --config-file submodules/CropFormer/configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml \
    --scene_dir "$SCENE_DATA" --image_path_pattern 'images/*' --dataset scannet \
    --opts MODEL.WEIGHTS submodules/CropFormer/ckpts/Mask2Former_hornet_3x_576d0b.pth
# Preserve the previous run so stale instance IDs cannot enter new templates.
if [[ -d "$SCENE_DATA/cluster_result" ]]; then
    mkdir -p "$SCENE_RESULT"
    mv "$SCENE_DATA/cluster_result" "$SCENE_RESULT/previous_cluster_result_$(date +%Y%m%d_%H%M%S_%N)"
fi
python examples/segmentation/instascene_gauscluster.py --data_dir "$SCENE_DATA" \
    --data_factor 1 --normalize_world_space --ckpt "$SCENE_CKPT" --min_mask_pixels 100
python examples/segmentation/prepare_atomic_geometry.py --data_dir "$SCENE_DATA" \
    --ckpt "$SCENE_CKPT" --skip_align
python examples/segmentation/generate_instance_views.py --data_dir "$SCENE_DATA" \
    --ckpt "$SCENE_CKPT" --render_res 512 --num_samples 256 --candidate_mode geometric \
    --export_topk 3 --no-constrain_candidates_to_scene
SCENE_VIEWS="$SCENE_DATA/cluster_result/candidate_views"
if [[ -n "${SAM3_CKPT:-}" ]]; then
    [[ -f "$SAM3_CKPT" ]] || { echo 'SAM3_CKPT does not point to a file.' >&2; exit 1; }
    python examples/segmentation/refine_instance_masks.py --candidate_views_dir "$SCENE_VIEWS" \
        --sam3_ckpt_path "$SAM3_CKPT" --pattern 'instance_*_rank_*.png' --output_subdir samrefiner
    python examples/segmentation/assemble_multiview_templates.py --candidate_views_dir "$SCENE_VIEWS" \
        --mask_subdir samrefiner --output_dir "$SCENE_DATA/cluster_result/inference_templates" --require_mask
else
    python examples/segmentation/assemble_multiview_templates.py --candidate_views_dir "$SCENE_VIEWS" \
        --mask_subdir projected_mask --output_dir "$SCENE_DATA/cluster_result/coarse_templates" \
        --cell_size 512 --require_mask
    echo 'Coarse-mask templates only. Download SAM3 with: python scripts/download_scene_gen_models.py --model sam3'
fi
echo "Reconstruction and segmentation complete: $SCENE_DATA/cluster_result"
if [[ "${SCENE_RUN_3D:-auto}" != 0 && -n "${SAM3_CKPT:-}" ]] &&
   [[ -f .cache/scene_gen/models/sam-3d-objects/download_manifest.json ]] &&
   conda run -n "${SCENE_3D_ENV:-scene_gen_3d}" python --version >/dev/null 2>&1; then
    bash scripts/run_scene_gen_3d.sh "$SCENE_DATA" "$SCENE_CKPT" "$SCENE_RESULT/sam3d_all"
elif [[ "${SCENE_RUN_3D:-auto}" == 1 ]]; then
    echo 'SAM 3D requested but its environment, checkpoints or SAM3 masks are missing.' >&2
    exit 1
else
    echo 'For SAM 3D generation: run setup_scene_gen_3d.sh and download_scene_gen_models.py --model sam3d.'
fi
