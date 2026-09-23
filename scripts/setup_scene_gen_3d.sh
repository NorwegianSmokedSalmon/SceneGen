#!/usr/bin/env bash
# Requires the scene_gen environment created by setup_scene_gen.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
SCENE_3D_ENV="${SCENE_3D_ENV:-scene_gen_3d}"
if ! conda run -n "$SCENE_3D_ENV" python --version >/dev/null 2>&1; then
    conda create -y -n "$SCENE_3D_ENV" --clone "${SCENE_ENV:-scene_gen}"
fi
conda activate "$SCENE_3D_ENV"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-6}"
export LD_LIBRARY_PATH=
export LIDRA_SKIP_INIT=1
export ATTN_BACKEND=xformers
export SPARSE_ATTN_BACKEND=xformers
export PATH="$CUDA_HOME/bin:$PATH"
# Required by PyTorch3D Pulsar with CUDA 13; upstream main uses these flags too.
export NVCC_FLAGS='--device-entity-has-hidden-visibility=false --static-global-template-stub=false'
conda env config vars set -n "$SCENE_3D_ENV" CUDA_HOME="$CUDA_HOME" \
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" MAX_JOBS="$MAX_JOBS" \
    LD_LIBRARY_PATH= LIDRA_SKIP_INIT=1 ATTN_BACKEND=xformers SPARSE_ATTN_BACKEND=xformers
python -c 'import torch; assert torch.__version__.startswith("2.10.0") and torch.version.cuda == "13.0", "Run setup_scene_gen.sh first"'
python -m pip install --no-deps xformers==0.0.34 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-scene-gen-3d.txt
fetch_source() {
    local name="$1" url="$2" revision="$3"
    local target=".cache/scene_gen/vendor/$name"
    if [[ ! -d "$target/.git" ]]; then
        git clone --filter=blob:none "$url" "$target"
        git -C "$target" checkout "$revision"
    elif [[ "$(git -C "$target" rev-parse HEAD)" != "$revision" ]]; then
        echo "Unexpected revision in $target; expected $revision." >&2
        exit 1
    fi
}
fetch_source sam-3d-objects https://github.com/facebookresearch/sam-3d-objects.git f91db411c50efee93d8db7aeb323885650f6f722
fetch_source pytorch3d https://github.com/facebookresearch/pytorch3d.git 75ebeeaea0908c5527e7b1e305fbc7681382db47
fetch_source dinov2 https://github.com/facebookresearch/dinov2.git 7764ea0f912e53c92e82eb78a2a1631e92725fc8
patch_file="$PWD/scripts/patches/sam3d-gaussian-only.patch"
if git -C .cache/scene_gen/vendor/sam-3d-objects apply --reverse --check "$patch_file" 2>/dev/null; then
    echo "SAM 3D Gaussian-only patch already applied."
else
    git -C .cache/scene_gen/vendor/sam-3d-objects apply --check "$patch_file"
    git -C .cache/scene_gen/vendor/sam-3d-objects apply "$patch_file"
fi
if ! python -c 'import torch; from pytorch3d import _C' >/dev/null 2>&1; then
    FORCE_CUDA=1 python -m pip install --no-build-isolation --no-deps -e .cache/scene_gen/vendor/pytorch3d
fi
python -m pip check
python - <<'PY'
import torch
from pytorch3d.ops import knn_points
import xformers.ops as xops
import spconv.pytorch as spconv
p = torch.randn(1, 8, 3, device='cuda')
assert knn_points(p, p, K=1).dists.max() == 0
q = torch.randn(1, 32, 8, 64, device='cuda', dtype=torch.float16)
assert torch.isfinite(xops.memory_efficient_attention(q, q, q)).all()
coords = torch.tensor([[0,x,y,z] for x in range(4) for y in range(4) for z in range(4)], device='cuda', dtype=torch.int32)
a = spconv.SparseConvTensor(torch.rand(64,16,device='cuda'), coords, [4,4,4], 1)
assert spconv.SubMConv3d(16,32,3,padding=1).cuda()(a).features.shape == (64,32)
print('SAM 3D native dependency checks passed')
PY
echo "Ready: conda activate $SCENE_3D_ENV"
echo 'Weights: python scripts/download_scene_gen_models.py --model sam3d'
