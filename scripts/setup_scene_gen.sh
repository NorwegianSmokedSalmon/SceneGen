#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SCENE_ENV="${SCENE_ENV:-scene_gen}"
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda run -n "$SCENE_ENV" python --version >/dev/null 2>&1; then
    conda create -y -n "$SCENE_ENV" python=3.12 pip
fi
conda activate "$SCENE_ENV"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-6}"
export PATH="$CUDA_HOME/bin:$PATH"
# Use the CUDA runtime bundled with PyTorch, not mixed system cuBLAS libraries.
export LD_LIBRARY_PATH=
conda env config vars set -n "$SCENE_ENV" CUDA_HOME="$CUDA_HOME" TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" MAX_JOBS="$MAX_JOBS" LD_LIBRARY_PATH=
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130

git submodule update --init --recursive
mkdir -p .cache/scene_gen/vendor
fetch_source() {
    local name="$1" url="$2" revision="$3"
    local target=".cache/scene_gen/vendor/$name"
    if [[ ! -d "$target/.git" ]]; then
        git clone --filter=blob:none "$url" "$target"
        git -C "$target" checkout "$revision"
    elif [[ "$(git -C "$target" rev-parse HEAD)" != "$revision" ]]; then
        echo "Unexpected revision in $target; use the documented revision $revision." >&2
        exit 1
    fi
}
fetch_source sam3 https://github.com/facebookresearch/sam3.git 660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7
fetch_source detectron2 https://github.com/facebookresearch/detectron2.git a2f4a8771ab77e8411c26b27f24f9489a28a2453
fetch_source hloc https://github.com/cvg/Hierarchical-Localization.git c13273bd0ecc2917a35910fd843712a1c6243193
fetch_source entity https://github.com/qqlu/Entity.git 6e7e13ac91ef508088e1b848167c01f19b00b512
git -C .cache/scene_gen/vendor/hloc submodule update --init --recursive
# This branch omitted these Python sources because its .gitignore ignores data/.
if [[ ! -d submodules/CropFormer/mask2former/data ]]; then
    cp -r .cache/scene_gen/vendor/entity/Entityv2/CropFormer/mask2former/data submodules/CropFormer/mask2former/
fi
# SAM3 needs iopath 0.1.10; Detectron2's old upper bound excludes it.
python - <<'PY'
from pathlib import Path
p = Path('.cache/scene_gen/vendor/detectron2/setup.py')
p.write_text(p.read_text().replace('iopath>=0.1.7,<0.1.10', 'iopath>=0.1.10,<0.2'))
p = Path('.cache/scene_gen/vendor/hloc/requirements.txt')
p.write_text(p.read_text().replace('lightglue @ git+https://github.com/cvg/LightGlue\n', 'lightglue @ git+https://github.com/cvg/LightGlue@eb42fee2d71449efb0aa5c10549752b5d75384d8\n'))
PY
python -m pip install "setuptools<81" wheel "numpy==1.26.4"
python -m pip install --no-build-isolation -r requirements-scene-gen.txt \
    -e .cache/scene_gen/vendor/sam3 -e .cache/scene_gen/vendor/hloc \
    'git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787'
python -m pip install --no-build-isolation -e . \
    'git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5' \
    'git+https://github.com/harry7557558/fused-bilagrid@90f9788e57d3545e3a033c1038bb9986549632fe' \
    -e .cache/scene_gen/vendor/detectron2 \
    ./submodules/CropFormer/mask2former/modeling/pixel_decoder/ops
mkdir -p submodules/CropFormer/ckpts
weights=submodules/CropFormer/ckpts/Mask2Former_hornet_3x_576d0b.pth
if [[ ! -s "$weights" ]]; then
    gdown 1W9xNC3PGQrHCZA0Q6TtijxGB_eEbwd63 -O "$weights"
fi
python -m pip check
python -c 'import torch,gsplat.csrc,sam3,detectron2,MultiScaleDeformableAttention; print(torch.__version__,torch.cuda.get_device_name())'
echo "Activate with: conda activate $SCENE_ENV"
