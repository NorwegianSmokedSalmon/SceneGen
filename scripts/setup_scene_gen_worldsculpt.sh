#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ -n "${SCENE_GEN_BASE_PYTHON:-}" ]]; then
    BASE_PYTHON="$SCENE_GEN_BASE_PYTHON"
else
    BASE_PYTHON="$(conda run -n "${SCENE_3D_ENV:-scene_gen_3d}" python -c 'import sys; print(sys.executable)')"
fi
PY="$PWD/.cache/scene_gen/envs/worldsculpt/bin/python"
export LD_LIBRARY_PATH="" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}" MAX_JOBS="${MAX_JOBS:-4}"
export PATH="$CUDA_HOME/bin:$PATH"
if [[ ! -x "$PY" ]]; then "$BASE_PYTHON" -m venv --system-site-packages .cache/scene_gen/envs/worldsculpt; fi
"$PY" -c 'import torch; assert torch.__version__ == "2.10.0+cu130", torch.__version__; assert torch.cuda.is_available()'
"$PY" -m pip install --no-cache-dir -r requirements-scene-gen-worldsculpt.txt
"$PY" -m pip install --no-cache-dir --no-deps \
  'https://github.com/SHI-Labs/NATTEN/releases/download/v0.21.6/natten-0.21.6%2Btorch2100cu130-cp312-cp312-linux_x86_64.whl' \
  'https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl'
clone_at() {
  local name="$1" url="$2" commit="$3" target=".cache/scene_gen/vendor/$1"
  if [[ ! -d "$target" ]]; then
    mkdir -p "$target"
    git -C "$target" init
    git -C "$target" remote add origin "$url"
    git -C "$target" fetch --depth 1 origin "$commit"
    git -C "$target" checkout --detach FETCH_HEAD
    git -C "$target" submodule update --init --recursive --depth 1
  fi
  [[ "$(git -C "$target" rev-parse HEAD)" == "$commit" ]] || { echo "Unexpected commit in $target; preserving checkout." >&2; exit 1; }
}
clone_at WorldSculpt https://github.com/AlayaLab/WorldSculpt.git fac6b83282368063a34e648a07c6cff169e05152
clone_at FlexGEMM https://github.com/JeffreyXiang/FlexGEMM.git 6dd94a859c26ee8246888502eada3dd8ad85532e
clone_at CuMesh https://github.com/JeffreyXiang/CuMesh.git 12289e1062f0603f2f0d0771b02e1395d247f26f
clone_at TRELLIS.2 https://github.com/microsoft/TRELLIS.2.git 75fbf0183001ed9876c8dbb35de6b68552ee08bd
clone_at nvdiffrast https://github.com/NVlabs/nvdiffrast.git 253ac4fcea7de5f396371124af597e6cc957bfae
for entry in 'flex_gemm FlexGEMM' 'cumesh CuMesh' 'o_voxel TRELLIS.2/o-voxel' 'nvdiffrast nvdiffrast'; do
  read -r module source <<< "$entry"
  if ! "$PY" -c "import $module"; then
    "$PY" -m pip install --no-cache-dir --no-build-isolation --no-deps ".cache/scene_gen/vendor/$source"
  fi
done
"$PY" scripts/download_worldsculpt_geometry.py
