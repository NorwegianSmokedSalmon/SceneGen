# 环境配置

以下是原工作区实际使用的安装配方，整理后使用仓库相对路径与 Conda 环境名。新克隆不附带环境、模型或数据。

| 用途 | 环境 | 配置 |
| --- | --- | --- |
| 重建、SAM3、物体关联 | `scene_gen` | Python 3.12、Torch 2.10.0+cu130、torchvision 0.25.0 |
| SAM 3D 与共享三维依赖 | `scene_gen_3d` | 由 `scene_gen` 克隆，加入 PyTorch3D 等 |
| WorldSculpt 几何 | `.cache/scene_gen/envs/worldsculpt` | 基于 `scene_gen_3d` 的隔离覆盖环境 |
| Isaac Sim | `scene_gen_isaac`（示例名称） | 原实验为 Isaac Sim 5.1.0.0 / Python 3.11，独立安装 |

验证硬件为 RTX 5090 32 GB，系统 CUDA toolkit 13.0。脚本默认编译架构 `12.0`，可通过 `TORCH_CUDA_ARCH_LIST`、`CUDA_HOME`、`MAX_JOBS` 覆盖；WorldSculpt 的 NATTEN wheel 固定为 Python 3.12 / Torch 2.10 / CUDA 13.0，不是任意环境通用 wheel。

## 重建和分割

```bash
git submodule update --init --recursive
bash scripts/setup_scene_gen.sh
conda run -n scene_gen python scripts/download_scene_gen_models.py --model sam3
```

脚本会安装本仓库 gsplat、固定提交的 SAM3/HLOC/Detectron2、CropFormer 的兼容扩展。源代码与权重分别位于 `.cache/scene_gen/vendor/` 和 `.cache/scene_gen/models/`；CropFormer 权重在 `submodules/CropFormer/ckpts/`。下载脚本校验权重，均不进入 Git。

`environment.scene-gen.yml` 只声明基础 Conda 环境，CUDA 扩展和完整 Python 依赖由安装脚本处理。不要将它误当成完整 lockfile。

## 物体生成

```bash
bash scripts/setup_scene_gen_3d.sh
bash scripts/setup_scene_gen_worldsculpt.sh
.cache/scene_gen/envs/worldsculpt/bin/python -m pip install -r requirements-scene-gen-simulation.txt
```

WorldSculpt 安装脚本通过 Conda 查找基础 Python，不依赖 `~/anaconda3`。已有兼容环境时可显式设置 `SCENE_GEN_BASE_PYTHON=/path/to/python`。它会下载约 13.56 GB 的几何模型；CUDA 扩展、其他模型和缓存另占空间。

仅运行历史 SAM 3D 高斯生成时，还需：

```bash
conda run -n scene_gen_3d python scripts/download_scene_gen_models.py --model sam3d
```

软件版本和固定提交见 [基础环境记录](scene_gen_environment.json)、[WorldSculpt 记录](scene_gen_worldsculpt_environment.json)。它们记录原实验的软件版本，不表示已在每台机器重新验证。

## API 与 Isaac

```bash
cp .env.example .env
```

在 `.env` 填写自己的 `MODELSCOPE_API_TOKEN`，或用同名环境变量。客户端不会执行 `.env` 内容；实验记录排除令牌和上传图片的 base64 内容。API 请求会将所选图片发送给服务商。

Isaac Sim 按 NVIDIA 安装流程单独配置，确认 `from isaacsim import SimulationApp` 可用后再运行打开/验证脚本。脚本不会自动安装 Isaac、接受许可或修改系统默认 Python。

原机器存在系统 CUDA 动态库与 Torch 自带库冲突，所以命令使用 `LD_LIBRARY_PATH=`。此设置由各脚本或单条命令作用于对应进程。

## 验证

```bash
python scripts/check_source_tree.py
LD_LIBRARY_PATH= conda run -n scene_gen_3d python -m unittest discover -s tests -p 'test_scene_gen_geometry.py'
```

完整几何生成和 Isaac 渲染需要对应模型、数据与 GPU。轻量源码检查通过不表示物体表面完整。
