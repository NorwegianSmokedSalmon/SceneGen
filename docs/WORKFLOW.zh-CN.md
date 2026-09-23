# 分阶段运行

本说明区分通用底层工具与当前 `Mip-NeRF 360 room` 实验编排。默认路径均相对于仓库根目录；从仓库根目录运行命令。

## 1. 准备数据、3DGS 与原相机

```bash
conda run -n scene_gen python scripts/download_indoor_scene.py --scenes room
SCENE_RUN_3D=0 bash scripts/run_scene_gen_indoor.sh
```

输出包括 `results/scene_gen_room/ckpts/ckpt_6999_rank0.pt`、`data/mipnerf360_indoor/room/cluster_result/gauscluster_tracking_data.pt` 和物体候选视角。已有同一步数检查点会复用，旧聚类结果会先备份。

自己的原始图片可通过 `examples/preprocess/run_hloc_sfm.py` 获得 `images/` 与 `sparse/0/`；`scripts/run_scene_gen.sh` 是保留的 CropFormer/SAM 3D 历史编排，输出与下面 WorldSculpt 流程不同。

## 2. 深度代理、全物体关联与选视图

```bash
conda run -n scene_gen python examples/mesh/reconstruct_scene_simulation.py --stage render
conda run -n scene_gen python examples/segmentation/full_scene_inventory.py
conda run -n scene_gen python examples/segmentation/track_full_scene_objects.py
```

这里只渲染深度缓存，不做房间 TSDF mesh。输出位于 `results/scene_gen_room/simulation/cache/` 和 `results/scene_gen_room/full_objects/`。

审核 `inventory_preview.jpg`、每个物体的 `selected_views.jpg` 与 `view_selection.json`。选视图使用 H 热力图、遮挡 O、可见率 V、相机方向差和新增代理表面覆盖；这些量都不能直接当作真实完整表面的覆盖率。

`refine_full_scene_inventory.py` 内的合并/排除编号，以及 `prepare_*`、`refine_*`、`retry_remaining_full_objects.py` 中的部分规则属于原 room 实验。新跟踪结果必须重新审核并修改规则，不能直接照搬编号。审核后可用 `track_full_scene_objects.py --prepare-only` 重建输入。

## 3. 多视图几何和可选补图

```bash
export LD_LIBRARY_PATH=
SCENE_MV_PY=.cache/scene_gen/envs/worldsculpt/bin/python
"$SCENE_MV_PY" examples/segmentation/generate_worldsculpt_objects.py   --sampler train --transforms   results/scene_gen_room/full_objects/objects/object_007/input/transforms.json
```

`--transforms` 可接受多个物体的配置路径。编号 `007` 只是原实验示例，运行前应检查实际 inventory。每个目录含 `view_*.png`、相机与归一化参数，输出包含 `reconstruction/mesh_world.glb`、`run.json`、输入指纹和模型版本。`--discard-raw` 只在成功导出后删除本次中间张量。

可选 API 原视角补图：

```bash
"$SCENE_MV_PY" scripts/complete_full_object_views.py --ids 7 3 22
```

该编排仅支持原 room 的椅子、沙发、边桌提示，并要求公开数据下载清单。通用调用入口是 `scripts/complete_multiview_api.py --help`。只接受通过原相机配准和可见像素保留检查的结果；accepted_views 为 0 时不能声称使用了有效补图。它尚未实现一致的环绕新视角生成。

`--silhouette-guide` 是实验性轮廓/深度约束。当前对遮挡区域的处理仍可能错误裁掉真实结构，已列入已知问题；不能将此开关当作完整性保证。

## 4. 检查、选择与修复

```bash
"$SCENE_MV_PY" examples/segmentation/inspect_full_scene_objects.py --branch input
```

其他分支或生成目录需配套使用 `--branch`、`--output-name`。先检查各物体的灰模、输入轮廓重投影、主体连续性和隐藏表面，再选择候选。

`examples/mesh/select_and_repair_full_objects.py` 保留了原 room 实验的约束候选要求，需先准备它要求的候选或按新实验调整 GUIDED 配置；不是只运行一次原始生成就一定可导出。它输出 `selected_meshes.json`，并执行 MeshFix / 体素封闭。闭合后的碎片依然可能组成残缺物体。

## 5. 坐标、回填和导出

在深度缓存已有后拟合地面和米制坐标。二选一：

```bash
# 用实际测得的尺度，替换数值；不会要求旧实验的椅子文件。
conda run -n scene_gen python examples/mesh/reconstruct_scene_simulation.py   --stage coordinates --meters-per-scene-unit 2.0

# 或从已经审核的椅子和假设高度估计尺度。
conda run -n scene_gen python examples/mesh/reconstruct_scene_simulation.py   --stage coordinates --chair-mesh /path/to/accepted_chair_world.ply --chair-height-m 1.0
```

上面的 2.0 是命令格式示例，不是任何新场景的已测标定值。估计尺度会明确写入 `coordinates.json`。

```bash
"$SCENE_MV_PY" examples/mesh/constrain_object_extents.py
"$SCENE_MV_PY" examples/mesh/assemble_full_object_scene.py
"$SCENE_MV_PY" examples/mesh/audit_object_completeness.py
```

范围约束保留裁切前版本并检查输入轮廓变化。最终输出位于 `results/scene_gen_room/full_objects/scene/`：GLB 为米制 Y-up，USD 为米制 Z-up，保留独立物体节点。USD 不带背景 GS/点云/房间 mesh，包含不可见地面碰撞支撑。

## 6. Isaac 查看与实际渲染

```bash
LD_LIBRARY_PATH= conda run --no-capture-output -n scene_gen_isaac python scripts/open_scene_isaac.py
LD_LIBRARY_PATH= conda run --no-capture-output -n scene_gen_isaac python scripts/validate_full_object_scene_isaac.py
```

检查输出图和 JSON 报告。验证范围是 stage 加载、网格/碰撞数量、四视角可见性与静态场景时间线启动；不证明动态稳定性、接触准确性或真实表面完整。

## 核心文件契约

| 文件 | 含义 |
| --- | --- |
| `gauscluster_tracking_data.pt` | 标定相机、图像路径和关联信息；仅加载可信的本流程文件 |
| `inventory.json` | 每个实例的类别、观测、中心、尺度、相对目录 |
| `input/transforms.json` | 裁剪内参、原 OpenCV 相机、归一化生成相机及图像路径 |
| `selected_meshes.json` | 被选择候选及最终修复网格的相对路径 |
| `coordinates.json` | `world_to_simulation`、尺度来源与地面估计 |
| `scene/manifest.json` | 每个物体来源、位置、顶点/面数与质量限制 |
| `scene/geometry_completeness_audit.json` | 碎片、视角分布与轮廓拟合诊断；不等同完整性认证 |
