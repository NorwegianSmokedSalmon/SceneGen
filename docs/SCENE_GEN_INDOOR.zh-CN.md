# 室内桌椅场景

> 2026-09 原工作区的历史实验记录。数据、权重和生成结果不随仓库分发；当前入口见 [README](../README.md)，安装见 [环境说明](SCENE_GEN_SETUP.zh-CN.md)。

已改用 [Mip-NeRF 360 官方数据](https://jonbarron.info/mipnerf360/)的 `room` 作为主场景。
照片里有方形茶几、扶手椅、脚凳、沙发和音箱；本次选择前三件家具进行三维生成和替换。
`bonsai` 也已下载作为备用室内物体场景，但没有在它上面训练或生成。

| 场景 | 已准备图片 | 状态 |
| --- | ---: | --- |
| `room` | 311 张，779 × 519 | 完成 3DGS、家具分割、聚类、SAM 3D 和场景组合 |
| `bonsai` | 292 张 | 完成图片、COLMAP 相机和稀疏点准备；主要是盆景、圆桌和自行车 |

数据位于 `data/mipnerf360_indoor/`。通过 HTTP Range 从作者发布的 `360_v2.zip` 中
选择性读取 `images_4/` 和 `sparse/0/`，实测传输约 259 MB，无需下载整个 12.5 GB 压缩包。
每个场景的 `download_manifest.json` 记录来源、官方 ZIP ETag、文件大小和 SHA256；
ZIP CRC 也已检查。`source_sparse/0/` 保留原始 COLMAP 模型，`sparse/0/` 按实际图片尺寸
同步缩放了内参和特征点坐标，所以后续使用 `--data-factor 1`。
本次使用数据集自带的相机位姿，没有重复做 SfM。

## 本次运行结果

- `scene_gen` 环境在 RTX 5090 上用全部 311 张照片训练 7000 步，最后 1500 步停止增密。
- 原始场景：2,867,415 个高斯。
- SAM3 文本提示为 `chair`、`table`、`footstool`；311 帧共得到 698 个二维检测实例。
- GausCluster 跨视角聚类后，选择实例 `0`（扶手椅）、`1`（茶几）、`2`（脚凳）。SAM3 将脚凳多数识别为 chair。
- 104 个分散视角进行深度一致性投票，扩展原始稀疏实例标签；至少 3 次可见支持，前景一致率至少 65%。
  可见背景也计入投票分母，避免用全局最近邻把墙面、地板都归给家具。
- 各实例输入采用真实照片、SAM3 + 聚类后的掩码，以及同相机渲染的 3DGS 深度。
  排名靠前的候选图优先选择完整、不碰图像边缘的物体，共导出 9 个视角。
- SAM 3D 对每件家具使用排名第一的单视角生成；生成高斯约为 26.8 万、43.3 万、40.7 万。
- 最终组合场景：3,887,165 个高斯。

抽样重投影指标为 PSNR 32.51 dB、SSIM 0.9452、LPIPS 0.0749。
**当前分支的训练集包含全部图像，评估图片与训练集重叠；这些是拟合检查指标，不是独立测试集性能。**

物体已可清楚辨认。扶手椅和脚凳的放置较贴合，颜色与局部形状仍有变化；茶几受到桌面物品遮挡影响，
完整桌面的形状和姿态匹配较粗。当前仍使用预测旋转、原实例稳健三维范围约束的平移和统一尺度，
没有做精细 ICP、桌上物品支撑关系调整或碰撞优化。玩具、碗等非目标物体保留在原场景中。

## 运行

当前机器环境与模型已经齐备，复跑主场景：

```bash
cd /path/to/SceneGen
bash scripts/run_scene_gen_indoor.sh
```

默认使用 `data/mipnerf360_indoor/room`、`results/scene_gen_room`、7000 步。
已存在对应训练检查点时会复用；重新运行聚类前会把旧 `cluster_result` 备份到结果目录。
实例 `0 1 2` 是本次 room 已检查的选择，其他场景必须先检查实例，再设置 `SCENE_INSTANCE_IDS`。

重新下载或在新目录准备数据：

```bash
conda activate scene_gen
python scripts/download_indoor_scene.py --scenes room bonsai
```

相关入口：

- `scripts/download_indoor_scene.py`：官方数据的按需下载与相机缩放。
- `examples/segmentation/segment_furniture_sam3.py`：桌椅文本分割。
- `examples/segmentation/export_observed_instance_views.py`：真实照片视角及配套深度导出。
- `examples/segmentation/densify_instance_labels.py`：带深度测试的多视角实例标签扩展。
- `examples/segmentation/generate_sam3d_objects.py --labels ...`：指定要替换的稠密实例标签。

## 查看结果

- 对比图：`results/scene_gen_room/sam3d_furniture/preview.jpg`
- 组合场景：`results/scene_gen_room/sam3d_furniture/scene_composed.ply`、`scene_composed.pt`
- 生成物体：`results/scene_gen_room/sam3d_furniture/instance_*/object_world.ply`
- 原始重建：`results/scene_gen_room/ply/point_cloud_6999.ply`
- 验证记录：原工作区中的实验结果，本代码仓库不附带运行日志。
- 运行参数与姿态：`results/scene_gen_room/sam3d_furniture/manifest.json`
- 日志：`.cache/scene_gen/room-*.log`

```bash
conda activate scene_gen
python examples/simple_viewer.py \
  --ckpt results/scene_gen_room/sam3d_furniture/scene_composed.pt \
  --output_dir results/scene_gen_room/sam3d_furniture --port 8080
```

已检查相机与图片尺寸一致、4 项几何回归通过、所有组合高斯参数有限，
且未替换的高斯逐项保持原值，三个生成物体完整合入新场景。
