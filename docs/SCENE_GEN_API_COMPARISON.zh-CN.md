# 免费补图 API 与多视图直接生成：实测结果

> 2026-09 原工作区的历史实验记录。数据、权重和生成结果不随仓库分发；当前入口见 [README](../README.md)，安装见 [环境说明](SCENE_GEN_SETUP.zh-CN.md)。

2026-09-18，RTX 5090，Mip-NeRF 360 `room` 的扶手椅 0 和 C 型沙发边桌 1。

**魔搭账号绑定已生效，免费 API 已实际返回六张图片。椅子的「三视图补全 → WorldSculpt → GLB」已完成对照；这次没有观察到可验证的几何提升。边桌补图改变视角，未能形成可信的同相机对照。**

## 结果

椅子使用相同的三张实拍视图、相机、seed=42、WorldSculpt 权重、1024 几何分辨率和约 20 万面导出预算。
每种采样设置均成对比较直接输入与 API 补全输入，没有在不同采样设置之间混算增益。

| 椅子采样设置 | 直接生成：输入视角 IoU | API 补全：输入视角 IoU | 直接生成：额外 10 视角 IoU | API 补全：额外 10 视角 IoU |
| --- | ---: | ---: | ---: | ---: |
| official | 0.818 | 0.807 | 0.815 | 0.808 |
| train | 0.849 | 0.811 | 0.852 | 0.811 |

补图在二维上填掉了椅子中间的白洞，但网格仍存在明显表面起伏、破损和非流形边；两组额外视角指标均下降。不能把“图片更完整”当作“几何更准确”。
`official` 对应作者场景示例的 SS 12 步 / CFG 7.5；`train` 对应训练时的 SS 50 步 / CFG 3。Shape 阶段均为 12 步，其余参数使用作者对应采样函数。

边桌直接生成的输入视角 IoU 为 0.554（official）/ 0.524（train），额外视角为 0.451 / 0.420。桌面有恢复，但细支架欠完整。
六个当前导出网格（四个直接生成、两个补图椅子）都不能当作已完成高质量几何重建。

### 看图和模型

结果根目录：`results/scene_gen_room/multiview_comparison/`。

- 六次 API 输出对照（本地生成产物：`results/scene_gen_room/multiview_comparison/api_attempts.jpg`）
- 椅子 official 对照图（本地生成产物：`results/scene_gen_room/multiview_comparison/comparison_official.jpg`）
- 椅子 train 对照图（本地生成产物：`results/scene_gen_room/multiview_comparison/comparison_train.jpg`）
- 椅子直接生成 GLB（train）（本地生成产物：`results/scene_gen_room/multiview_comparison/direct/instance_0/reconstruction_train/mesh_world.glb`）
- 椅子 API 补全后 GLB（train）（本地生成产物：`results/scene_gen_room/multiview_comparison/completed/instance_0/reconstruction_train/mesh_world.glb`）
- 边桌直接生成 GLB（official）（本地生成产物：`results/scene_gen_room/multiview_comparison/direct/instance_1/reconstruction/mesh_world.glb`）
- `paired_metrics_official.json`、`paired_metrics_train.json`：指标、拓扑统计、文件路径。
- `additional_view_metrics.json`：额外真实视图逐项指标；`comparison_status.json`：汇总及 API 记录。
- 每个 `reconstruction*` 内的 `mesh.pt` 保留简化前高分辨率网格，`run.json` 记录参数，`observed_view_comparison.jpg` 显示三个相机的重投影。

## 免费 API 实际表现

使用魔搭 `Qwen/Qwen-Image-Edit-2511` 免费推理域名 `api-inference.modelscope.cn`，未调用付费替代平台。
账号绑定后六次 ModelScope 请求全部返回了图片；本地轮询超时的任务通过原 task_id 续查，没有重复提交。

| 方案 | 椅子 | 边桌 |
| --- | --- | --- |
| A：服务默认尺寸和采样 | 白洞填上，但返回 760×1280 竖版，整张排版变化；配准后用于实验 | 参考图拼成全景，物体变成正/侧面框架图，拒绝 |
| B：1536×1024、40 步、guidance 4 | 缺口仍在，参考排有移动，拒绝 | 缺口仍在，参考排有移动，拒绝 |
| C：1536×1024、服务默认采样、明确网格和短中文指令 | 补齐，但第三个前视角改成背面，拒绝 | 桌面补齐，但相机与支架改变，配准失败，拒绝 |

A 的旧请求记录曾写入客户端默认步数；当时未把该字段发送给服务端，应按服务默认采样解释。B/C 的实际提交参数保存在 `submitted_parameters`。
更早的 Hugging Face 官方演示受免费 GPU 时长上限阻塞；四步演示返回服务端 RuntimeError。它们没有产生图片，不能参与画质排名。

### 椅子对照如何保留原相机

A 的原始整张图片没有通过严格布局检查。实验额外使用 `prepare_registered_completion.py`：

1. 手工记录三个底排面板的包围框，以等比缩放和白边填充恢复正方形面板。
2. 使用原始可见物体纹理做 SIFT/RANSAC 二维相似变换配准，只允许旋转、等比缩放、平移。
3. 三视角分别有 29/26/22 个配准内点，内点 RMS 误差 1.53/1.41/1.63 像素，已见轮廓覆盖率 99.3%/99.1%/99.6%。
4. 完全不透明的原始 RGBA 像素逐字节保留；半透明抗锯齿边缘按 alpha 合成，把生成内容放在观测颜色后面，避免内部白圈。
5. 新增内容限制在原轮廓凸包内，三个相机记录原样复制。桌子的纹理配准检查没有通过，不导出其补图 `transforms.json`。

这是一条有明确假设的实验支路。二维配准不能证明隐藏表面准确，也不能证明严格三维一致；已知轮廓凸包的限制会阻止恢复凸包外的真实缺失部件。不能将本实验推广成任意物体的通用补全器。

## 指标边界

IoU 使用真实分割掩码，并借助原场景 3DGS 深度判断遮挡。深度只是代理参考，没有真实完整物体网格，因此这些数字不是三维形状准确率。

每个物体额外选择 10 个真实相机，排除三个物体输入图及前后 5 帧，再按相机方向分散选择。这些照片仍参与过场景 3DGS 训练，不能称为独立场景测试集。
本次只有一个有效配对物体、一个 seed、两种采样设置，不能据此断言所有免费 API 或所有补图方法都无效。
世界坐标是规范化场景单位，不是实测米。尚未恢复可信的不可见背面。

本次输入视角来自已有面积/帧间隔筛选，**没有把 H/O/V 热力图选角算法算作已运行**。
当前瓶颈是生成图的视角保持和跨视图几何一致性；此样例应保留真实多视图直接重建作为基线。

## 环境和复跑

基于 Conda `scene_gen_3d` 建立项目内隔离环境，复用 Torch/CUDA。版本与源码提交见 `docs/scene_gen_worldsculpt_environment.json`。
WorldSculpt 官方权重共下载约 13.56 GB，并核验公布的 SHA256；只安装运行所需几何部分。
当前机器需要清空外部 CUDA 动态库路径，避免冲突：

```bash
export LD_LIBRARY_PATH=
SCENE_MV_PY=.cache/scene_gen/envs/worldsculpt/bin/python
SCENE_MV_OUT=results/scene_gen_room/multiview_comparison
```

重新准备真实输入和生成基线：

```bash
"$SCENE_MV_PY" examples/segmentation/prepare_multiview_comparison.py
"$SCENE_MV_PY" examples/segmentation/generate_worldsculpt_objects.py \
  --sampler official --transforms "$SCENE_MV_OUT/direct/instance_0/transforms.json" \
  "$SCENE_MV_OUT/direct/instance_1/transforms.json"
```

API 客户端读取被 Git 忽略的 `.env` 中的 `MODELSCOPE_API_TOKEN`，不执行文件内容、不把令牌写入实验记录。使用者需在自己的账号配置令牌；免费额度由服务端决定。
使用现有记录续查/读取返回图片的示例：

```bash
"$SCENE_MV_PY" scripts/complete_multiview_api.py --provider modelscope \
  --provider-defaults --size 1536x1024 --timeout 1200 \
  --image "$SCENE_MV_OUT/api/instance_0/input_template_grid.png" \
  --prompt "$SCENE_MV_OUT/api/instance_0/prompt_grid.txt" \
  --output "$SCENE_MV_OUT/api/instance_0/completed_modelscope_grid.png"
```

这条示例读取的是已判为视角不合格的 C 方案，用于复现 API 行为。改变图片、提示词或参数必须使用新输出文件名，避免覆盖实验记录。
严格的原布局提取脚本为 `prepare_completed_multiview.py`；本次采用的配准实验可用以下命令复跑：

```bash
"$SCENE_MV_PY" examples/segmentation/prepare_registered_completion.py \
  --direct-transforms "$SCENE_MV_OUT/direct/instance_0/transforms.json" \
  --completed "$SCENE_MV_OUT/api/instance_0/completed_modelscope.png" \
  --panel-boxes "$SCENE_MV_OUT/api/instance_0/portrait_panel_boxes.json" \
  --output "$SCENE_MV_OUT/completed/instance_0"
"$SCENE_MV_PY" examples/segmentation/generate_worldsculpt_objects.py \
  --sampler official --transforms "$SCENE_MV_OUT/completed/instance_0/transforms.json"
"$SCENE_MV_PY" examples/segmentation/inspect_multiview_geometry.py \
  --transforms "$SCENE_MV_OUT/completed/instance_0/transforms.json"
"$SCENE_MV_PY" examples/segmentation/evaluate_multiview_additional_views.py
"$SCENE_MV_PY" examples/segmentation/compare_multiview_results.py --instances 0 --sampler official
```

复跑 train 时，生成和检查命令增加 `--output-name reconstruction_train`，生成和对照汇总使用 `--sampler train`。
更换输入时重建缓存会同时检查相机/图像/权重清单指纹，防止误用旧 GLB。
`*_hardalpha` 是修复内部白圈前的诊断备份，不参与当前报告。

验证包括：相机投影往返误差小于 0.001 像素、非对称三角形光栅方向/深度检查、有效 GLB 载入、配对参数一致、原始不透明像素保留、额外视角逐图评估、API 断点续查。
只上传了作者公开数据集图片，六张原照片及候选图像素的来源验证保存在 `public_data_provenance.json`。

## 来源

- [Mip-NeRF 360 官方数据](https://jonbarron.info/mipnerf360/)
- [WorldSculpt 代码](https://github.com/AlayaLab/WorldSculpt)
- [WorldSculpt 权重](https://huggingface.co/AlayaLab/WorldSculpt)
- [Qwen 官方公开演示](https://huggingface.co/spaces/Qwen/Qwen-Image-Edit-2511)
