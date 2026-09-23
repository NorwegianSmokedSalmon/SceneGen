> 此文档记录旧的全房间 Mesh 实验。当前用户要求仅导入独立物体 Mesh，默认入口已改为 [不含背景的物体场景](SCENE_GEN_HYBRID.zh-CN.md)。下面的全房间资产仅供回看，旧物理测试结果不适用于新混合场景。

> 2026-09 原工作区的历史实验记录。数据、权重和生成结果不随仓库分发；当前入口见 [README](../README.md)，安装见 [环境说明](SCENE_GEN_SETUP.zh-CN.md)。

# 完整房间 Mesh 与 Isaac Sim 场景

入口：`results/scene_gen_room/simulation/scene.usda`。
通用完整网格：`results/scene_gen_room/simulation/scene.glb`。

2026-09-21：已在本机 **Isaac Sim 5.1.0.0** 中实际打开完整场景，8 个网格和 5 个碰撞体载入成功。另完成 5 秒 / 600 步 PhysX 检查：4 个地面球体和 3 个家具方块的接触、支撑与静止检查全部通过。

## 在 Isaac Sim 中使用

本机可直接运行 `LD_LIBRARY_PATH= conda run --no-capture-output -n scene_gen_isaac python scripts/open_scene_isaac.py --scene results/scene_gen_room/simulation/scene.usda`，窗口会保持打开并自动切到房间相机。

1. 保持整个 `simulation/` 目录结构，在 Isaac Sim 用 **File → Open** 打开 `scene.usda`。
2. 场景根节点为 `/World/Room`，默认米制、Z 向上，已设置重力、静态碰撞体和接触材质。
3. 视口切换到 `/World/Cameras/RecordedView_0` 等相机，即可查看原始照片对应位置；也可选中 `/World/Room` 后按 F 查看全景。
4. 添加机器人或动态刚体后进行环境碰撞实验。家具节点目前是静态的，尚未定义质量、关节或可抓取物体的动力学。

也可以将 `assets/room_meshes.usdc` 作为引用加入现有 Isaac 场景；现有场景应使用米制、Z-up，并已有 PhysicsScene。
`scene.glb` 提供完整可见几何，不携带本项目的 USD 物理设置。GLB 按规范转换成 Y-up，USD/PLY/OBJ 为 Z-up，不要在同一场景中按原点直接叠加两种坐标约定。

## 实际内容

全部 311 个已标定视角的 3DGS 深度与真实 RGB 进行 TSDF 融合，视觉网格约 90 万三角形，覆盖客厅与相连拍摄区域。
本次主场景采用全视图实测融合几何；此前 API 补图 / WorldSculpt 生成的单物体网格质量尚未达到可靠替换要求，仍保留在原对比目录。

| 节点 | 可见三角形 | 碰撞三角形 |
| --- | ---: | ---: |
| Background：房间及其他物品 | 874,286 | 541,670 |
| Armchair：扶手椅 | 14,695 | 12,452 |
| SideTable：边桌 | 4,891 | 3,860 |
| Footstool：脚凳 | 6,128 | 5,106 |

另有 `FloorSupport`：从观测地面拟合出的有限厚度支撑盒。四个可见网格的面数总和严格等于原网格，没有在分组时重复添加物体。
彩色外观使用顶点色；USD 中转为线性色并连接 `UsdPreviewSurface`，无需额外贴图。

静态碰撞采用独立的封闭表面壳，2.5 cm 体素、局部闭运算、连续场提取、20 微米退化顶点合并。
保留房间和家具的凹结构，不把整个房间设成单一凸包，不填满房间内部。
USD 碰撞为 `UsdPhysics.CollisionAPI` + `MeshCollisionAPI(approximation="none")`，适用于静态环境。
[NVIDIA 官方碰撞说明](https://nvidia-omniverse.github.io/PhysX/ovphysx/latest/simulation_setup/collision.html)明确区分静态三角网格与动态物体所需的凸分解等表示。

## 尺度和使用边界

- 没有真实尺寸标定。暂按扶手椅顶部距拟合地面 **1.0 m** 估算，比例为 **2.06199079 m / 规范化场景单位**。
- 观测范围约 **8.3 × 3.7 × 2.6 m**。变换矩阵及逆矩阵在 `coordinates.json`；获得实测尺寸后应重新统一缩放全部资产。
- 摩擦系数为初始假设：静摩擦 0.7、动摩擦 0.6、恢复系数 0.02，没有做材料辨识。
- 碰撞近似量级约 5 cm，适合初步环境碰撞/导航实验，不适合用来验证毫米级操作任务。
- 可见网格仍有缺口；深度不足、透明/反光、纹理缺乏和从未被拍摄到的表面没有可靠恢复。碰撞壳封闭不等于完整真实几何已被观测。
- 地面支撑盒是推断的平面补足，覆盖矩形范围；矩形边缘不是经过测量的房间边界。未补造未知墙体。

## 已完成的验证

`asset_validation.json` 记录：

- USD 重新读取成功，默认根节点、相对引用、单位及上方向正确。
- 四组碰撞网格都有限、封闭、绕序一致，无退化三角形。
- GLB 重新读取后仍为 900,000 个三角形。
- 311 个原始相机中心均位于碰撞壳之外，自由位置没有被“房间大凸包”占住。
- 从实际扫描地面及家具表面选择落球点；这些是几何检查，不能代替 PhysX 实测。

本机使用 Isaac Sim **5.1.0.0**（Conda `RoboDojo`）。用户已于 2026-09-21 明确同意 NVIDIA EULA，许可阻塞已解除。

实际结果在 `isaac_validation.json`：7 个射线均命中目标碰撞体；7 个刚体均记录到目标接触，并在 600 步后满足支撑高度和速度检查。测试物体不写回原 USD。

最初使用全部球体时，椅子与边桌上的球体滚落，5 秒内未静止；该诊断结果保存在 `isaac_validation_spheres.json`。家具支撑检查随后采用 6 cm 方块，并额外要求最终仍停在目标表面高度。场景几何和摩擦参数未因此改变。该验证证明抽样静态接触可用，不代表任意机器人任务或真实物理精度已验证。

## 文件

- `scene.usda`：带重力、灯光和观察相机的 Isaac Sim 入口。
- `assets/room_meshes.usdc`：独立可引用的完整房间，包含视觉/碰撞/材质。
- `scene.glb`：通用完整房间可见网格（Y-up）。
- `assets/*_visual.ply`：分组的彩色视觉网格（Z-up）。
- `assets/*_collision.obj` / `*_collision.ply`：独立碰撞网格（Z-up）。
- `mesh_preview.jpg`：四个实拍相机下的实际网格渲染。
- `manifest.json` / `coordinates.json` / `reconstruction.json` / `asset_validation.json`：来源、尺度、参数与验证。
- `cache/`：深度和碰撞构建缓存，导入 Isaac Sim 时不需要。

## 复跑

在已有项目环境上增加了 `usd-core==26.8`，没有修改原 Conda 环境中的 Torch/CUDA。

```bash
export LD_LIBRARY_PATH=
SCENE_MESH_PY=.cache/scene_gen/envs/worldsculpt/bin/python
OMP_NUM_THREADS=8 "$SCENE_MESH_PY" examples/mesh/reconstruct_scene_simulation.py
OMP_NUM_THREADS=8 "$SCENE_MESH_PY" examples/mesh/export_isaac_scene.py
OMP_NUM_THREADS=8 "$SCENE_MESH_PY" examples/mesh/validate_scene_assets.py
```

深度已缓存，重跑不会重复渲染。仅重新估算尺度与融合时可以使用 `--stage coordinates` 和 `--stage fuse`。
如有椅子真实高度，用 `--chair-height-m 数值`；如果有从规范化场景到米的准确比例，用 `--meters-per-scene-unit 数值`。
新的尺度会改变几何、碰撞和测试点，之后须重新运行导出和验证。

使用 Isaac Sim 自带或对应 Conda 的 Python 复跑验证：

```bash
LD_LIBRARY_PATH= conda run --no-capture-output -n scene_gen_isaac python \
  scripts/validate_scene_isaac.py --directory results/scene_gen_room/simulation
```

脚本不会自动接受许可。
