# 独立物体 Mesh 与 Isaac 场景

> 2026-09 原工作区的历史实验记录。数据、权重和生成结果不随仓库分发；当前入口见 [README](../README.md)，安装见 [环境说明](SCENE_GEN_SETUP.zh-CN.md)。

## 当前默认：只导入物体

用户已要求不导入 3DGS 背景。默认入口是 `results/scene_gen_room/hybrid/objects_only.usda`，只加载扶手椅和边桌两个独立生成的 Mesh，保持它们在原场景中的位置。文件内没有高斯背景引用，也没有房间网格；脚凳等尚未生成 Mesh 的物体不会导入。

保留原有静态物体碰撞、一个不可见的解析地面支撑盒、灯光和相机。当前物体为静态物体，原生成网格的缺损仍然存在。

```bash
cd /path/to/SceneGen
LD_LIBRARY_PATH= conda run --no-capture-output -n scene_gen_isaac python scripts/open_scene_isaac.py
```

也可明确指定 `--scene results/scene_gen_room/hybrid/objects_only.usda`。如果旧窗口仍显示高斯背景，请在 File → Open 打开这个文件，或关闭旧窗口后重新运行命令。

已验证此 USD 的组成是 2 个 Mesh、0 个高斯体积、0 个点云，实际引用的资产只有 `assets/objects.usdc`。单独复制 `objects_only.usda` 和 `assets/objects.usdc` 即可加载，无需复制高斯文件。重新运行 `compose_hybrid_scene.py` 也会导出这个入口。

以下内容保留为可选带高斯背景的混合场景说明。

此前混合场景按物体进行替换：照片重建 3DGS → SAM 分割并关联高斯实例 → 选取物体多视图 → WorldSculpt 生成独立 Mesh → 移除原物体高斯 → 按原坐标放回生成 Mesh。**背景保持 3DGS，不做 TSDF、Marching Cubes 或房间网格导出。**

## 可选混合场景资产（非默认）

入口：`results/scene_gen_room/hybrid/scene.usda`，可直接用于当前 Isaac Sim 5.1。

已在本机 Isaac Sim 5.1 完成两个实拍相机的原生渲染验证：分别隐藏背景和物体后，画面变化符合预期。USD 中确认为 2 个物体 Mesh、1 个高斯体积、0 个背景 Mesh。背景保留 2,802,624 个高斯，移除两个被替换物体的 64,791 个高斯；保留参数逐项与原检查点一致。

| 场景部分 | 表示 | 来源 |
| --- | --- | --- |
| 背景与未替换物体 | NuRec 高斯体积 | 原始 3DGS 移除实例 0、1 后保留的高斯 |
| 扶手椅 0 | 独立 Mesh | 多视图 WorldSculpt，direct/train |
| 边桌 1 | 独立 Mesh | 多视图 WorldSculpt，direct/official |
| 脚凳及其余物体 | 原始高斯 | 尚未替换成生成 Mesh |
| 地面支撑 | 不可见 Cube 碰撞体 | 拟合地面位置的解析薄盒，不是房间重建网格 |

两个生成物体来自之前实际运行的多视图模型，不是从全房间 TSDF 中切出的扫描网格。选用现有直接生成结果，依据之前额外视角的轮廓拟合结果；免费 API 补图没有改善本次椅子结果，边桌补图视角不一致，故未采用。详情见 [对照实验](SCENE_GEN_API_COMPARISON.zh-CN.md)。

生成网格本身仍存在破损和局部起伏，这次修正的是场景组成，不代表生成质量已经解决。近邻高斯 DC 颜色用于物体顶点着色，不是新生成的贴图，也不代表真实材质。分割误差仍可能在替换边界留下少量残影。

## 启动

```bash
cd /path/to/SceneGen
LD_LIBRARY_PATH= conda run --no-capture-output -n scene_gen_isaac python scripts/open_scene_isaac.py --scene results/scene_gen_room/hybrid/scene.usda
```

打开后默认是记录相机 0，可在相机列表切换 `RecordedView_120`、`RecordedView_200`、`RecordedView_270`。背景在 `/World/BackgroundGS`，两个物体在 `/World/Objects`，分别隐藏即可检查替换结果。

物体目前是静态碰撞体；按 Play 不会让家具自由下落。高斯背景只用于显示，没有墙壁、脚凳或其余背景的碰撞。当前配置不能当作具有完整房间碰撞的机器人导航场景；以后按需要单独添加简化支撑/碰撞，背景仍无需网格化。

场景单位为米，Z 向上。沿用已有地面和尺度变换，椅高按 1 米估计，并非实测尺寸。

## 文件

- `scene.usda`：完整混合场景入口，引用相对路径 `assets/`；移动时保留整个目录。
- `assets/background_gs.usdz`：Isaac 原生 NuRec 高斯背景，不含房间 Mesh。
- `assets/background_gs.pt`：原始精度的背景高斯检查点。
- `assets/background_source_indices.npy`：背景对应原检查点的高斯索引。
- `assets/objects.usdc`：两个独立生成网格、顶点颜色、原位置及静态三角碰撞。
- `assets/armchair_placed.glb`、`assets/sidetable_placed.glb`：单物体网格，已放到场景位置，glTF Y 向上。
- `objects_placed.glb`：只有两个物体的 GLB；普通 GLB 不包含 NuRec 背景，应在 Isaac 中打开 USD 入口。
- `coordinates.json`：原 GS 世界坐标到米制 Z-up 的完整变换。
- `manifest.json`：来源、删除高斯数量、物体面数、坐标往返检查及能力边界。
- `isaac_hybrid_validation.json` 和 `isaac_hybrid_comparison.jpg`：原生渲染检查结果与对照图，由验证脚本实际运行生成。

NuRec 采用 NVIDIA 发布版转换器输出，保存半精度高斯数据。原 `.pt` 保留浮点 32 位参数；二者都是高斯表示，不是普通点云或网格。世界变换加在高斯体积外层，物体和相机使用相同变换。

## 复跑

在现有 `.cache/scene_gen/envs/worldsculpt` 环境上安装 `requirements-scene-gen-hybrid.txt`，将 NVIDIA `3dgrut` 仓库放在 `.cache/scene_gen/vendor/3dgrut`，固定到 `v1.1.0`（`0a5832248698ab8456b181d6ea17fe02eda58637`）。无需重新训练高斯或生成物体。

```bash
LD_LIBRARY_PATH= .cache/scene_gen/envs/worldsculpt/bin/python examples/mesh/compose_hybrid_scene.py
LD_LIBRARY_PATH= conda run --no-capture-output -n scene_gen_isaac python scripts/validate_hybrid_isaac.py
```

导出只读取旧 `simulation/coordinates.json` 的坐标标定，不读取或生成房间网格。改变实例清单时，仅移除确实已有生成 Mesh 的实例，避免从背景挖走未生成的物体。

NVIDIA 说明：[Isaac Sim 5.1 NuRec 场景支持](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/assets/usd_assets_nurec.html)，[官方转换器](https://github.com/nv-tlabs/3dgrut/tree/v1.1.0/threedgrut/export)。
