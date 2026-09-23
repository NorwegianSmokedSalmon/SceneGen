# 全场景逐物体生成实验

> 本页记录原工作区的 66 物体实验，产物不随源码分发。闭合检查通过，但存在主体断裂、缺面与碎片；完整性尚未达标，详见 [已知问题](KNOWN_LIMITATIONS.zh-CN.md)。

当前入口为 `results/scene_gen_room/full_objects/scene/scene.usda`。场景由独立物体网格组成；不包含 3DGS 背景、点云、房间 TSDF 网格或可见地板/墙面。范围是照片中可辨识、能够跨视图关联的物体；不是对每个像素都有几何保证的完整数字孪生。

```bash
cd /path/to/SceneGen
LD_LIBRARY_PATH= conda run --no-capture-output -n scene_gen_isaac python scripts/open_scene_isaac.py
```

默认打开全景相机。切回原始拍摄位置：

```bash
LD_LIBRARY_PATH= conda run --no-capture-output -n scene_gen_isaac python scripts/open_scene_isaac.py --camera /World/Cameras/RecordedView_0
```

输出位置：

- `scene/scene.glb`：完整物体组合，米制、Y 向上；保留独立物体节点。
- `scene/scene.usda`、`scene/assets/objects.usdc`：Isaac Sim 场景，米制、Z 向上。拷贝时需要整个 `scene/` 目录。
- `scene/assets/*.glb`：每个物体已放回场景坐标的单独文件。
- `scene/manifest.json`：物体清单、来源、面数、位置、几何质量限制和坐标变换。
- `scene/scene_preview.jpg`、`scene/isaac_*.png`：原生 Isaac 渲染预览。
- `scene/isaac_validation.json`：实际加载、渲染、静态碰撞场景启动和时间线检查结果。

## 本次验证结果

本次输出 **66 个独立闭合网格、7,542,858 个三角面**。完整 GLB 的当前大小与校验值见 `scene/delivery_validation.json`。导出后重新读取 GLB 的全部 66 个节点，逐个检查有限坐标与闭合拓扑；USD 在原生 Isaac Sim 中加载、四个视角渲染与静态碰撞时间线启动均已通过。USD 包含 66 个物体网格碰撞和 1 个不可见解析地面支撑，共 67 个碰撞体；点云与 Gaussian Volume 均为 0。

这些通过项表示文件和静态场景可用，不等于所有物体已恢复真实几何。电视柜、部分帘布、画框、远处小物件仍有较低拟合或推断表面；具体来源、候选分数、约束取舍和警告在清单中保留。

## 数据与流程

使用公开的 [Mip-NeRF 360 room](https://jonbarron.info/mipnerf360/) 数据，共 311 张带标定相机的照片。SAM3 对其中 52 张分布于整段拍摄轨迹的照片执行 29 类提示分割；跨视图跟踪生成 81 个候选，再按同一物体重叠、组合检测及假阳性审核得到 66 个实例。清单与合并记录在 `inventory.json` 和 `inventory_review.json`。书本、玻璃器皿及远处的小物体有些只占很少像素，其形状和尺度可靠性低于主要家具。

每个物体选择 3–6 个真实标定视图。选视图复用项目里的 H 热力图和排序函数，结合遮挡 O、可见覆盖 V，以及角度差和新增表面覆盖。这里 O/V 使用深度验证后的物体表面点代理估计，不能当作真实完整几何的可见率。输出在各物体的 `view_selection.json` 和 `selected_views.jpg`，没有把单张图复制成多个视角。

几何模型为本地 `AlayaLab/WorldSculpt`，1024 分辨率、多视图特征融合、仅几何生成。代码版本及输入指纹记录在每次生成的 `run.json`。部分失败候选使用官方采样器对照；不能仅凭生成器返回成功就接受网格。

沙发、椅子、边桌尝试了免费的 ModelScope 逐视图补图；每张公开来源图片保留 SHA256 证明。补图必须通过配准、尺寸、旋转与已观测区域覆盖检查。改变视角的输出被拒绝，已观测的非透明像素保持原值。API 调用与接受数量见 `objects/object_*/api/` 和 `completed/completion.json`，不把未通过检查的 API 结果算作有效输入。

对漏掉大块结构或输出空几何的模型，使用多视图轮廓与已观测物体表面的深度约束补充稀疏形状，再由同一神经形状解码器生成网格。深度来自已有 3DGS 的表面代理，只用于单个物体的空间约束；没有将背景 3DGS 或背景网格导入最终场景。每个候选的 `shape_guide.json` 记录实际约束。可见外轮廓不能确定背面的凹陷；约束后的隐藏表面仍是推断。

对于电视、画框、书本等近似实体的物体，可用类别限定的轮廓补全。边桌单独只补木桌面，保留金属框架空隙。钢琴、带腿家具、花盆等不能简单把整个二维外包轮廓填成实体。

## 修复与验证范围

候选会重投影到输入照片，检查可见轮廓覆盖和重合率，输出 `inspection.json/jpg`。这些指标衡量输入视图拟合，不是独立测试集几何精度，也不能证明背面正确。指标较弱的实例会在清单中记录警告。

封洞先尝试 MeshFix，并检查修复是否丢掉椅腿等结构；失败时采用窄体素表面封闭。修复前后检查网格边界、非流形边、有限坐标、位置范围和表面保留。后续修复使用真实点到三角面距离；早期通过的保守检查使用最近顶点距离，数值记录在每个 `final/mesh_world.json` 中。网格闭合只表示拓扑检查通过，不代表物体真实表面完全恢复。

实际场景预览后还会检查生成物体是否越出自身空间范围或伸入估计地面以下。`constrain_object_extents.py` 对明确越界的几何做封闭裁切，并重投影验证；只有输入视图轮廓覆盖下降不超过 4 个百分点时才接受。原网格和每次裁切结果分别保留，详见 `layout_constraint_report.json`。这一步使用估计的物体范围和地面高度，不能替代实测尺寸；地面约束在必要时保留 5 cm 的估计误差余量。大地毯单独使用推断的 1 cm 厚度抑制纹理造成的几何凸起，同样须通过原视图覆盖检查。电视柜优先采用空间范围正确的候选，允许相对最优轮廓拟合不超过 4 个百分点的取舍。

跨视图范围筛选也需要区分“被遮挡”和“物体不存在”。`refine_occluded_object_bounds.py` 已用于恢复被其他视图遮挡的柜体顶部/腿部；`refine_clipped_book_pose.py` 修正了一册书因为两张照片截断而产生的定位错误。

颜色来自原始多视图可见表面，并向未知面插值；没有运行独立的高质量纹理生成模型。尺度沿用已有的椅子高度估计，未经实物测量。放回位置使用统一坐标变换，并检查变换往返误差。

USD 中物体具有独立的静态三角面碰撞。额外存在一个不可见的解析地面支撑，它不是重建房间网格。当前导出不提供逐物体质量、关节或动态稳定性保证，适合先检查场景布局与静态碰撞；抓取、堆叠等任务仍需要对应物体的物理参数和接触验证。



### 发暗问题与显示修正

2026-09-22 检查确认：USD 的顶点颜色存在且材质绑定正常；照片中的暗部已被投影到顶点颜色，原有环境光预设又使这些区域偏暗。原生 Isaac 同视角对照了原环境光、两档增强环境光、颜色发光直显和中性灰材质。最终环境光强度从 1200 调为 4800，并固定曝光与色调映射参数。该设置只用于清楚查看物体，不能解释为估计出的真实房间灯光。

GLB 原先只有顶点颜色，没有显式材质。根据 [glTF 规范](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#materials)，默认材质的金属度为 1，容易在缺乏环境照明的查看器中发黑。现在所有物体显式使用非金属材质（metallic=0、roughness=0.85），导出顶点法线，并将照片的 sRGB 顶点颜色转换为规范要求的线性颜色，使用 16 位归一化存储保留暗部层次。USD 原有 sRGB 到线性转换是正确的，未重复转换。

这次不改变任何物体的位置或三角面；通过导出前后的逐物体顶点位置与索引哈希比对确认。对照图保存在 `scene/appearance_diagnosis/`。这些修正不会去除照片里已经拍进去的阴影，也不会补出准确的背面纹理；获得可重新打光的真实材质，仍需单独的纹理/反照率重建。

## 复现与继续处理

几何环境：`.cache/scene_gen/envs/worldsculpt/bin/python`；分割环境：`conda run -n scene_gen python`；Isaac 环境：`conda run -n scene_gen_isaac python`。

主要脚本依次为：

1. `examples/segmentation/full_scene_inventory.py`：批量多类别分割。
2. `track_full_scene_objects.py`、`refine_full_scene_inventory.py`：关联、审核、深度筛选、选视图及裁剪相机。
3. `prepare_solid_object_inputs.py`、`scripts/complete_full_object_views.py`：有记录的遮挡区域补全。
4. `generate_worldsculpt_objects.py`：每个物体独立多视图生成。`--silhouette-guide` 加入已校准的单物体轮廓/深度约束；`--discard-raw` 只清理本次成功导出后的中间张量文件，保留 GLB 与复现记录。
5. `inspect_full_scene_objects.py`：真实输入视图重投影检查。
6. `examples/mesh/select_and_repair_full_objects.py`：选择、封闭及形状保留检查；缺失实例时拒绝完整导出。
7. `examples/mesh/constrain_object_extents.py`：经实际视图检查的空间范围、地面和地毯约束。
8. `examples/mesh/assemble_full_object_scene.py`：逐物体位置恢复、着色、GLB/USD 导出和完整数量检查。
9. `scripts/validate_full_object_scene_isaac.py`：原生 Isaac 检查与预览。

所有具体候选选择在 `selected_meshes.json`。旧的两个物体示例 `hybrid/objects_only.usda` 和旧的全房间 `simulation/scene.usda` 保留为历史产物，不再作为默认启动入口。
