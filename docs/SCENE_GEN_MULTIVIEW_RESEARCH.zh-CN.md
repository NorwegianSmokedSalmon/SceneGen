# 多视图补全与三维生成选型

> 2026-09 原工作区的历史实验记录。数据、权重和生成结果不随仓库分发；当前入口见 [README](../README.md)，安装见 [环境说明](SCENE_GEN_SETUP.zh-CN.md)。

核查日期：2026-09-18。以下是官方文档、作者仓库与论文调研；尚未调用图像 API，也未在本机运行这些新增模型。论文指标不能直接视为当前 room 桌椅的实测结果。

## 免费图像编辑 API

| 候选 | 已确认信息 | 接入判断 |
| --- | --- | --- |
| ModelScope / Qwen-Image-Edit-2511 | 魔搭官方在发布公告中提供免费 API-Inference 体验；编辑 API 支持参考图片 | 首个试验候选，但当前账户资格、模型在线状态及剩余额度尚未核验；不能把平台总请求额度当成该模型的可用生图张数 |
| Cloudflare Workers AI / FLUX.2 klein 4B | 官方提供每日 10,000 Neurons 免费额度；模型支持最多 4 张参考图片 | 免费额度规则明确，可作为备用；用量按输入、输出图像计费单位累计，额度不是 10,000 张图片 |
| Gemini / Nano Banana Pro | 官方 API 价格表的 Free Tier 为 Not available | 不属于免费 API 方案；网页端体验与 API 配额是两件事 |
| Hugging Face Inference Providers | 免费账户每月 $0.10，官方注明可调整 | 适合极小试验，不适合作为批量补全的主要额度 |

来源：[魔搭 2511 官方公告](https://modelscope.csdn.net/694ccefe836da32144879134.html)、[魔搭 API 编辑示例及账户要求](https://modelscope.csdn.net/691c36ee82fbe0098caca391.html)、[Cloudflare 免费额度](https://developers.cloudflare.com/workers-ai/platform/pricing/)、[FLUX.2 klein 4B 参考图能力](https://developers.cloudflare.com/changelog/post/2026-01-15-flux-2-klein-4b-workers-ai/)、[Gemini 价格](https://ai.google.dev/gemini-api/docs/pricing)、[HF 额度](https://huggingface.co/docs/inference-providers/pricing)。

魔搭官方接入说明列出实名认证、绑定阿里云账号及 Access Token。当前限额文档是动态页面，本次没有读到可核实的模型级限额，因此不承诺固定免费张数。

## 已有可用代码的三维候选

| 工作 | 输入与输出 | 对本项目的意义 |
| --- | --- | --- |
| [ReconViaGen v0.5](https://github.com/GAP-LAB-CUHK-SZ/ReconViaGen/tree/v0.5) | 多张 RGBA；ReconViaGen 结构估计 + TRELLIS.2 多视图融合，输出 Mesh/PBR | 用户原有基准的直接升级；现成 app_v05 入口移除了高斯 decoder，接回纯 3DGS 场景需要额外转换与拟合 |
| [Mix3R V2](https://github.com/jsnln/mix3r) | 多视图联合几何生成与相机估计；Pi3 + TRELLIS | 作者推荐 V2 用于真实照片和更灵活相机；权重发布在 [ModelScope](https://modelscope.cn/models/jsnln00/mix3r/)，值得与 ReconViaGen 比较 |
| [Pixal3D 多视图版](https://github.com/TencentARC/Pixal3D) | 多视图及相机 transforms.json，独立多视图权重；Mesh/PBR | 2026-09 发布多视图入口；已有 COLMAP 相机可利用。裁切、缩放必须同步改内参，不能套用示例的四个固定环绕相机 |
| [WorldSculpt](https://github.com/AlayaLab/WorldSculpt) | 被遮挡的多视图、实例掩码、相机与三维包围盒；组合网格场景 | 2026-09 公开代码和权重；基于 Pixal3D 适配遮挡输入，且展示从 3DGS 世界生成独立物体网格。任务形式最贴近我们的室内物体补全与组装，但最终是网格 |
| [RecGen](https://github.com/TRI-ML/recgen) | 多视图 RGB-D、掩码与内参；Mesh、Gaussian、6-DoF 姿态 | 适合保留高斯输出及场景放置；需要可靠深度。原生双视图，更多视图通过去噪阶段逐对融合；不是下面的 RecGen3D |
| [MV-SAM3D](https://github.com/devinli123/MV-SAM3D) | SAM 3D 多视图扩展；高斯/网格与布局优化 | 可复用现有 SAM 3D 基础。论文采用注意力熵与可见性加权；不是逐图单独生成再拼合。仓库 README 的 weighted 入口与根目录文件名有差异，实际接入要先核对 |

以上是接入优先级候选，不代表统一榜单排名；当前没有在相同 room 视图、同样补全输入下的对照结果。

## 更近期论文与其他方向

- [RecGen3D，原 UniRecGen](https://github.com/zsh523/RecGen3D)：SIGGRAPH Asia 2026；作者的 GSO/Toys4K 几何评测优于所比较的 ReconViaGen，但 README 表示代码和权重要到 2026-11 上旬发布。不能当成已经能部署的选项，也不能据此断言优于 ReconViaGen v0.5。
- [ReconPlusGen](https://arxiv.org/abs/2609.11129)：2026-09-10 的新论文，利用重建点云进行噪声反演和置信度调制；本次未找到已发布的官方代码/权重入口，先跟踪。
- [Free-Range Gaussians](https://free-range-gaussians.github.io/)：ECCV 2026，直接生成非网格约束的高斯；作者项目页未提供代码入口，先跟踪。
- [GenRecon](https://github.com/kasothaphie/GenRecon)：2026-06 已公开代码/权重，从多视图重建室内场景的 PBR 网格；属于整场景生成方向，接入范围大于替换几个家具。
- [DecomVoxel](https://zr-zhou0o0.github.io/DecomVoxel-Webpage/)：项目展示原场景坐标中的物体补全，但本次未能读取其代码仓库，不将其列为已确认可部署选项。

## 已确认的输出要求

用户确认：Mesh/GLB 也可以，优先物体几何质量。因此以 WorldSculpt / Pixal3D 作为主要评估候选，ReconViaGen v0.5 作为对照；不以原生高斯输出为选型约束。该选择基于任务适配性，尚不是本机质量评测结论。

## 本机条件与建议试验

已核对 RTX 5090（32GB 级显存），当前项目所在分区剩余约 31GB。上述新模型都尚未做 5090 兼容验证。
ReconViaGen 官方的 16 视图 <18GB / 带 refinement <24GB 数据对应 v0.2，不能套到 v0.5。
v0.5 和 Pixal3D 提供低显存路径，实际峰值还需测试；旧版安装脚本中的 PyTorch/CUDA 组合也不能直接照搬到 5090。
RecGen 论文报告约 14.1GB 总显存，但该测试排除了模型加载和后处理，不能解释成任意多视图完整流程的显存上限。[RecGen 论文附录 C](https://arxiv.org/html/2604.27106v1)

建议先恢复原有 H/O/V 评分和视角覆盖约束，针对扶手椅、茶几、脚凳各导出一组真实多视图：

1. 保留真实图像与可见掩码、相机、深度；完成图作为单独派生输入。
2. 同一组视图比较「直接多视图生成」和「API 联图补全后生成」，避免将图像补全默认视为必然改善。
3. 评价未用于物体生成的真实视角轮廓、可见区域深度与颜色一致性，并检查桌腿/椅腿数量、接地、穿插和场景尺度。

若允许 Mesh/GLB，优先评估 WorldSculpt / Pixal3D，并以 ReconViaGen v0.5 作对照；若必须维持纯 3DGS 输出，优先评估 RecGen / MV-SAM3D，再考虑 Mesh 到高斯的额外拟合成本。
API 应尽量只补遮挡区域，维持已有视角、投影及可见结构；编辑后仍须验证三维一致性。补全出的新像素不能直接配上原遮挡物或背景的深度，尤其在 RGB-D 模型输入中需要重新处理。
