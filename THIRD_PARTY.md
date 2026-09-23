# 来源与第三方依赖

| 代码 / 模型 | 来源 | 本仓库中的形式 |
| --- | --- | --- |
| Extended-GS 场景流程 | [AshadowZ/Extended-GS](https://github.com/AshadowZ/Extended-GS) | 从 `scene_gen` 工作区提取，含未提交的场景生成扩展；来源见 `docs/source_provenance.json` |
| gsplat | [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) | 保留实际使用的源码、Apache-2.0 LICENSE、CITATION.bib 与文件版权声明 |
| GLM | [g-truc/glm](https://github.com/g-truc/glm) | 固定提交的 Git submodule；许可证在上游子模块内 |
| CropFormer / Mask2Former | [qqlu/Entity](https://github.com/qqlu/Entity/tree/6e7e13ac91ef508088e1b848167c01f19b00b512/Entityv2/CropFormer) | 兼容旧分割流程的源代码；保留 subtree 的 MIT LICENSE 与 Entity 根目录的 CC BY-NC 4.0 文本；不能将顶层 Apache LICENSE 视为覆盖全部第三方材料 |
| SAM3、SAM 3D Objects、HLOC、Detectron2、PyTorch3D | 安装脚本中列出的官方仓库 | 按提交下载至 `.cache/scene_gen/vendor/`，不随 Git 分发 |
| WorldSculpt、FlexGEMM、CuMesh、TRELLIS.2、nvdiffrast | `scripts/setup_scene_gen_worldsculpt.sh` 中的官方仓库 | 按提交下载，几何模型权重另行下载 |
| Mip-NeRF 360 数据 | [作者项目页](https://jonbarron.info/mipnerf360/) | 下载脚本，不附带照片或重建产物 |
| Isaac Sim | NVIDIA Isaac Sim | 独立安装；本仓库只提供导入、渲染和验证脚本 |

下载得到的第三方代码、权重、数据遵循各自上游的许可证和使用条款。
