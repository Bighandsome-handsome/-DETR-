# -DETR-
[![Project Status](https://img.shields.io/badge/Project-National_Innovation_Program-blue)](https://github.com/your-username/Sentinel-ShipDet)
[![Framework](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Backend](https://img.shields.io/badge/Flask-lightgrey?logo=flask)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **国家级本科生创新创业训练计划项目成果** > **项目名称**：基于 Sentinel-1 遥感卫星的海面目标检测与识别  
> **核心架构**：Deformable DETR (Transformer) + YOLOv8 (CNN) + Flask Web Stack
### 项目进度 (Project Milestones)
#### 第一阶段：模型验证与基准测试 (已于 2025年12月15日 完成)
- [1] **数据集训练**：选取 HRSID 官方训练集，在 DETR 与 Deformable DETR 上完成深度训练。
- [2] **自主测试集构建**：搭建 Test01 数据集，验证模型在实际场景下的泛化能力。
- [3] **跨架构性能评估**：完成 Transformer 架构 (DETR系列) 与 CNN 架构 (YOLOv8) 的全方位对比测试。
- [4] **成果汇总**：撰写科研论文，系统总结研究成果。
#### 第二阶段：结构优化与数据集重构 (2025年12月20日 启动 - 进行中)
- [1] **模型深度优化**：以 Deformable DETR 为核心，探索更先进的特征提取与注意力机制优化。
- [2] **数据集扩充**：重构数据集，引入更多 **极化方式** (如 VV, VH) 的 SAR 图片，提升模型对复杂电磁特性的理解。

## 1. 关于数据集Test01
### 1.1 兴趣区域 ROI 的选取
   我们选取了欧洲航天局ESA提供的Sentinel-1卫星影像作为主要数据源。通过访问 Copernicus Open Access Hub，结合研究区域的地理范围与时间窗口，下载了覆盖典型海岸线与港口区域的Sentinel-1 SLC（Single Look Complex）产品。SLC数据保留了完整的幅度与相位信息，适用于后续的轨道校正、地形处理与目标增强分析。
   我们先后选取了多处具有代表性的典型水域区域，分别围绕杭州湾近岸区域、洋山港深水港区，构建多样化的遥感目标检测数据集。杭州湾和洋山港位于我国长三角地区，经济发达，航海贸易繁荣，来往船只众多，可以为我们提供充足的研究样本。杭州湾近岸区域受潮汐影响显著，水体浅、泥沙含量高，背景散射特征复杂多变。区域内船舶尺度差异较大，既包含小型渔船，也存在中型运输船，适合评估模型在复杂背景与尺度变化条件下的检测能力。洋山港深水港区作为我国重要的集装箱枢纽港，该区域船舶密集、靠泊频繁，且存在大量港口设施与人工结构物，背景干扰强烈。该区域为高密度目标检测提供了理想测试场景，有助于检验模型在遮挡、重叠等复杂条件下的鲁棒性。图1所示的区域（中心点30.4895°N，122.1451°E）为本次数据集重点研究的范围。
   
<img width="702" height="303" alt="image" src="https://github.com/user-attachments/assets/4952096c-e115-4f81-86f5-605c2d8b8b80" />

图1.1 数据集核心研究区域示意图

   为提升数据的代表性与多样性，我们先后选取了不同成像时间段的Sentinel-1 SLC影像，涵盖多种海况与船舶分布状态。同时，我们选用IW模式下的VV极化数据，以增强对不同目标散射特性的刻画能力。每个区域均选取多个时间点的影像样本，确保数据在时间、空间与目标类型上的广泛覆盖，从而提升模型在多样遥感场景中的泛化能力与实用价值。
### 1.2 数据处理过程
为确保SAR影像具备良好的几何一致性与辐射一致性，我们采用SNAP软件对原始 Sentinel-1 SLC数据进行了系统化预处理并最终得到我们的数据集，整个流程如图1.2所示。

<img width="742" height="283" alt="image" src="https://github.com/user-attachments/assets/d34cc563-bf09-481c-a9f9-c2201fab8c90" />

图1.2 数据处理核心流程图

   在轨道文件校正Apply Orbit File中，我们引入ESA官方发布的精化轨道文件SPK对原始SLC数据进行轨道校正。这个步骤可显著提升影像的地理定位精度，减少轨道误差对后续地形校正与配准精度的影响，是整个预处理流程的基础环节。在热噪声去除Thermal Noise Removal中，我们利用 SNAP 工具链对图像进行热噪声剔除，去除系统硬件带来的背景干扰，尤其在海面等低散射区域中效果显著，有助于提升图像的整体对比度与目标可分性。在去突处理Deburst中，由于Sentinel-1 IW模式下的SLC数据以burst（子脉冲）形式存储，图像在方位向存在间断。通过Deburst操作可将多个burst拼接为连续图像，恢复完整的空间结构，便于后续的几何与辐射处理。随后我们辐射校正，将影像像素值转换为后向散射系数σ⁰，使其具备物理意义，便于不同时间、区域或极化通道之间的定量比较。至此，我们可以得到如图1.3所示的影像，在本影像中，我们可以大致区分出港口或陆地、岛屿和海域，但是我们还无法准确分辨船只。

<img width="724" height="303" alt="image" src="https://github.com/user-attachments/assets/f6bd945d-a082-4e2c-bf3a-1eb5bd790f63" />

图1.3 进行初步处理的影像数据

   随后，我们在预处理完成的影像数据中选取了若干兴趣区域（Region of Interest, ROI）进行进一步处理与分析。兴趣区域的选择不仅有助于聚焦于目标密集或变化显著的关键区域，还能有效降低图像的冗余信息与计算负担，提升后续标注与模型训练的效率与精度。具体而言，兴趣区域的划定主要依据船舶分布密度、岸线结构复杂度以及背景干扰程度等因素，确保所选区域具备代表性与挑战性，能够充分反映模型在多样遥感场景下的适应能力。
为进一步提升影像的几何精度与空间一致性，我们对影像进行了地形校正Terrain Correction。该步骤基于SRTM 3Sec分辨率的数字高程模型DEM，结合精化轨道信息，对影像进行几何重投影与正射校正。通过消除地形起伏对成像几何的影响，地形校正不仅提升了图像的空间定位精度，也确保了多时相、多区域影像之间的几何一致性，为后续的目标检测与时序分析提供了可靠的空间基础。为提升图像的可视化效果与标注效率，对处理后的影像进行了线性拉伸。通过将像素值线性映射至0~255的灰度范围，显著增强了图像的亮度对比与细节表现，使目标轮廓更加清晰，便于人工标注与模型输入。最后，利用SNAP中的波段表达式计算Band Math工具对图像进行增强处理，具体操作包括对VV极化通道进行对数变换等操作。
### 1.3 数据标注过程
   为构建高质量的监督样本，我们采用开源标注工具Labelme对预处理后的Sentinel-1图像进行目标标注。图1.4为我们标注的数据之一。Labelme支持多边形、矩形等多种标注形式，具备轻量、可视化强、格式标准化等优点，广泛应用于遥感图像与计算机视觉任务中。在具体标注过程中，考虑到SAR 图像本身存在speckle噪声强、目标边缘模糊等问题，直接在雷达图像上进行精确标注存在一定困难。
为提升标注的准确性与一致性，我们引入Google Earth提供的高分辨率光学影像 作为辅助参考。我们在Labelme中加载处理后的SAR图像，并通过比对与其成像日期尽可能接近的光学影像，辅助识别舰船目标的位置与轮廓。需要指出的是，我们尽力保证Sentinel-1与Google Earth影像属于同一天获取，但由于成像时间存在差异，部分船舶可能已发生移动或状态变化。因此，在标注过程中，我们优先选择静态目标（如停泊船舶、或正在靠港的船只）作为标注对象，并结合多时相SAR图像进行交叉验证，确保标注结果的时空一致性与几何准确性。最终，所有标注均采用Labelme的JSON格式保存，包含目标类别、边界坐标与图像元信息，为后续的数据集构建与模型训练提供了标准化的输入。

<img width="718" height="415" alt="image" src="https://github.com/user-attachments/assets/3062f64a-f283-4ee6-84e8-d7cf32cf879a" />

图1.4 使用labelme完成数据标注

至此，我们已完成数据集的构建工作，数据集可以在本项目的`Test01`中进行下载和查看。

## 2. 关于模型`Deformable DETR`
本项目是国家级大创项目《基于 Sentinel-1 遥感卫星的海面目标检测与识别》的核心子项目。我们专注于利用 **Deformable DETR** (可变形 Transformer) 解决 SAR 影像中舰船目标检测的挑战。
## 🚀 为什么选择 Deformable DETR?
传统的 DETR 虽然实现了端到端的检测，但在处理 SAR 影像时存在两个致命伤：**收敛极慢**和**小目标检测弱**。Deformable DETR 通过以下创新完美解决了这些问题：
- **可变形注意力机制 (Deformable Attention)**：不再像传统 Transformer 那样关注全局所有像素，而是只关注参考点周围的一小部分关键采样点。这使得计算量大幅下降，训练收敛速度提升了 **10倍以上**。
- **多尺度特征提取 (Multi-Scale Feature Maps)**：利用了类似 FPN 的多尺度特征，这对于 SAR 影像中那些只有几个像素点的远洋小船至关重要。
- **迭代边界框精修 (Iterative Bounding Box Refinement)**：通过多层 Decoder 的循环优化，让舰船的边缘定位更加精准。
---
## 项目里程碑 (Project Milestones)
### 第一阶段：基准测试与验证 (已完成 - 2025.12.15)
- [1] **模型对标训练**：在 HRSID 数据集上完成了标准 DETR 与 Deformable DETR 的对比训练。
- [2] **真机场景测试**：构建了自主测试集 `Test01`，模拟真实海域抓拍，验证模型泛化力。
- [3] **跨架构性能评估**：对比了基于 Transformer 的模型与基于 CNN (YOLOv8) 的性能差异。
- [4] **学术成果产出**：汇总实验数据，完成了项目中期研究论文。

### 第二阶段：结构优化与极化增强 (进行中 - 2025.12.20 启动)
- [1] **模型架构魔改**：针对舰船目标的长宽比特性，优化 Deformable Attention 的采样策略。
- [2] **多极化特征融合**：重构数据集，探索 **Dual-pol (VV/VH)** 数据对提高金属目标检测率的贡献。
- [3] **端到端部署**：优化推理流程，进一步降低在嵌入式端处理高分辨率 SAR 切片的延迟。
---
## 📊 性能数据预览
| 模型 (Model) | 训练轮次 (Epochs) | mAP@0.5 | 优势说明 |
| :--- | :--- | :--- | :--- |
| **Standard DETR** | 500 | 59.85% | 基础基准，收敛极慢 |
| **Deformable DETR** | **50** | **79.81%** | **高精度、快速收敛、适合多尺度** |
| **YOLOv8-s** | 100 | 约86.0% | 实时性极佳，但精度略逊 |
---
<img width="721" height="336" alt="image" src="https://github.com/user-attachments/assets/9b8f1ec6-922c-4a39-8fad-6fc8511d7ce7" />  <br>
Figure COCO Detection Metric Feedback Statistics for Deformable DETR Model Rounds 50, 60, and 100

## 🛠️ 环境与运行 (Setup)
### 编译定制算子
Deformable DETR 依赖于高效的 CUDA 算子，需手动编译：
```bash
cd ./models/ops
sh make.sh # 请确保已安装 CUDA Toolkit
```
### 快速训练
```bash
python main.py \
    --model_name deformable_detr \
    --dataset_file ship_srtp \
    --data_path ./data/HRSID \
    --epochs 100
```
## 预训练模型 
训练好的模型权重文件已上传至百度网盘： 
  - 链接：https://pan.baidu.com/s/1wuqODsIBTGRn9Zy0nMwR7w
  - 提取码：fbkm
  - 文件说明：`checkpoint0099.pth` 为最终训练模型（约488MB）
## 致谢 
感谢原始Deformable DETR团队的开源贡献。
