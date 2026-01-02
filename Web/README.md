# 🚢 基于 Deformable DETR 的海面目标检测与解译平台

[![Project Status](https://img.shields.io/badge/Project-National_Innovation_Program-blue)](https://github.com/your-username/Sentinel-ShipDet)
[![Framework](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Backend](https://img.shields.io/badge/Flask-lightgrey?logo=flask)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **国家级本科生创新创业训练计划项目成果** > **项目名称**：基于 Sentinel-1 遥感卫星的海面目标检测与识别  
> **核心架构**：Deformable DETR (Transformer) + YOLOv8 (CNN) + Flask Web Stack

---

## 🌟 项目综述
本项目针对 **Sentinel-1 SAR（合成孔径雷达）** 影像在复杂海况下的舰船检测难题，通过引入 **Deformable DETR** 架构，攻克了传统模型在 SAR 图像中收敛慢、小目标识别率低的技术瓶颈。同时，我们开发了一套深海赛博风格的 Web 交互系统，实现了深度学习模型从实验环境向用户端应用的完整转化。

## 🚀 核心算法：为什么是 Deformable DETR？
针对 SAR 影像特有的斑点噪声（Speckle Noise）与多尺度目标，我们重点优化了 Deformable DETR：
- **可变形注意力机制 (Deformable Attention)**：仅采样参考点附近的特征，计算复杂度大幅降低，训练收敛速度较标准 DETR 提升 **10倍** 以上。
- **多尺度特征融合 (Multi-scale Features)**：引入 FPN 思想，显著增强了对远海极小尺度舰船的捕捉能力。
- **高精度定位**：通过迭代边界框精修（Iterative Bounding Box Refinement），在回归精度上超越了主流 CNN 模型。

## 💻 Web 交互系统：沉浸式解译体验
我们将复杂的推理脚本封装为一套 **B/S 架构** 的在线服务，其核心设计包括：
- **深海赛博美学**：基于 Tailwind CSS 打造的深蓝紫渐变视觉、毛玻璃拟态面板与动态流体背景。
- **“两栏三列”布局**：
  - **左侧**：集成 Marked.js 与 KaTeX，实现项目文档与数学公式的实时 Markdown 渲染。
  - **右侧**：支持图像拖拽上传、`FileReader` 实时预览，并提供 YOLO 与 DETR 模型的无感切换。
- **异步调度架构**：后端采用 Flask + `subprocess` 松耦合策略，通过 Fetch API 实现无刷新检测，确保了推理过程的稳定与高效。

## 🛠️ 快速上手
1. **环境准备**：
   ```bash
   pip install -r requirements.txt
   cd ./models/ops && sh make.sh # 编译 CUDA 算子
   ```
2. **快速上手**：
   ```python
   # 请先确认你已经下载了我们训练好的`Deformable DETR`模型权重
   # 本网站还支持`YOLOV8`模型，我们后续会公开相应的模型权重文件
   python server.py
   # 访问地址：http://localhost:5000
   ```
