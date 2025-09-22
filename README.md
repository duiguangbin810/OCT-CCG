# OCT-CCG: Orthogonal Category Token for Controllable Counting Generation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.3.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 项目简介

OCT-CCG (Orthogonal Category Token for Controllable Counting Generation) 是一个基于Stable Diffusion的可控计数生成框架。通过正交类别token（OCT）技术和对偶优化方法，实现对生成图像中目标数量的精确控制。

该框架的主要特点：
- **精确计数控制**：通过正交类别token实现对生成图像中目标数量的精确控制
- **可微分计数**：提供基于注意力机制的软计数方法
- **对偶优化引导**：采用原始-对偶优化算法引导生成过程
- **灵活配置**：支持多种类别和数量组合

## 技术栈

- Python 3.8+
- PyTorch 2.3.0+
- Diffusers
- Transformers
- Accelerate

## 安装指南

### 环境要求

- Python 3.8 或更高版本
- CUDA 11.8+ (如果使用GPU加速)
- 8GB+ RAM (推荐16GB+)

### 安装步骤

1. 克隆项目仓库：
```bash
git clone <repository_url>
cd OCT-CCG
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 项目结构