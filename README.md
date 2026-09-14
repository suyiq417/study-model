# Study Model

使用 Python 和 PyTorch，从基础开始学习并实现常见的深度学习模型。

## 环境

本项目使用 [Pixi](https://pixi.sh/) 管理 Python 环境和项目依赖。

安装项目环境：

```bash
pixi install
```

运行学习代码：

```bash
pixi run python transformer/08_qkv_projections.py
```

## Transformer 学习记录

目前已完成：

- PyTorch 张量的创建、形状和数据类型
- 张量的索引与切片
- 逐元素运算和矩阵乘法
- 张量转置与形状变换
- 三维张量和批量矩阵乘法
- Embedding 权重表、token 查表和输出形状
- 固定随机种子以复现实验结果
- 正弦和余弦位置编码
- 缩放点积注意力
- 三维张量的批量注意力计算
- Causal Mask 因果掩码
- 使用线性层生成 Query、Key 和 Value
