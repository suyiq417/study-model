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
pixi run python transformer/12_encoder_block.py
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
- 使用 nn.Module 封装单头自注意力
- 多头自注意力的拆头、合头与输出投影
- 前馈网络、残差连接、LayerNorm 和 Dropout
- Post-LN Encoder Block 的组合与堆叠
- Python 脚本入口保护

## 代码组织

`01`–`12` 按学习顺序编号，每个文件都可以独立运行。
需要用到的模型组件直接保留在对应文件中，并配有中文注释，方便从头阅读完整计算过程。
例如，`12_encoder_block.py` 同时包含多头自注意力、前馈网络和编码器块的实现。

`09`–`12` 的演示代码放在 `main()` 中，并用 `if __name__ == "__main__":` 作为入口，
保证只有直接运行文件时才执行演示。
