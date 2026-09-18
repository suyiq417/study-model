"""独立学习示例：包含完整类实现和运行验证。"""

import torch
from torch import nn


class FeedForward(nn.Module):
    """对每个 token 独立应用相同的两层前馈网络。"""

    def __init__(self, model_dimension, hidden_dimension):
        super().__init__()

        self.linear_1 = nn.Linear(model_dimension, hidden_dimension)
        self.activation = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_dimension, model_dimension)

    def forward(self, x):
        # [B, T, D] -> [B, T, Dff]
        hidden = self.linear_1(x)

        # ReLU 保留正数，将负数变成 0，引入非线性。
        hidden = self.activation(hidden)

        # [B, T, Dff] -> [B, T, D]
        return self.linear_2(hidden)


def main():
    torch.manual_seed(42)

    batch_size = 2
    sequence_length = 3
    model_dimension = 4
    hidden_dimension = 16

    x = torch.rand(batch_size, sequence_length, model_dimension)

    feed_forward = FeedForward(model_dimension, hidden_dimension)

    # 观察前馈网络内部的形状变化。
    hidden = feed_forward.activation(feed_forward.linear_1(x))
    ffn_output = feed_forward(x)

    print("input shape:", x.shape)
    print("hidden shape:", hidden.shape)
    print("ffn output shape:", ffn_output.shape)

    # 每个 token 独立使用同一组权重，不会与其他 token 混合。
    single_token_output = feed_forward(x[0, 0])
    print(
        "\nsingle token result matches:",
        torch.allclose(single_token_output, ffn_output[0, 0]),
    )

    # Dropout 在训练时随机将部分元素置零，并缩放保留的元素。
    dropout = nn.Dropout(p=0.2)

    # 先使用评估模式，使本节的数值验证不受随机丢弃影响。
    dropout.eval()

    # 残差连接：在输入基础上加上前馈网络产生的更新。
    residual = x + dropout(ffn_output)

    # 对每个 token 的最后 D 个特征分别做归一化。
    normalization = nn.LayerNorm(model_dimension)
    output = normalization(residual)

    print("\nresidual shape:", residual.shape)
    print("normalized output shape:", output.shape)

    # LayerNorm 使用最后一维的均值与总体方差。
    # keepdim=True 保留末尾的单维，便于向 [B, T, D] 广播。
    mean = residual.mean(dim=-1, keepdim=True)
    variance = residual.var(dim=-1, keepdim=True, unbiased=False)

    manual_output = (residual - mean) / torch.sqrt(
        variance + normalization.eps
    )

    # LayerNorm 还包含可学习的缩放参数和偏移参数。
    manual_output = (
        manual_output * normalization.weight + normalization.bias
    )

    print(
        "manual LayerNorm matches:",
        torch.allclose(manual_output, output, atol=1e-6),
    )

    print("\noutput mean per token:")
    print(output.mean(dim=-1))

    print("output variance per token:")
    print(output.var(dim=-1, unbiased=False))

    # 使用全 1 输入，更容易观察 Dropout 的训练与评估行为。
    probe = torch.ones(2, 8)

    dropout.train()
    print("\ndropout in training mode:")
    print(dropout(probe))

    # eval() 关闭 Dropout 的随机丢弃，但不会关闭自动求导。
    dropout.eval()
    print("\ndropout in evaluation mode:")
    print(dropout(probe))


# 直接运行时执行示例；被其他模块导入时不执行。
if __name__ == "__main__":
    main()
