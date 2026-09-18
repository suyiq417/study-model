"""独立学习示例：包含完整类实现和运行验证。"""

import math

import torch
from torch import nn


class SelfAttention(nn.Module):
    """将 QKV 投影与缩放点积注意力封装为单头自注意力层。"""

    def __init__(self, model_dimension):
        super().__init__()

        # 三个独立的线性层，都接收同一个输入。
        # 本节令 Q、K、V 的特征维度都等于 model_dimension。
        self.query_projection = nn.Linear(
            model_dimension, model_dimension, bias=False
        )
        self.key_projection = nn.Linear(
            model_dimension, model_dimension, bias=False
        )
        self.value_projection = nn.Linear(
            model_dimension, model_dimension, bias=False
        )

    def forward(self, x, mask=None):
        # 输入 x：[B, T, D]；投影后的 Q、K、V 形状不变。
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        # [B, T, D] @ [B, D, T] -> [B, T, T]
        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(key.shape[-1])

        if mask is not None:
            # 本例约定 True 表示允许关注，False 表示禁止关注。
            scores = scores.masked_fill(~mask, float("-inf"))

        # 沿 key 所在的最后一维归一化，每个 query 的权重之和为 1。
        attention_weights = torch.softmax(scores, dim=-1)

        # [B, T, T] @ [B, T, D] -> [B, T, D]
        output = attention_weights @ value

        return output, attention_weights


def main():
    torch.manual_seed(42)

    batch_size = 2
    sequence_length = 4
    model_dimension = 6

    x = torch.rand(batch_size, sequence_length, model_dimension)

    # 创建一次对象，后面多次调用时使用相同的权重。
    attention = SelfAttention(model_dimension)

    print("attention layer:")
    print(attention)

    # 不使用掩码：每个位置都可以关注所有位置。
    output, weights = attention(x)

    print("\nwithout mask:")
    print("input shape:", x.shape)
    print("output shape:", output.shape)
    print("weights shape:", weights.shape)
    print("batch 0 weights:")
    print(weights[0])

    # 使用因果掩码：每个位置只能关注自己及之前的位置。
    causal_mask = torch.tril(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool)
    )

    masked_output, masked_weights = attention(x, mask=causal_mask)

    print("\nwith causal mask:")
    print("output shape:", masked_output.shape)
    print("batch 0 weights:")
    print(masked_weights[0])

    # 验证允许位置上的权重仍然归一化。
    row_sums = masked_weights.sum(dim=-1)
    print(
        "\nall row sums are one:",
        torch.allclose(row_sums, torch.ones_like(row_sums)),
    )

    # 验证未来位置的权重全部为 0。
    future_weights = masked_weights.masked_select(~causal_mask)
    print("all future weights are zero:", torch.all(future_weights == 0))

    # 第 0 个 token 只能关注自己，因此输出应等于它自己的 V。
    projected_value = attention.value_projection(x)
    print(
        "first output equals first value:",
        torch.allclose(masked_output[:, 0], projected_value[:, 0]),
    )

    # 查看 nn.Module 自动登记的可学习参数。
    print("\nlearnable parameters:")
    for name, parameter in attention.named_parameters():
        print(name, parameter.shape)


# 直接运行时执行示例；被其他模块导入时不执行。
if __name__ == "__main__":
    main()
