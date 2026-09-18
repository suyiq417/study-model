"""独立学习示例：包含完整类实现和运行验证。"""

import math

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """将特征分成多个头，分别计算注意力，再合并并投影。"""

    def __init__(self, model_dimension, num_heads):
        super().__init__()

        # 本实现将模型特征平均分给各个头。
        if num_heads <= 0 or model_dimension % num_heads != 0:
            raise ValueError(
                "num_heads 必须是正整数，且能整除 model_dimension。"
            )

        self.num_heads = num_heads
        self.head_dimension = model_dimension // num_heads

        self.query_projection = nn.Linear(
            model_dimension, model_dimension, bias=False
        )
        self.key_projection = nn.Linear(
            model_dimension, model_dimension, bias=False
        )
        self.value_projection = nn.Linear(
            model_dimension, model_dimension, bias=False
        )

        # 对拼接后的多头结果做线性变换，混合各个头的信息。
        self.output_projection = nn.Linear(
            model_dimension, model_dimension, bias=False
        )

    def split_heads(self, tensor):
        batch_size, sequence_length, _ = tensor.shape

        # 将特征维拆成“头数 × 每头特征维”。
        # [B, T, D] -> [B, T, H, Dh]
        tensor = tensor.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dimension,
        )

        # 把头放在序列维之前，使每个头独立执行矩阵乘法。
        # [B, T, H, Dh] -> [B, H, T, Dh]
        return tensor.transpose(1, 2)

    def forward(self, x, mask=None):
        batch_size, sequence_length, model_dimension = x.shape

        # 先进行线性投影，再将 Q、K、V 拆分为多个头。
        query = self.split_heads(self.query_projection(x))
        key = self.split_heads(self.key_projection(x))
        value = self.split_heads(self.value_projection(x))

        # [B, H, T, Dh] @ [B, H, Dh, T] -> [B, H, T, T]
        scores = query @ key.transpose(-2, -1)

        # 每次点积只使用一个头的特征，因此除以 sqrt(Dh)。
        scores = scores / math.sqrt(self.head_dimension)

        if mask is not None:
            # 本例使用 [T, T] 布尔掩码，广播到所有 batch 和 head。
            # True 表示允许关注。
            # 每行至少保留一个位置；全被屏蔽时 Softmax 会产生 NaN。
            scores = scores.masked_fill(~mask, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)

        # 每个头分别对自己的 value 加权求和。
        # [B, H, T, T] @ [B, H, T, Dh] -> [B, H, T, Dh]
        head_outputs = attention_weights @ value

        # 先把同一个 token 的所有头放在一起，再合并特征。
        # [B, H, T, Dh] -> [B, T, H, Dh] -> [B, T, D]
        combined = head_outputs.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            model_dimension,
        )

        output = self.output_projection(combined)

        return output, attention_weights


def main():
    torch.manual_seed(42)

    batch_size = 2
    sequence_length = 3
    model_dimension = 8
    num_heads = 2

    x = torch.rand(batch_size, sequence_length, model_dimension)

    attention = MultiHeadSelfAttention(model_dimension, num_heads)

    # 观察投影、拆分和合并的形状。
    projected_query = attention.query_projection(x)
    split_query = attention.split_heads(projected_query)

    print("input shape:", x.shape)
    print("projected query shape:", projected_query.shape)
    print("split query shape:", split_query.shape)

    # 将拆分后的 Q 合并回来，验证元素顺序保持正确。
    restored_query = split_query.transpose(1, 2).reshape(
        batch_size, sequence_length, model_dimension
    )
    print(
        "split and merge preserve query:",
        torch.allclose(restored_query, projected_query),
    )

    # 不使用掩码，观察不同头各自的注意力矩阵。
    output, weights = attention(x)

    print("\nwithout mask:")
    print("output shape:", output.shape)
    print("weights shape:", weights.shape)
    print("batch 0, head 0:")
    print(weights[0, 0])
    print("batch 0, head 1:")
    print(weights[0, 1])

    # 同一份因果掩码应用到所有 batch 和所有头。
    causal_mask = torch.tril(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool)
    )

    masked_output, masked_weights = attention(x, mask=causal_mask)

    print("\nwith causal mask:")
    print("output shape:", masked_output.shape)
    print("batch 0, head 0:")
    print(masked_weights[0, 0])
    print("batch 0, head 1:")
    print(masked_weights[0, 1])

    # 每个 batch、每个头、每个 query 的权重之和都应为 1。
    row_sums = masked_weights.sum(dim=-1)
    print(
        "\nall row sums are one:",
        torch.allclose(row_sums, torch.ones_like(row_sums)),
    )

    future_weights = masked_weights.masked_select(~causal_mask)
    print("all future weights are zero:", torch.all(future_weights == 0))


# 直接运行时执行示例；被其他模块导入时不执行。
if __name__ == "__main__":
    main()
