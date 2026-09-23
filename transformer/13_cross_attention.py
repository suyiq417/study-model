"""独立学习示例：包含完整类实现和运行验证。"""

import math

import torch
from torch import nn


class MultiHeadCrossAttention(nn.Module):
    """用目标序列生成 Q，用编码器输出生成 K、V，计算多头交叉注意力。"""

    def __init__(self, model_dimension, num_heads):
        super().__init__()

        # 与自注意力相同，每个头分到相同数量的特征。
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

        # 合并各头后再投影回模型维度。
        self.output_projection = nn.Linear(
            model_dimension, model_dimension, bias=False
        )

    def split_heads(self, tensor):
        batch_size, sequence_length, _ = tensor.shape

        # 从张量自身读取长度，因此目标序列 T 与源序列 S 可以不同。
        # L 表示当前张量的序列长度。
        # [B, L, D] -> [B, L, H, Dh] -> [B, H, L, Dh]
        tensor = tensor.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dimension,
        )
        return tensor.transpose(1, 2)

    def forward(self, x, memory, mask=None):
        # x：[B, T, D] 是目标序列特征；memory：[B, S, D] 是编码器输出。
        batch_size, target_length, model_dimension = x.shape

        # 自注意力的 Q、K、V 都来自同一输入；这里 Q 来自 x，K、V 来自 memory。
        query = self.split_heads(self.query_projection(x))
        key = self.split_heads(self.key_projection(memory))
        value = self.split_heads(self.value_projection(memory))

        # [B, H, T, Dh] @ [B, H, Dh, S] -> [B, H, T, S]
        scores = query @ key.transpose(-2, -1)

        # 每个头只使用 Dh 个特征，因此缩放因子是 sqrt(Dh)。
        scores = scores / math.sqrt(self.head_dimension)

        if mask is not None:
            # True 表示允许关注；掩码需能广播到 [B, H, T, S]。
            # 每个目标位置至少保留一个源位置，否则 Softmax 会产生 NaN。
            scores = scores.masked_fill(~mask, float("-inf"))

        # 沿源序列维归一化：每个目标位置对允许的源位置分配权重。
        attention_weights = torch.softmax(scores, dim=-1)

        # 每个头分别对自己的 value 加权求和。
        # [B, H, T, S] @ [B, H, S, Dh] -> [B, H, T, Dh]
        head_outputs = attention_weights @ value

        # 权重有 T 行，所以合并后的输出保留目标序列长度 T。
        # [B, H, T, Dh] -> [B, T, H, Dh] -> [B, T, D]
        combined = head_outputs.transpose(1, 2).reshape(
            batch_size,
            target_length,
            model_dimension,
        )

        output = self.output_projection(combined)

        return output, attention_weights


def main():
    torch.manual_seed(42)

    batch_size = 2
    target_length = 3
    source_length = 5
    model_dimension = 8
    num_heads = 2

    # 故意让 T != S，观察注意力权重和输出分别跟随哪个长度。
    x = torch.rand(batch_size, target_length, model_dimension)
    memory = torch.rand(batch_size, source_length, model_dimension)

    attention = MultiHeadCrossAttention(model_dimension, num_heads)

    # 不使用掩码：每个目标位置都能关注所有源位置。
    output, weights = attention(x, memory)

    print("without mask:")
    print("decoder input shape:", x.shape)
    print("encoder memory shape:", memory.shape)
    print("output shape:", output.shape)
    print("weights shape:", weights.shape)

    print("\nbatch 0, head 0 weights:")
    print(weights[0, 0])

    row_sums = weights.sum(dim=-1)
    print(
        "\nall row sums are one:",
        torch.allclose(row_sums, torch.ones_like(row_sums)),
    )

    # 两个样本共用一份源序列补齐掩码，最后一个源位置不参与注意力。
    source_valid = torch.tensor([True, True, True, True, False])

    # [S] -> [1, 1, 1, S]，广播到所有样本、注意力头和目标位置。
    source_mask = source_valid.reshape(1, 1, 1, source_length)

    masked_output, masked_weights = attention(
        x, memory, mask=source_mask
    )

    print("\nwith source padding mask:")
    print(masked_weights[0, 0])
    print(
        "last source position has zero weight:",
        torch.all(masked_weights[..., -1] == 0),
    )

    masked_row_sums = masked_weights.sum(dim=-1)
    print(
        "masked row sums are one:",
        torch.allclose(
            masked_row_sums,
            torch.ones_like(masked_row_sums),
        ),
    )

    # 如果掩码生效，被屏蔽源位置的内容改变后，输出应保持不变。
    changed_memory = memory.clone()
    changed_memory[:, -1, :] += 100.0

    changed_output, _ = attention(
        x, changed_memory, mask=source_mask
    )

    print(
        "masked source content does not affect output:",
        torch.allclose(masked_output, changed_output, atol=1e-6),
    )


# 直接运行时执行示例；被其他模块导入时不执行。
if __name__ == "__main__":
    main()
