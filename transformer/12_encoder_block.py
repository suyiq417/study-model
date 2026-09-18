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


class EncoderBlock(nn.Module):
    """包含自注意力和前馈网络的 Post-LN 编码器块。"""

    def __init__(
        self,
        model_dimension,
        num_heads,
        hidden_dimension,
        dropout_probability=0.1,
    ):
        super().__init__()

        self.attention = MultiHeadSelfAttention(
            model_dimension,
            num_heads,
        )
        self.feed_forward = FeedForward(
            model_dimension,
            hidden_dimension,
        )

        # 两个子层分别使用自己的 LayerNorm 参数。
        self.normalization_1 = nn.LayerNorm(model_dimension)
        self.normalization_2 = nn.LayerNorm(model_dimension)

        # 对两个子层的输出分别应用 Dropout。
        self.dropout_1 = nn.Dropout(dropout_probability)
        self.dropout_2 = nn.Dropout(dropout_probability)

    def forward(self, x, mask=None):
        # 自注意力让各个 token 交换信息。
        attention_output, attention_weights = self.attention(
            x, mask=mask
        )

        # 第一条残差连接：保留输入 x。
        # 所有张量的形状都是 [B, T, D]。
        hidden = self.normalization_1(
            x + self.dropout_1(attention_output)
        )

        # 前馈网络独立处理每个 token 的特征。
        feed_forward_output = self.feed_forward(hidden)

        # 第二条残差连接：保留当前的 hidden。
        output = self.normalization_2(
            hidden + self.dropout_2(feed_forward_output)
        )

        return output, attention_weights


def main():
    torch.manual_seed(42)

    batch_size = 2
    sequence_length = 3
    model_dimension = 8
    num_heads = 2
    hidden_dimension = 32

    # 模拟已经加入位置编码的输入。
    x = torch.rand(batch_size, sequence_length, model_dimension)

    encoder_block = EncoderBlock(
        model_dimension=model_dimension,
        num_heads=num_heads,
        hidden_dimension=hidden_dimension,
        dropout_probability=0.1,
    )

    # 递归切换所有子模块到评估模式，让验证不受 Dropout 随机性影响。
    encoder_block.eval()

    # 普通 Encoder 允许双向关注，这里不使用因果掩码。
    output, attention_weights = encoder_block(x)

    print("input shape:", x.shape)
    print("output shape:", output.shape)
    print("attention weights shape:", attention_weights.shape)

    print("\nbatch 0, head 0 weights:")
    print(attention_weights[0, 0])

    row_sums = attention_weights.sum(dim=-1)
    print(
        "\nall attention row sums are one:",
        torch.allclose(row_sums, torch.ones_like(row_sums)),
    )

    # 单独运行第 0 个样本，验证不同 batch 之间不会交换信息。
    single_output, _ = encoder_block(x[:1])

    print(
        "single sample matches batch result:",
        torch.allclose(single_output, output[:1], atol=1e-6),
    )

    # 再接一个独立的 Encoder Block，观察形状仍保持不变。
    second_block = EncoderBlock(
        model_dimension=model_dimension,
        num_heads=num_heads,
        hidden_dimension=hidden_dimension,
        dropout_probability=0.1,
    )
    second_block.eval()

    second_output, _ = second_block(output)

    print("\nafter two encoder blocks:", second_output.shape)


# 直接运行时执行示例；被其他模块导入时不执行。
if __name__ == "__main__":
    main()
