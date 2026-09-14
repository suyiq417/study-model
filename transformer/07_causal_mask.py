import math

import torch


def scaled_dot_product_attention(query, key, value, mask=None):
    """计算支持可选掩码的缩放点积注意力。"""
    key_dimension = key.shape[-1]

    # [B, T, D] @ [B, D, T] -> [B, T, T]
    scores = query @ key.transpose(-2, -1)
    scaled_scores = scores / math.sqrt(key_dimension)

    if mask is not None:
        # False 表示禁止关注；在 Softmax 前将对应分数设为负无穷。
        scaled_scores = scaled_scores.masked_fill(
            ~mask,
            float("-inf"),
        )

    attention_weights = torch.softmax(scaled_scores, dim=-1)

    # [B, T, T] @ [B, T, D] -> [B, T, D]
    output = attention_weights @ value

    return output, attention_weights, scaled_scores


torch.manual_seed(42)

batch_size = 1
sequence_length = 4
model_dimension = 2

query = torch.rand(
    batch_size,
    sequence_length,
    model_dimension,
)

key = torch.rand(
    batch_size,
    sequence_length,
    model_dimension,
)

value = torch.rand(
    batch_size,
    sequence_length,
    model_dimension,
)

# 下三角中的 True 允许关注当前位置和之前的位置。
causal_mask = torch.tril(
    torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
    )
)

print("causal mask:")
print(causal_mask)
print("causal mask shape:", causal_mask.shape)

output, attention_weights, masked_scores = (
    scaled_dot_product_attention(
        query,
        key,
        value,
        mask=causal_mask,
    )
)

print("\nmasked scores:")
print(masked_scores)

print("\nattention weights:")
print(attention_weights)
print("attention weights shape:", attention_weights.shape)

print("\nrow sums:")
row_sums = attention_weights.sum(dim=-1)
print(row_sums)
print(
    "all row sums are one:",
    torch.allclose(row_sums, torch.ones_like(row_sums)),
)

print("\noutput:")
print(output)
print("output shape:", output.shape)

# 取出右上三角（未来位置）的权重，验证它们全部为 0。
future_weights = attention_weights.masked_select(~causal_mask)

print("\nfuture attention weights:")
print(future_weights)

print(
    "all future weights are zero:",
    torch.all(future_weights == 0),
)
