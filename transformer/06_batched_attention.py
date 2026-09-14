import math

import torch


def scaled_dot_product_attention(query, key, value):
    """计算缩放点积注意力，并返回输出、权重和缩放后的分数。"""
    key_dimension = key.shape[-1]

    # [B, T, Dk] @ [B, Dk, T] -> [B, T, T]
    scores = query @ key.transpose(-2, -1)
    scaled_scores = scores / math.sqrt(key_dimension)

    # 对每个 query 对应的最后一行分数做 Softmax。
    attention_weights = torch.softmax(scaled_scores, dim=-1)

    # [B, T, T] @ [B, T, Dv] -> [B, T, Dv]
    output = attention_weights @ value

    return output, attention_weights, scaled_scores


torch.manual_seed(42)

batch_size = 2
sequence_length = 3
key_dimension = 4
value_dimension = 5

query = torch.rand(
    batch_size,
    sequence_length,
    key_dimension,
)

key = torch.rand(
    batch_size,
    sequence_length,
    key_dimension,
)

value = torch.rand(
    batch_size,
    sequence_length,
    value_dimension,
)

print("query shape:", query.shape)
print("key shape:", key.shape)
print("value shape:", value.shape)

output, attention_weights, scaled_scores = scaled_dot_product_attention(
    query,
    key,
    value,
)

print("\nscaled scores shape:", scaled_scores.shape)

print("\nattention_weights:")
print(attention_weights)
print("attention_weights shape:", attention_weights.shape)

row_sums = attention_weights.sum(dim=-1)
print("\nrow sums:")
print(row_sums)
print("row sums shape:", row_sums.shape)

print("\noutput:")
print(output)
print("output shape:", output.shape)

# 分别查看两个 batch 的注意力矩阵。
print("\nbatch 0 attention weights:")
print(attention_weights[0])

print("\nbatch 1 attention weights:")
print(attention_weights[1])

# 单独计算第 0 个 batch，验证批量计算没有混合不同 batch 的数据。
manual_scores_0 = (
    query[0] @ key[0].transpose(-2, -1)
    / math.sqrt(key_dimension)
)

print("\nbatched scaled_scores[0]:")
print(scaled_scores[0])

print("\nmanually calculated scores for batch 0:")
print(manual_scores_0)

print(
    "\nbatch 0 scores are close:",
    torch.allclose(manual_scores_0, scaled_scores[0]),
)
