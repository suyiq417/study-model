import math

import torch

query = torch.tensor(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
)

key = torch.tensor(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
)

value = torch.tensor(
    [
        [10.0, 0.0],
        [0.0, 10.0],
        [5.0, 5.0],
    ]
)

print("query shape:", query.shape)
print("key shape:", key.shape)
print("value shape:", value.shape)


# 第一步：计算每个 query 与所有 key 的点积分数。
scores = query @ key.T
print("\nscores:")
print(scores)
print("scores shape:", scores.shape)

key_dimension = key.shape[-1]

# 第二步：缩放分数，避免 Softmax 的输入值随维度增大而过大。
scaled_scores = scores / math.sqrt(key_dimension)
print("\nkey dimension:", key_dimension)
print("scaled_scores:")
print(scaled_scores)


# 第三步：对每一行做 Softmax，得到总和为 1 的注意力权重。
attention_weights = torch.softmax(
    scaled_scores,
    dim=-1,
)
print("\nattention_weights:")
print(attention_weights)
print("\nrow sums:")
print(attention_weights.sum(dim=-1))

# 第四步：使用注意力权重对 value 进行加权求和。
output = attention_weights @ value
print("\noutput:")
print(output)
print("output shape:", output.shape)

manual_output_0 = (
    attention_weights[0, 0] * value[0]
    + attention_weights[0, 1] * value[1]
    + attention_weights[0, 2] * value[2]
)
print("\nmanual_output_0:")
print(manual_output_0)

print("automatic output 0:")
print(output[0])

print(
    "Are they close?",
    torch.allclose(manual_output_0, output[0]),
)
