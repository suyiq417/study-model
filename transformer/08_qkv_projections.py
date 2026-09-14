import torch
from torch import nn

torch.manual_seed(42)

batch_size = 2
sequence_length = 3
model_dimension = 4
key_dimension = 3
value_dimension = 5

# 模拟 Token Embedding 与位置编码相加后的输入。
x = torch.rand(
    batch_size,
    sequence_length,
    model_dimension,
)

print("x shape:", x.shape)

# 使用三组独立的可学习权重，将同一个输入投影为 Q、K、V。
query_projection = nn.Linear(
    in_features=model_dimension,
    out_features=key_dimension,
    bias=False,
)

key_projection = nn.Linear(
    in_features=model_dimension,
    out_features=key_dimension,
    bias=False,
)

value_projection = nn.Linear(
    in_features=model_dimension,
    out_features=value_dimension,
    bias=False,
)

print("\nquery projection:")
print(query_projection)
print("query weight shape:", query_projection.weight.shape)

print("\nkey projection:")
print(key_projection)
print("key weight shape:", key_projection.weight.shape)

print("\nvalue projection:")
print(value_projection)
print("value weight shape:", value_projection.weight.shape)

# nn.Linear 只改变最后一个特征维度，不改变 batch 和序列维。
query = query_projection(x)
key = key_projection(x)
value = value_projection(x)

print("\nquery shape:", query.shape)
print("key shape:", key.shape)
print("value shape:", value.shape)

# bias=False 时，线性层等价于 x @ weight.T。
manual_query = x @ query_projection.weight.T

print("\nquery from Linear:")
print(query)

print("\nmanually calculated query:")
print(manual_query)

print(
    "\nquery results are close:",
    torch.allclose(query, manual_query),
)

# Q 和 K 的最后一维相同，才能计算每对 token 的点积分数。
scores = query @ key.transpose(-2, -1)

print("\nscores shape:", scores.shape)
