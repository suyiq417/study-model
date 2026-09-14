import math

import torch
from torch import nn

torch.manual_seed(42)

max_sequence_length = 6
embedding_dim = 4

# 转成列向量，使每个位置都能与所有频率项进行广播运算。
positions = torch.arange(
    max_sequence_length,
    dtype=torch.float32,
).unsqueeze(1)

even_dimensions = torch.arange(
    0,
    embedding_dim,
    2,
    dtype=torch.float32,
)

print("positions:")
print(positions)
print("positions shape:", positions.shape)

print("\neven_dimensions:")
print(even_dimensions)
print("even_dimensions shape:", even_dimensions.shape)

# 每两个特征维度共享一个频率，分别用于 sin 和 cos。
div_term = torch.exp(
    even_dimensions * (-math.log(10000.0) / embedding_dim)
)

print("\ndiv_term:")
print(div_term)
print("div_term shape:", div_term.shape)

angles = positions * div_term
print("\nangles:")
print(angles)
print("angles shape:", angles.shape)

positional_encoding = torch.zeros(
    max_sequence_length,
    embedding_dim,
)

# 偶数维使用 sin，奇数维使用 cos。
positional_encoding[:, 0::2] = torch.sin(angles)
positional_encoding[:, 1::2] = torch.cos(angles)

print("\npositional_encoding:")
print(positional_encoding)
print("positional_encoding shape:", positional_encoding.shape)

embedding = nn.Embedding(
    num_embeddings=5,
    embedding_dim=embedding_dim,
)

token_ids = torch.tensor(
    [
        [2, 2, 2, 2],
    ]
)

token_embeddings = embedding(token_ids)

sequence_length = token_ids.shape[1]

# 增加 batch 维：[T, D] -> [1, T, D]。
positional_encodings = positional_encoding[:sequence_length]
positional_encodings = positional_encodings.unsqueeze(0)

transformer_input = token_embeddings + positional_encodings

print("\ntoken_IDs:")
print(token_ids)
print("token_embeddings[0, 0]:", token_embeddings[0, 0])
print("token_embeddings[0, 1]:", token_embeddings[0, 1])
print("token embeddings shape:", token_embeddings.shape)
print("positional encodings shape:", positional_encodings.shape)

print("\ntransformer_input:")
print(transformer_input)
print("transformer_input[0, 0]:", transformer_input[0, 0])
print("transformer_input[0, 1]:", transformer_input[0, 1])
print("transformer_input shape:", transformer_input.shape)

token_ids_2 = torch.tensor(
    [
        [2, 2, 2, 2],
        [1, 3, 4, 0],
    ]
)

token_embeddings_2 = embedding(token_ids_2)

sequence_length_2 = token_ids_2.shape[1]

positional_encodings_2 = positional_encoding[:sequence_length_2]
positional_encodings_2 = positional_encodings_2.unsqueeze(0)

# [1, T, D] 的位置编码会广播到两个 batch。
transformer_input_2 = token_embeddings_2 + positional_encodings_2

print("\ntoken_embeddings_2 shape:", token_embeddings_2.shape)
print("token_embeddings_2:")
print(token_embeddings_2)

print("\npositional_encodings_2:")
print(positional_encodings_2)
print("positional_encodings_2 shape:", positional_encodings_2.shape)

print("\ntransformer_input_2:")
print(transformer_input_2)
print("transformer_input_2 shape:", transformer_input_2.shape)
