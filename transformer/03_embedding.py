import torch
from torch import nn

torch.manual_seed(42)

vocab_size = 5
embedding_dim = 3

embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim,
)

print("embedding:")
print(embedding)

print("\nembedding.weight:")
print(embedding.weight)
print("weight shape:", embedding.weight.shape)

token_ids = torch.tensor(
    [
        [0, 2, 1, 3],
        [3, 1, 4, 0],
    ]
)

print("\ntoken IDs:")
print(token_ids)
print("token IDs shape:", token_ids.shape)
print("token IDs dtype:", token_ids.dtype)

embedded = embedding(token_ids)
print("\nembedded:")
print(embedded)
print("embedded shape:", embedded.shape)

print("\nlookup verification:")
print("token_ids[0, 0]:", token_ids[0, 0])
print("token_ids[0, 1]:", token_ids[0, 1])
print("embedded[0, 0]:", embedded[0, 0])
print("embedded[0, 1]:", embedded[0, 1])
print("embedding.weight[0]:", embedding.weight[0])
print("embedding(torch.tensor(0)):", embedding(torch.tensor(0)))

print(
    "Are they equal?",
    torch.equal(embedded[0, 0], embedding.weight[0]),
)

embedding_dim_4 = 4

embedding_4 = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim_4,
)

print("\nsame token verification:")
print("token_ids[0, 2]:", token_ids[0, 2])
print("token_ids[1, 1]:", token_ids[1, 1])
print("embedded[0, 2]:", embedded[0, 2])
print("embedded[1, 1]:", embedded[1, 1])

print(
    "same token has same embedding:",
    torch.equal(embedded[0, 2], embedded[1, 1]),
)

embedded_4 = embedding_4(token_ids)

print("\nembedding with embedding_dim=4:")
print(embedding_4)
print("embedding_4 weight shape:", embedding_4.weight.shape)
print("embedded_4 shape:", embedded_4.shape)