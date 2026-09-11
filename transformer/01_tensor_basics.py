import torch

# 一维张量：可以暂时理解为向量
vector = torch.tensor([1.0, 2.0, 3.0])

# 二维张量：可以暂时理解为矩阵
matrix = torch.tensor(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
)

print("vector:")
print(vector)
print("vector shape:", vector.shape)

print("\nmatrix:")
print(matrix)
print("matrix shape:", matrix.shape)
print("matrix dtype:", matrix.dtype)

print("\nindexing:")
print("第 0 行:", matrix[0])
print("第 1 行:", matrix[1])
print("第 0 行第 1 列:", matrix[0, 1])
print("所有行的第 1 列:", matrix[:, 1])
