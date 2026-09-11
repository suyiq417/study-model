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
print("第 1 行第 2 列:", matrix[1, 2])
print("第 0 列:", matrix[:, 0])

zero_matrix = torch.zeros((3, 4))
print("\nzero_matrix:")
print(zero_matrix)
print("zero_matrix shape:", zero_matrix.shape)
print("zero_matrix dtype:", zero_matrix.dtype)

random_matrix = torch.rand((2, 3))
print("\nrandom_matrix:")
print(random_matrix)
print("random_matrix shape:", random_matrix.shape)
print("random_matrix dtype:", random_matrix.dtype)

value1 = matrix[1, 2]
print(value1.shape)
print(value1.dtype)
print(value1.item())
value2 = matrix[:, 1]
print(value2.shape)
print(value2.dtype)
print(value2.tolist())

a = torch.tensor(
    [
        [1.0, 2.0], 
        [3.0, 4.0],
    ]
)

b = torch.tensor(
    [
        [5.0, 6.0],
        [7.0, 8.0],
    ]
)

print("\nelement-wise operations:")
print("a + b:")
print(a + b)

print("a * b:")
print(a * b)

left = torch.tensor(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
)

right = torch.tensor(
    [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ]
)

result = left @ right #另一种写法是 torch.matmul(left, right)
print("\nmatrix multiplication:")
print("left shape:", left.shape)
print("right shape:", right.shape)
print("result shape:", result.shape)
print("result:")
print(result)

print("\ntranspose:")
print("left:")
print(left)
print("left.T:")
print(left.T)
print("left.T shape:", left.T.shape)

query = torch.tensor(
    [
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
)

key = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ]
)

print("\nquery shape:", query.shape)
print("key shape:", key.shape)
print("key.T:")
print(key.T)
print("key.T shape:", key.T.shape)
scores = query @ key.T
print("scores:")
print(scores)
print("scores shape:", scores.shape)
