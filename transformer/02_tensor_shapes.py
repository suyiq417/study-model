import torch

x = torch.arange(24)

print("x:")
print(x)
print("x shape:", x.shape)
print("x dtype:", x.dtype)

print("x[1]:", x[1])
print("x[1] shape:", x[1].shape)

print("x[:1]:", x[:1])
print("x[:1] shape:", x[:1].shape)

# reshape 只改变张量的形状，不改变元素数量和排列顺序。
x = x.reshape(2, 3, 4)
print("\nreshaped x:")
print(x)
print("reshaped x shape:", x.shape)
x = x.reshape(2, 12)
print("\nreshaped x:")
print(x)
print("reshaped x shape:", x.shape)
x = x.reshape(2, 3, 4)
print("\nreshaped x:")
print(x)
print("reshaped x shape:", x.shape)
print("reshaped x dtype:", x.dtype)

print("x[0]:", x[0])
print("x[0] shape:", x[0].shape)

print("x[:, 0]:", x[:, 0])
print("x[:, 0] shape:", x[:, 0].shape)

print("x[:, :, 0]:", x[:, :, 0])
print("x[:, :, 0] shape:", x[:, :, 0].shape)

print("\nindexing:")
print("第 0 个样本:")
print(x[0])
print("第 0 个样本的第 1 个 token:")
print(x[0, 1])
print("第 0 个样本的第 1 个 token 的第 2 个维度:")
print(x[0, 1, 2])

query = torch.rand(2, 3, 4)
key = torch.rand(2, 3, 4)
print("\nquery shape:", query.shape)
print("key shape:", key.shape)

# 交换序列维和特征维，为批量矩阵乘法做准备。
key_T = key.transpose(1, 2)
print("\nkey_T shape:", key_T.shape)

print("key_T:")
print(key_T)

scores = query @ key_T
print("\nscores shape:", scores.shape)
