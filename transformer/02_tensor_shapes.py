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

key_T = key.transpose(1, 2) # 从0开始数，1表示第2个维度，2表示第3个维度
print("\nkey_T shape:", key_T.shape)

print("key_T:")
print(key_T)

scores = query @ key_T
print("\nscores shape:", scores.shape)