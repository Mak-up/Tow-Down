import torch
batch = 5
seq_len = [10, 8, 6, 4, 2]
window = 2
adj = torch.cat([torch.eye(seq_len[0]).unsqueeze(0) for i in range(batch)])
print(adj.shape)
print(adj[0])
for i in range(batch):
    for j in range(seq_len[i]):
        adj[i, j, max(0, j - window):min(seq_len[i], j + window + 1)] = 1.
print(adj[0])
print(torch.eye(seq_len[0]).unsqueeze(0).shape)
adj = torch.cat([torch.zeros(seq_len[0],seq_len[0]).unsqueeze(0) for i in range(batch)])
for i in range(batch):
    for j in range(seq_len[i]):
        adj[i, j, :seq_len[i]] = 1
print(adj[0])
print(adj.shape)    


