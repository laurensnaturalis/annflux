
import torch
import torch.nn as nn

from annflux.algorithms.embeddings import compute_umap

class NonlinearAligner(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class FeatureAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)



def m():
    import numpy as np
    # Example: 100 samples, 64-dimensional features
    source_feat2 = \
    np.load("/mnt/big/datasets/bioannflux/annflux/datarepo/resultset-20250926165503-a1bcf51d/dna_embeddings.npz")[
        "lastFull"].astype(np.float32)
    source_features = torch.from_numpy(source_feat2)
    target_feat2 = \
    np.load("/mnt/big/datasets/bioannflux/annflux/datarepo/resultset-20250926165503-a1bcf51d/image_embeddings.npz")[
        "lastFull"].astype(np.float32)
    target_features = torch.from_numpy(target_feat2)

    print(np.concatenate([source_feat2, target_feat2]).shape)
    embed = compute_umap(np.concatenate([source_feat2, target_feat2]))
    import matplotlib.pyplot as plt
    plt.scatter(embed[:1000, 0], embed[:1000, 1])
    plt.scatter(embed[1000:, 0], embed[1000:, 1])
    plt.show()
    # target_embed = compute_umap(target_features)

    print(source_features.shape)
    print(target_features.shape)

    model = FeatureAligner(input_dim=source_features.shape[1], output_dim=target_features.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    num_epochs = 5000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # Forward pass
        aligned_source = model(source_features)
        # Compute loss
        loss = criterion(aligned_source, target_features)
        # Backward pass
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    aligned_source = model(source_features)

    print(aligned_source.shape)
    embed = compute_umap(np.concatenate([aligned_source.detach().numpy(), target_feat2]))
    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.scatter(embed[:10, 0], embed[:10, 1])
    plt.subplot(122)
    plt.scatter(embed[1000:1010, 0], embed[1000:1010, 1])
    plt.show()


if __name__ == '__main__':
    m()