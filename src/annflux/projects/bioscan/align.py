
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from annflux.repository.resultset import Resultset
from annflux.shared import AnnfluxSource
from annflux.algorithms.embeddings import compute_umap

class NonlinearAligner(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.Sigmoid(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class FeatureAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def train_aligner(
    source_features: np.ndarray,
    target_features: np.ndarray,
    checkpoint_path: str,
) -> np.ndarray:
    """Train a NonlinearAligner to map source_features into the target feature space.

    Returns the aligned source features as a numpy array.
    """
    src = torch.from_numpy(source_features.astype(np.float32))
    tgt = torch.from_numpy(target_features.astype(np.float32))

    # model = NonlinearAligner(
    #     input_dim=src.shape[1],
    #     hidden_dim=src.shape[1] // 2 if src.shape[1] > 2 else 32,
    #     output_dim=tgt.shape[1],
    # )
    model = FeatureAligner(src.shape[1], src.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50, min_lr=1e-6
    )

    num_epochs = 50000
    prev_loss = float("inf")
    best_loss = float("inf")
    no_improvement_count = 0
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        aligned = model(src)
        loss = criterion(aligned, tgt)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), checkpoint_path)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Best: {best_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        if abs(prev_loss - loss.item()) < 1e-5:
            no_improvement_count += 1
            if no_improvement_count >= 100:
                print(f"Early stopping at epoch {epoch}, Loss: {loss.item():.6f}")
                break
        else:
            no_improvement_count = 0
        prev_loss = loss.item()

    print(f"Loading best checkpoint (loss={best_loss:.6f}) from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    with torch.no_grad():
        aligned = model(src)
    return aligned.numpy()


def align(project_root: str, source_features_path: str = None):
    if source_features_path is None:
        source_features_path = os.path.join(project_root, "features", "embeddings.npz")
    source = AnnfluxSource(project_root)
    repo = source.repository
    result_set = repo.get(label=Resultset, tag="unseen").first()
    if result_set is None:
        raise RuntimeError(f"No resultset found in {project_root}")
    print(f"Using resultset: {result_set.path}")

    source_feat2 = np.load(source_features_path)["lastFull"].astype(np.float32)
    target_feat2 = np.load(result_set.last_full_path)["lastFull"].astype(np.float32)

    checkpoint_path = os.path.join(os.path.dirname(source_features_path), "aligner_best.pt")
    output_path = os.path.join(os.path.dirname(source_features_path), "embeddings_aligned.npz")

    aligned = train_aligner(source_feat2, target_feat2, checkpoint_path)
    np.savez(output_path, lastFull=aligned)
    print(f"Saved aligned features to {output_path}, shape={aligned.shape}")

    embed = compute_umap(np.concatenate([aligned, target_feat2]))
    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.scatter(embed[:10, 0], embed[:10, 1])
    plt.subplot(122)
    plt.scatter(embed[1000:1010, 0], embed[1000:1010, 1])
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align source features to the first resultset in a project")
    parser.add_argument("project_root", help="Path to the annflux project root")
    parser.add_argument("source_features", nargs="?", default=None, help="Path to source features .npz file (must contain 'lastFull'). Defaults to PROJECT_ROOT/features/embeddings.npz")
    args = parser.parse_args()
    align(args.project_root, args.source_features)