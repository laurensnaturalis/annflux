import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter, defaultdict
import numpy as np
import os
from typing import Dict, List

def l2_normalize(x, axis=1):
    norm = torch.norm(x, p=2, dim=axis, keepdim=True)
    return x / norm


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning loss (Khosla et al., 2020)"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: [batch_size, feature_dim] - normalized embeddings
        labels: [batch_size] - class indices (single-label)
        """
        device = features.device
        # Create label mask: [batch_size, batch_size]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask out self-contrast (diagonal)
        logits_mask = torch.ones_like(mask).to(device)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Mean of log-likelihood over positives
        mask_positives = mask.sum(1)
        # Avoid division by zero - items with no positives (shouldn't happen with batch construction)
        mask_positives = torch.where(mask_positives == 0, torch.ones_like(mask_positives), mask_positives)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_positives

        # Loss
        loss = -mean_log_prob_pos.mean()
        return loss


class SupConModel(nn.Module):
    """Encoder + projection head for supervised contrastive learning"""
    def __init__(self, input_dim, projection_dim=128, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim

        # Encoder: transforms input features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )

        # Projection head: maps to contrastive space
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x, return_projection=True):
        h = self.encoder(x)
        if return_projection:
            z = self.projection_head(h)
            z = l2_normalize(z, axis=1)
            return h, z
        return h


class BalanceDataset(Dataset):
    def __init__(self, x_set, y_set, balance: bool = False):
        self.x = torch.tensor(x_set, dtype=torch.float32)
        self.y = torch.tensor(y_set, dtype=torch.float32)
        class_counts = Counter(torch.argmax(self.y, dim=1).numpy())
        self.classes_ = list(class_counts.keys())
        if balance:
            self.class_weights = None
        else:
            self.class_weights = torch.tensor(
                [class_counts[x_] for x_ in self.classes_], dtype=torch.float32
            )
            self.class_weights /= self.class_weights.sum()
        self.class_to_indices: Dict[int, List[int]] = defaultdict(list)
        for i_, class_ in enumerate(torch.argmax(self.y, dim=1)):
            self.class_to_indices[class_.item()].append(i_)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):  # ty: ignore[invalid-method-override]
        if self.class_weights is None:
            return self.x[idx], self.y[idx]  # ty: ignore[invalid-return-type]
        else:
            class_ = np.random.choice(self.classes_, p=self.class_weights.numpy())
            idx = np.random.choice(self.class_to_indices[class_])
            return self.x[idx], self.y[idx]  # ty: ignore[invalid-return-type]


def linear_retraining(state, status_callback):
    if state.labeled_indices is None or len(state.labeled_indices) == 0:
        return
    print(f"{len(state.labeled_indices)=}")

    balance = True
    binarizer = MultiLabelBinarizer()
    no_label_for_labeled_idx = np.where(
        state.label_array[state.labeled_indices] is None
    )[0]
    if len(no_label_for_labeled_idx) > 0:
        raise RuntimeError(
            f"no label for idx {np.array(state.labeled_indices)[no_label_for_labeled_idx]}"
        )
    binarizer.fit(state.label_array[state.labeled_indices])
    targets = binarizer.transform(state.label_array[state.labeled_indices])
    test_targets = binarizer.transform(
        state.label_array_test[state.labeled_test_indices]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        state.features[state.labeled_indices], targets, test_size=0.10, random_state=42
    )

    # PyTorch model definition - 2-layer MLP with GELU
    class LinearModel(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
            )
            self.classifier = nn.Linear(input_dim, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = l2_normalize(x, axis=1)
            return torch.sigmoid(self.classifier(x))

    model = LinearModel(state.features.shape[1], len(binarizer.classes_))
    # model2 = nn.Sequential(
    #     nn.Linear(state.features.shape[1], state.features.shape[1]),
    #     nn.ReLU(),
    # )

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    # DataLoaders
    train_dataset = BalanceDataset(x_train, y_train, balance=balance)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_dataset = BalanceDataset(x_test, y_test, balance=False)
    val_loader = DataLoader(val_dataset, batch_size=1024)

    # Training loop
    best_val_loss = float("inf")
    weights_path = os.path.join(state.annflux_folder, "linear.weights.pt")
    patience = 10
    epochs_no_improve = 0

    for epoch in range(200):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                outputs = model(x_val)
                val_loss += criterion(outputs, y_val).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        print("epoch, val_loss", epoch, val_loss, scheduler.get_last_lr())

        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), weights_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

        status_callback(epoch, val_loss)

    # Load best weights
    # model.load_state_dict(torch.load(weights_path))

    # Test
    with torch.no_grad():
        test_features = torch.tensor(
            state.features[state.labeled_test_indices], dtype=torch.float32
        )
        test_predictions = model(test_features).numpy()
        acc_test = accuracy_score(test_targets, (test_predictions > 0.5).astype(int))
        print(f"linear from features acc = {acc_test}")

    # Recompute features
    state.g_quick_status = "recomputing features"
    with torch.no_grad():
        state.features = model.features(
            torch.tensor(state.features, dtype=torch.float32)
        ).numpy()

    return weights_path


def supcon_retraining(state, status_callback):
    """
    Supervised Contrastive Learning retraining (Khosla et al., 2020)
    Drop-in replacement for linear_retraining with better feature adaptation.
    """
    if state.labeled_indices is None or len(state.labeled_indices) == 0:
        return
    print(f"supcon_retraining: {len(state.labeled_indices)=}")

    # Prepare labels: treat each unique multi-label combination as a distinct class
    labels_raw = state.label_array[state.labeled_indices]
    # Convert each label set to a canonical string representation
    def canonical_label(lbl):
        if isinstance(lbl, (list, tuple)):
            return ",".join(sorted(str(x) for x in lbl)) if len(lbl) > 0 else "_empty_"
        return str(lbl)
    labels_list = [canonical_label(lbl) for lbl in labels_raw]
    unique_labels = sorted(set(labels_list))
    label_to_idx = {label_name: i for i, label_name in enumerate(unique_labels)}
    y_indices = np.array([label_to_idx[label_name] for label_name in labels_list])
    print(f"supcon_retraining: {len(unique_labels)} unique label combinations from {len(labels_list)} samples")

    # Train/val split
    x_train, x_val, y_train, y_val = train_test_split(
        state.features[state.labeled_indices], y_indices, test_size=0.10, random_state=42
    )

    # Convert to tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Model
    input_dim = state.features.shape[1]
    model = SupConModel(input_dim=input_dim, projection_dim=128, hidden_dim=input_dim)
    criterion = SupConLoss(temperature=0.07)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    # Training
    batch_size = 512  # Increased from 256 for more negatives in contrastive learning
    mixup_alpha = 0.2  # Mixup interpolation parameter
    mixup_prob = 0.5  # Probability of applying mixup
    best_val_loss = float("inf")
    patience = 30  # Increased from 15 to allow longer training
    min_delta = 0.001  # Minimum improvement to count as improvement
    epochs_no_improve = 0
    weights_path = os.path.join(state.annflux_folder, "supcon.weights.pt")

    for epoch in range(200):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # Shuffle and batch
        perm = torch.randperm(len(x_train))
        for i in range(0, len(x_train), batch_size):
            idx = perm[i:i+batch_size]
            # Need at least 2 samples per class for contrastive loss
            if len(idx) < 2:
                continue
            x_batch = x_train[idx]
            y_batch = y_train[idx]

            # Mixup: interpolate between random pairs of embeddings
            if np.random.rand() < mixup_prob and len(idx) > 1:
                # Generate random permutation for mixing
                perm_idx = torch.randperm(len(idx))
                x_batch_perm = x_batch[perm_idx]
                y_batch_perm = y_batch[perm_idx]
                # Sample mixing coefficient from Beta distribution
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                # Mix embeddings
                x_batch = lam * x_batch + (1 - lam) * x_batch_perm
                # For labels: use soft assignment - treat as positive for both if mixed
                # Simple approach: keep dominant label
                if lam >= 0.5:
                    y_batch = y_batch
                else:
                    y_batch = y_batch_perm

            optimizer.zero_grad()
            h, z = model(x_batch, return_projection=True)
            loss = criterion(z, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            if len(x_val) >= 2:
                h_val, z_val = model(x_val, return_projection=True)
                val_loss = criterion(z_val, y_val).item()
            else:
                val_loss = train_loss

        scheduler.step(val_loss)
        print(f"supcon epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), weights_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"SupCon early stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs)")
                break

        status_callback(epoch, val_loss)

    # Load best and recompute features using encoder (not projection head)
    state.g_quick_status = "recomputing features (SupCon)"
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    with torch.no_grad():
        all_features = torch.tensor(state.features, dtype=torch.float32)
        batch_size = 1024
        new_features = []
        for i in range(0, len(all_features), batch_size):
            batch = all_features[i:i+batch_size]
            h = model(batch, return_projection=False)
            h = l2_normalize(h, axis=1)
            new_features.append(h.numpy())
        state.features = np.vstack(new_features)

    print(f"supcon_retraining complete. Features shape: {state.features.shape}")
    return weights_path
