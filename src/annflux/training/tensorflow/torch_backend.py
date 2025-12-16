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

from torch.utils.data.dataset import _T_co


def l2_normalize(x, axis=1):
    norm = torch.norm(x, p=2, dim=axis, keepdim=True)
    return x / norm


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

    def __getitem__(self, idx) -> _T_co:  # ty: ignore[invalid-method-override]
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

    # PyTorch model definition
    class LinearModel(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
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
