import itertools
import pickle
from collections import Counter
from xml.etree.ElementInclude import default_loader
from collections import defaultdict
import pandas
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class TransformerModel(nn.Module):
    def __init__(
        self,
        feature_dim,
        position_dim,
        num_classes,
        d_model=512,
        nhead=2,
        num_layers=2,
    ):
        super(TransformerModel, self).__init__()
        self.feature_embedding = nn.Linear(feature_dim, d_model)
        self.position_embedding = nn.Linear(position_dim, 4)
        self.transformer = nn.Transformer(
            d_model=d_model + 4,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=0
        )
        self.fc = nn.Linear(d_model + 4, num_classes)

    def forward(self, src_features, src_positions):
        feature_emb = self.feature_embedding(src_features)
        position_emb = self.position_embedding(src_positions)
        src = torch.cat((feature_emb, position_emb), dim=1)  # Concatenate embeddings
        src = src.unsqueeze(0)  # Add sequence length dimension
        output = self.transformer(src, src)
        output = output.squeeze(0)
        output = self.fc(output)
        # output = torch.sum(output, dim=0)
        # output = output / torch.sum(output)
        return torch.sigmoid(output)


def m():
    features = np.load(
        "/mnt/big/indeed/legasea/annflux/datarepo/resultset-20250509071930-4e24a4d1/last_full.npz"
    )["lastFull"]
    patch_data = pandas.read_csv("/mnt/big/indeed/legasea/annflux/annflux.csv")
    print(patch_data.columns)
    print(patch_data.label_predicted.unique())
    print(Counter(patch_data.label_original))
    # exit(0)
    assert len(features) == len(patch_data)

    grouped_features = []
    grouped_positions = []
    grouped_labels = []
    patch_data = patch_data[patch_data.label_original.isin(["kies", "dijbeen"])]
    patch_data = patch_data[
        (patch_data.label_predicted != "Empty")
        & (~pandas.isna(patch_data.label_predicted))
    ]
    # features = features[patch_data.index.to_numpy()]
    # patch_data.reset_index(drop=True, inplace=True)
    unique_org_labels = patch_data.label_original.unique()
    print(f"{len(unique_org_labels)=}")
    label_org_to_index = dict(zip(unique_org_labels, range(len(unique_org_labels))))
    # TODO: use predicted patch labels
    for original_image_id in patch_data.original_image_id.unique():
        results_for_image = patch_data[
            (patch_data.original_image_id == original_image_id)
        ]
        # print(len(results_for_image))
        grouped_features.append(
            torch.from_numpy(
                features[results_for_image.index.to_numpy()].astype(np.float32)
            )
        )
        grouped_positions.append(
            torch.from_numpy(
                results_for_image[["patch_x", "patch_y"]].values.astype(np.float32) / 3000.
            )
            # / 5000
        )
        patch_labels = []
        for _, row in results_for_image.iterrows():
            one_hot = np.zeros((1, len(unique_org_labels)), dtype=int)
            one_hot[0, label_org_to_index[row.label_original]] = 1
            patch_labels.append(one_hot)
        patch_labels = torch.from_numpy(np.vstack(patch_labels))
        # print(summarize(patch_labels))
        # one_hot = np.zeros((len(unique_org_labels)), dtype=int)
        # one_hot[label_org_to_index[results_for_image.label_original.unique()[0]]] = 1
        # print(one_hot.shape)
        # grouped_labels.append(torch.from_numpy(one_hot))
        grouped_labels.append(patch_labels)
        # grouped_labels.append(results_for_image[["patch_x", "patch_y"]].values.tolist())
        # print(grouped_positions[-1])

    # itertools.chain.from_iterable()
    sanity = False
    if sanity:
        X = np.vstack(list(itertools.chain.from_iterable(grouped_features)))
        y = np.argmax(np.vstack(list(itertools.chain.from_iterable(grouped_labels))), axis=1)
        X_train, X_test, y_train, y_test = train_test_split(
            features[patch_data.index.tolist()],patch_data.label_original , test_size=0.33, random_state=42
        )

        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        print(rf.score(X_test, y_test))
    print([int(summarize(x_).numpy()) for x_ in grouped_labels])
    print(Counter([int(summarize(x_).numpy()) for x_ in grouped_labels]))

    group_labels = np.array([int(summarize(x_).numpy()) for x_ in grouped_labels])
    counts = Counter(group_labels)
    print(len(counts))
    p = np.zeros((len(counts),))
    print(p)
    for k, v in counts.items():
        print(k, v)
        p[k] = v
    print(p)
    p = p.astype(float) / p.sum()

    print(grouped_features[-1].dtype)
    print(grouped_positions[-1].dtype)
    print(grouped_labels[-1].dtype)
    # Create a dataset for each group
    grouped_datasets = [
        TensorDataset(features, positions, labels)
        for features, positions, labels in zip(
            grouped_features, grouped_positions, grouped_labels
        )
    ]
    grouped_dataloaders = [
        DataLoader(dataset, batch_size=256, shuffle=False) for dataset in grouped_datasets
    ]

    # Step 2: Model Definition

    model = TransformerModel(
        feature_dim=features.shape[1],
        position_dim=2,
        num_classes=len(unique_org_labels),
    )
    model = model

    # Step 3: Training Loop
    criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    emp_counts = defaultdict(lambda: 0)
    for epoch in range(100):  # Number of epochs
        model.train()
        for group_idx, dataloader in enumerate(grouped_dataloaders):
            group_label = group_labels[group_idx]
            if np.random.rand() < p[group_label]:
                continue
            emp_counts[group_label] += 1
            # print(f"{emp_counts.items()=}")
            for batch_features, batch_positions, batch_labels in dataloader:
                outputs = model(batch_features, batch_positions)
                loss = criterion(outputs, batch_labels.float())
                loss = loss / len(grouped_dataloaders)
                loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch {epoch + 1}, Group {group_idx + 1}, Loss: {loss.item()}")

        # Step 4: Evaluation
        model.eval()
        with torch.no_grad():
            for group_idx in range(8):
                test_features = grouped_features[
                    group_idx
                ]  # Example test features for the group
                test_positions = grouped_positions[
                    group_idx
                ]  # Example test positions for the group
                predictions = model(test_features, test_positions)
                print(
                    f"Group {group_idx + 1} Predictions: {summarize(predictions)} {summarize(predictions, no_argmax=True)}, {group_labels[group_idx]}"
                )


def summarize(predictions, no_argmax=False):
    result = torch.sum(predictions, dim=0, keepdim=True)
    return torch.argmax(result) if not no_argmax else result


if __name__ == "__main__":
    m()
