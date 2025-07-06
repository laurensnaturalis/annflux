import pickle


from annflux.training.annflux.group_classifier import summarize


def m():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    # Step 1: Data Preparation
    # Assume `grouped_features` is a list of tensors, each of shape (num_samples_in_group, num_features)
    # `grouped_positions` is a list of tensors, each of shape (num_samples_in_group, spatial_dimensions)
    # `grouped_labels` is a list of tensors, each of shape (num_samples_in_group, num_classes)
    grouped_features = [
        torch.randn(10, 7) for _ in range(5)
    ]  # Example grouped features
    grouped_positions = [
        torch.randn(10, 3) for _ in range(5)
    ]  # Example grouped spatial positions
    grouped_labels = [
        torch.randint(0, 2, (10, 5)) for _ in range(5)
    ]  # Example grouped multi-label targets

    # Create a dataset for each group
    grouped_features, grouped_positions, grouped_labels = pickle.load(
        open("bla.pickle", "rb")
    )
    import numpy as np

    group_labels = np.array([int(summarize(x_).numpy()) for x_ in grouped_labels])
    indices0 = np.random.choice(np.where(group_labels == 0)[0], size=14, replace=False)
    indices1 = np.random.choice(np.where(group_labels == 1)[0], size=14, replace=False)

    selected = set(sorted(indices0.tolist() + indices1.tolist()))
    grouped_features = [
        x_
        for i_, x_ in enumerate(grouped_features)
        if i_ in selected
    ]
    grouped_positions = [
        x_ / 3000.
        for i_, x_ in enumerate(grouped_positions)
        if i_ in selected
    ]
    grouped_labels = [
        x_
        for i_, x_ in enumerate(grouped_labels)
        if i_ in selected
    ]
    group_labels = np.array([int(summarize(x_).numpy()) for x_ in grouped_labels])
    # f_train, f_test, p_train, p_test, l_train, l_test = train_test_split(
    #     grouped_features,
    #     grouped_positions,
    #     grouped_labels,
    #     test_size=0.33,
    #     random_state=42,
    #     stratify=
    # )

    grouped_datasets = [
        TensorDataset(features, positions, labels)
        for features, positions, labels in zip(
            grouped_features, grouped_positions, grouped_labels
        )
    ]
    grouped_dataloaders = [
        DataLoader(dataset, batch_size=256, shuffle=True) for dataset in grouped_datasets
    ]

    # Step 2: Model Definition
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
            self.position_embedding = nn.Linear(position_dim, 2)
            self.transformer = nn.Transformer(
                d_model=d_model + 2,  # Double the dimension because of concatenation
                nhead=nhead,
                num_encoder_layers=num_layers,
                num_decoder_layers=num_layers,
                dropout=0
            )
            self.fc = nn.Linear(d_model + 2, num_classes)

        def forward(self, src_features, src_positions):
            feature_emb = self.feature_embedding(src_features)
            position_emb = self.position_embedding(src_positions)
            src = torch.cat(
                (feature_emb, position_emb), dim=1
            )  # Concatenate embeddings
            src = src.unsqueeze(0)  # Add sequence length dimension
            output = self.transformer(src, src)
            output = output.squeeze(0)
            output = self.fc(output)
            return torch.sigmoid(output)

    model = TransformerModel(feature_dim=512, position_dim=2, num_classes=2)

    # Step 3: Training Loop
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # with torch.set_grad_enabled(True):

    for epoch in range(100):  # Number of epochs
        model.train()
        for group_idx, dataloader in enumerate(grouped_dataloaders):
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
                print(f"Group {group_idx + 1} Predictions: {summarize(predictions)} {summarize(predictions, no_argmax=True)}, {group_labels[group_idx]}")



if __name__ == "__main__":
    m()
