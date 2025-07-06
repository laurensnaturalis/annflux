import os
import itertools
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def labels_to_matrix(all_labels, name_to_index, t, true_labels):
    true_features = np.zeros((len(t), len(all_labels)))
    for i, labels in enumerate(true_labels):
        for label in labels:
            true_features[i, name_to_index[label]] = 1
    return true_features


def classify_path(features_path, annflux_path) -> (np.array, float, pandas.DataFrame):
    features = np.load(features_path)["lastFull"]
    patch_data = pandas.read_csv(annflux_path)
    classify(features, patch_data)


def make_feature_image_advanced(features, image_dim, record_instance_data, min_):
    # tmp = record_instance_data.copy().sort_values(by="minute", inplace=True)
    taken: set[int] = set()
    patch = np.ones((image_dim, image_dim)) * min_
    for r, row in record_instance_data.iterrows():
        patch_i = int(row.minute / 60 * 223)
        while patch_i in taken and patch_i < 223:
            patch_i += (
                1  # TODO: this will run over the boundary of another 10-minute section
            )
        taken.add(patch_i)
        patch[patch_i, :] = features[r, :]

    # plt.imshow(patch[:224, :])
    # print(record_instance_data[["label_predicted", "label_true", "minute"]])
    # plt.show()

    return patch


def classify(features, instance_data: pandas.DataFrame) -> (np.array, float, list[str]):
    assert len(features) == len(instance_data), (len(features), len(instance_data))
    # if "label_original" not in instance_data.columns:
    #     return
    # # TODO: check if there's fewer groups than images, otherwise skip step

    grouped_labels = []
    grouped_images = []
    group_ids = []
    # unique_org_labels = instance_data.label_original.unique()
    # print(f"{len(unique_org_labels)=}")
    # print(f"{unique_org_labels=}")
    # label_org_to_index = dict(zip(unique_org_labels, range(len(unique_org_labels))))
    # TODO: use predicted patch labels
    instance_data.dropna(subset="hour", inplace=True)
    grouper = ["year", "month", "day", "hour"]
    for g in grouper:
        instance_data[g] = instance_data[g].astype(int)

    instance_data["datetime"] = instance_data[grouper].apply(
        lambda t_: datetime(*t_), axis=1
    )
    instance_data.sort_values(by="datetime", inplace=True)

    predicted_labels, predicted_labels_set = get_labels(
        instance_data, "label_predicted"
    )
    # TODO: tmp hack
    instance_data.label_true = instance_data.label_true.fillna("")
    print(np.where(pandas.isna(instance_data.label_true)))
    # end hack
    true_labels, true_labels_set = get_labels(instance_data, "label_true")
    all_labels = list(predicted_labels_set | true_labels_set)
    name_to_index = dict(zip(all_labels, range(len(all_labels))))

    #
    true_label_matrix = PCA(n_components=12).fit_transform(
        labels_to_matrix(all_labels, name_to_index, instance_data, true_labels)
    )
    predicted_label_matrix = PCA(n_components=12).fit_transform(
        labels_to_matrix(all_labels, name_to_index, instance_data, predicted_labels)
    )

    P = 224 - 3 * 12
    pca = PCA(n_components=P)
    pca.fit(features)

    principal_features = pca.transform(features)

    all_features = np.zeros((principal_features.shape[0], 224))
    all_features[:, :P] = np.log(principal_features - principal_features.min() + 1)
    all_features[:, P : P + 12] = true_label_matrix - predicted_label_matrix
    all_features[:, P + 12 : P + 24] = true_label_matrix
    all_features[:, P + 24 :] = predicted_label_matrix

    label_org_to_index = dict(zip(list(true_labels_set), range(len(true_labels_set))))

    print(instance_data.datetime.min(), instance_data.datetime.max())
    for record_id in instance_data.record_id.unique():
        group_ids.append(record_id)
        image_dim = 224
        # feature_image, results_for_image = make_feature_image_basic(features, image_dim, original_image_id, patch_data)
        results_for_record = instance_data[instance_data.record_id == record_id]
        feature_image = make_feature_image_advanced(
            all_features, image_dim, results_for_record, all_features.min()
        )

        # feature_image = np.repeat(np.expand_dims(feature_image, axis=2), 3, axis=2)

        print(feature_image.shape)
        feature_image_for_display = feature_image.copy()
        feature_image_for_display = normalize(feature_image_for_display)

        feature_image_red = feature_image_for_display.copy()
        plt.imshow(feature_image_red)
        # plt.show()

        feature_image_red = normalize(feature_image_red)
        feature_image_red[:, P:] = feature_image_red.min()
        # feature_image_red
        plt.imshow(feature_image_red)
        # plt.show()
        feature_image_green = feature_image_for_display.copy()
        feature_image_green[:, :P] = feature_image_green.min()
        feature_image_green[:, P + 24 :] = feature_image_green.min()
        plt.imshow(feature_image_green)
        # plt.show()
        feature_image_blue = feature_image_for_display.copy()
        feature_image_blue[:, :P] = feature_image_blue.min()
        feature_image_blue[:, P + 12 : P + 24 :] = feature_image_blue.min()
        plt.imshow(feature_image_blue)
        # plt.show()
        feature_image_for_display = np.stack(
            [feature_image_red, feature_image_green, feature_image_blue], axis=-1
        )
        feature_image_for_display -= feature_image_for_display.min()
        feature_image_for_display /= feature_image_for_display.max()
        plt.imshow(feature_image_for_display)
        # plt.show()
        plt.imsave(
            os.path.join(
                "/home/laurens/Documents/data/ami_oh2_hour/images_group",  # TODO:
                f"{record_id}.jpg",
            ),
            (feature_image_for_display * 255).astype(np.uint8),
            pil_kwargs={"quality": 99},
        )
        feature_image = feature_image_for_display
        assert feature_image.shape == (image_dim, image_dim, 3), feature_image.shape

        grouped_labels.append(
            label_org_to_index[results_for_record.label_true.values[0]]
        )  # TODO: assert one value
        grouped_images.append(feature_image.astype(np.float32))

    tmp = np.vstack(grouped_images)
    min_val = tmp.min()
    max_val = tmp.max()
    grouped_images = [(x_ - min_val) / max_val for x_ in grouped_images]

    index_to_label = {value: key for key, value in label_org_to_index.items()}
    features, accuracy, predicted_labels, true_labels, probabilities = train_pytorch(
        grouped_images, grouped_labels, index_to_label, num_epochs=2
    )
    print(len(group_ids), len(predicted_labels), len(true_labels), len(probabilities))
    out_table = pandas.DataFrame(
        {
            "group_id": group_ids,
            "label_predicted": predicted_labels,
            "label_true": true_labels,
            "score_predicted": probabilities,
        }
    )
    return features, accuracy, out_table


def normalize(feature_image_for_display):
    min_ = feature_image_for_display.min()  # np.sort(feature_image_for_display[:])[1]
    feature_image_for_display -= min_
    feature_image_for_display /= feature_image_for_display.max()
    return feature_image_for_display


def get_labels(instance_data, label_field) -> (list[str], set[str]):
    labels = [
        x_.split(",")
        if not pandas.isna(x_)
        else [
            None,
        ]
        for x_ in instance_data[label_field]
    ]
    labels_set = set(itertools.chain(*labels))
    return labels, labels_set


def make_feature_image_basic(features, image_dim, original_image_id, patch_data):
    feature_image = np.zeros(shape=(image_dim, image_dim))
    results_for_image = patch_data[(patch_data.record_id == original_image_id)]
    features_from_parts = features[results_for_image.index.to_numpy()].astype(
        np.float32
    )
    feature_image[: features_from_parts.shape[0], :] = features_from_parts[
        :, :image_dim
    ]
    feature_image[-features_from_parts.shape[0] :, :] = features_from_parts[
        :, image_dim : image_dim * 2
    ]
    return feature_image, results_for_image


class InMemoryDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def train_pytorch(
    images, labels: list[int], index_to_label, num_epochs=10
) -> [np.array, float, list[str], list[str], list[float]]:
    """
    return features, accuracy, predicted labels, true labels, probabilities
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import models, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    X = images
    y = labels
    print(index_to_label)
    print([index_to_label[x_] for _, x_ in Counter(labels).items()])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify=labels
    )

    train_loader = DataLoader(
        InMemoryDataset(X_train, y_train, transform=transform),
        batch_size=32,
        shuffle=True,
    )

    model = models.efficientnet_b0(pretrained=True)

    num_classes = len(set(labels))  # Change this to your number of classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    print("Training Finished")

    test_loader = DataLoader(
        InMemoryDataset(X_test, y_test, transform=transform),
        batch_size=32,
        shuffle=False,
    )

    _, predictions_test, _ = predict(device, index_to_label, model, test_loader)

    all_loader = DataLoader(
        InMemoryDataset(X, y, transform=transform),
        batch_size=32,
        shuffle=False,
    )
    group_features, predictions_all, probs = predict(
        device, index_to_label, model, all_loader
    )

    return (
        np.vstack(group_features),
        accuracy_score(y_test, predictions_test),
        [index_to_label[x_] for x_ in predictions_all],
        [index_to_label[int(x_)] for x_ in y],
        probs,
    )


def predict(device, index_to_label, model, loader):
    model.eval()
    predictions = []
    probs = []
    group_features = []
    n = 0
    for inputs, labels in loader:
        with torch.no_grad():
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            outputs = outputs.cpu().numpy()

            predictions.extend(np.argmax(outputs, axis=1).tolist())
            print(np.max(outputs, axis=1).shape)
            probs.extend(np.max(outputs, axis=1).tolist())
            # print(
            #     list(
            #         zip(
            #             [index_to_label[x_] for x_ in np.argmax(outputs, axis=1)],
            #             [index_to_label[x_] for x_ in labels.cpu().numpy()],
            #         )
            #     )
            # )
            # TODO: predictions and features at the same time
            feature_vector = model.features(inputs).cpu().numpy().mean(axis=(2, 3))
            group_features.extend(feature_vector)
            n += len(feature_vector)
            print(f"{n=}, {len(probs)=}")
    return group_features, predictions, probs


if __name__ == "__main__":
    classify_path(
        features_path="/home/laurens/Documents/data/ami_oh2_hour/annflux/datarepo/resultset-20250527114436-334e2d26/last_full.npz",
        annflux_path="/home/laurens/Documents/data/ami_oh2_hour/annflux/annflux.csv",
    )
    # classify_path(
    #     features_path="/mnt/big/indeed/legasea/annflux/datarepo/resultset-20250509071930-4e24a4d1/last_full.npz",
    #     annflux_path="/mnt/big/indeed/legasea/annflux/annflux.csv",
    # )
