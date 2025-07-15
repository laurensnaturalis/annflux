import os
import itertools
import sys
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
from numpy._typing import NDArray
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from annflux.repository.resultset import Resultset
from annflux.shared import AnnfluxSource
from annflux.tools.io import basename_no_extension


def labels_to_matrix(all_labels, name_to_index, t, true_labels):
    true_features = np.zeros((len(t), len(all_labels)))
    for i, labels in enumerate(true_labels):
        for label in labels:
            true_features[i, name_to_index[label]] = 1
    return true_features


def classify_path(
    features_path, annflux_path, feature_image_out_folder, group_data_path
) -> (np.array, float, pandas.DataFrame):
    features = np.load(features_path)["lastFull"]
    patch_data = pandas.read_csv(annflux_path)
    os.makedirs(feature_image_out_folder, exist_ok=True)
    if not os.path.exists(group_data_path):
        uids = [
            basename_no_extension(x_)
            for x_ in os.listdir("/mnt/big/indeed/legasea_big/images_group")
        ]  # TODO
        label_true = [
            None,
        ] * len(uids)
        t = pandas.DataFrame({"uid": uids, "label_true": label_true})
        t.to_csv(group_data_path, index=False)
    group_data = pandas.read_csv(group_data_path)
    record_to_label = dict(zip(group_data.uid, group_data.label_true))
    return classify(features, patch_data, feature_image_out_folder, record_to_label, num_epochs=30)


def make_feature_image_advanced(features, image_dim, record_instance_data, min_value):
    # tmp = record_instance_data.copy().sort_values(by="minute", inplace=True)
    taken: set[int] = set()
    patch = np.ones((image_dim, image_dim)) * min_value
    patch_i = 0  # int(row.minute / 60 * 223) # TODO: this will run over the boundary of another 10-minute section
    for r, row in record_instance_data.iterrows():
        while patch_i in taken and patch_i < 223:
            patch_i += 1
        taken.add(patch_i)
        patch[patch_i, :] = features[r, :]

    # plt.imshow(patch[:224, :])
    # print(record_instance_data[["label_predicted", "label_true", "minute"]])
    # plt.show()

    return patch


def classify(
    features,
    instance_data: pandas.DataFrame,
    feature_image_out_folder,
    record_to_label: dict[str, str],
    num_epochs=10,
) -> (np.array, float, pandas.DataFrame):
    """
    :return features, accuracy, out_table
    """
    assert len(features) == len(instance_data), (len(features), len(instance_data))
    # if "label_original" not in instance_data.columns:
    #     return
    # # TODO: check if there's fewer groups than images, otherwise skip step

    grouped_labels = []
    grouped_images = []
    group_ids = []
    # TODO: use predicted patch labels
    if "hour" in instance_data.columns:  # camera trap / ecology specific
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
    num_components = min(len(set(itertools.chain.from_iterable(true_labels))), 12)
    true_label_matrix = PCA(n_components=num_components).fit_transform(
        labels_to_matrix(all_labels, name_to_index, instance_data, true_labels)
    )
    predicted_label_matrix = PCA(n_components=num_components).fit_transform(
        labels_to_matrix(all_labels, name_to_index, instance_data, predicted_labels)
    )

    image_dim = 224

    feature_reduced_size = image_dim - 3 * num_components
    pca = PCA(n_components=feature_reduced_size)
    pca.fit(features)

    principal_features = pca.transform(features)

    all_features = np.zeros((principal_features.shape[0], image_dim))
    all_features[:, :feature_reduced_size] = np.log(
        principal_features - principal_features.min() + 1
    )
    all_features[:, feature_reduced_size : feature_reduced_size + num_components] = (
        true_label_matrix - predicted_label_matrix
    )
    all_features[
        :,
        feature_reduced_size + num_components : feature_reduced_size
        + 2 * num_components,
    ] = true_label_matrix
    all_features[:, feature_reduced_size + 2 * num_components :] = (
        predicted_label_matrix
    )

    for record_id in tqdm(instance_data.record_id.unique()):
        group_ids.append(record_id)
        feature_image_path = os.path.join(
            feature_image_out_folder,
            f"{record_id}.jpg",
        )
        if not os.path.exists(feature_image_path):
            # feature_image, results_for_image = make_feature_image_basic(features, image_dim, original_image_id, patch_data)
            results_for_record = instance_data[instance_data.record_id == record_id]
            # print(f"{len(results_for_record)=}")
            feature_image = make_feature_image_advanced(
                all_features, image_dim, results_for_record, all_features.min()
            )

            # feature_image = np.repeat(np.expand_dims(feature_image, axis=2), 3, axis=2)

            # print(feature_image.shape)
            feature_image_for_display = feature_image.copy()
            feature_image_for_display = normalize(feature_image_for_display)

            feature_image_red = feature_image_for_display.copy()
            plt.imshow(feature_image_red)
            # plt.show()

            feature_image_red = normalize(feature_image_red)
            feature_image_red[:, feature_reduced_size:] = feature_image_red.min()
            # feature_image_red
            plt.imshow(feature_image_red)
            # plt.show()
            feature_image_green = feature_image_for_display.copy()
            feature_image_green[:, :feature_reduced_size] = feature_image_green.min()
            feature_image_green[:, feature_reduced_size + 24 :] = (
                feature_image_green.min()
            )
            plt.imshow(feature_image_green)
            # plt.show()
            feature_image_blue = feature_image_for_display.copy()
            feature_image_blue[:, :feature_reduced_size] = feature_image_blue.min()
            feature_image_blue[
                :, feature_reduced_size + 12 : feature_reduced_size + 24 :
            ] = feature_image_blue.min()
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
                feature_image_path,
                (feature_image_for_display * 255).astype(np.uint8),
                pil_kwargs={"quality": 99},
            )
            feature_image = feature_image_for_display
        else:
            feature_image = plt.imread(feature_image_path)
        assert feature_image.shape == (image_dim, image_dim, 3), feature_image.shape

        grouped_labels.append(record_to_label[record_id])
        grouped_images.append(feature_image.astype(np.float32))

    tmp = np.vstack(grouped_images)
    min_val = tmp.min()
    max_val = tmp.max()
    grouped_images = [(x_ - min_val) / max_val for x_ in grouped_images]

    grouped_label_to_index = dict(
        zip(set(grouped_labels), range(len(set(grouped_labels))))
    )
    index_to_grouped_label = {val: key for key, val in grouped_label_to_index.items()}
    features, accuracy, predictions = train_pytorch(
        grouped_images,
        [grouped_label_to_index[x_] for x_ in grouped_labels],
        num_epochs=num_epochs,
    )
    probabilities = np.max(predictions, axis=-1)

    out_table = pandas.DataFrame(
        {
            "uid": group_ids,
            "label_predicted": [
                index_to_grouped_label[i_] for i_ in np.argmax(predictions, axis=1)
            ],
            "label_true": grouped_labels,
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
    images, labels: list[int], num_epochs=10, min_number_of_examples=4
) -> (NDArray, float, NDArray):
    """
    return features, accuracy, probability vectors
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
    print(f"{len(images)} images")
    X = np.stack(images)
    y = np.array(labels)
    print(Counter(labels))
    ignore = [
        t_[0] for t_ in Counter(labels).most_common() if t_[1] < min_number_of_examples
    ]
    sel_ = [x_ not in set(ignore) for x_ in y]
    print(f"{len(sel_)=}, {sel_[:10]}")
    X_sufficient_labeled = X[sel_]
    y_sufficient_labeled = y[sel_]
    print(f"{len(X_sufficient_labeled)=}, {len(y_sufficient_labeled)=}")
    X_train, X_test, y_train, y_test = train_test_split(
        X_sufficient_labeled,
        y_sufficient_labeled,
        test_size=0.33,
        random_state=42,
        stratify=np.array(labels)[sel_],
    )

    train_loader = DataLoader(
        InMemoryDataset(X_train, y_train, transform=transform),
        batch_size=32,
        shuffle=True,
    )

    model = models.efficientnet_b0(pretrained=True)

    num_classes = len(set(labels))
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

    _, predictions_test = predict(device, model, test_loader)

    all_loader = DataLoader(
        InMemoryDataset(X, y, transform=transform),
        batch_size=32,
        shuffle=False,
    )
    group_features, predictions_all = predict(device, model, all_loader)

    return (
        np.vstack(group_features),
        accuracy_score(y_test, np.argmax(predictions_test, axis=-1)),
        np.vstack(predictions_all),
    )


def predict(device, model, loader) -> (np.array, list[NDArray]):
    """
    :return group_features, probability vectors
    """
    model.eval()
    predictions = []
    group_features = []
    for inputs, labels in loader:
        with torch.no_grad():
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            outputs = outputs.cpu().numpy()

            predictions.extend(outputs)

            # TODO: predictions and features at the same time
            feature_vector = model.features(inputs).cpu().numpy().mean(axis=(2, 3))
            group_features.extend(feature_vector)
    return group_features, predictions


if __name__ == "__main__":
    source_ = AnnfluxSource(sys.argv[1])
    resultset: Resultset = source_.repository.get(label=Resultset).last()
    features, accuracy, out_table = classify_path(
        features_path=resultset.last_full_path,
        annflux_path=source_.data_state_path,
        feature_image_out_folder=os.path.join(source_.folder, "group_feature_images"),
        group_data_path=source_.group_flux_data_path(),
    )
    print(f"{accuracy=}")
    np.savez(source_.group_features_path(), lastFull=features)
    out_table.to_csv(source_.group_flux_data_path())
