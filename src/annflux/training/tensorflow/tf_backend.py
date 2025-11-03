# Copyright 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import math
import os
from collections import defaultdict, Counter
from typing import Dict, List

import keras.src.callbacks
import numpy as np
import pandas
from keras import Input, Model
from keras.src.callbacks import (
    ReduceLROnPlateau,
    ModelCheckpoint,
    EarlyStopping,
    Callback,
)
from keras.src.layers import Dense, Lambda
from keras.src.legacy.backend import l2_normalize
from keras.src.optimizers import Adam

from keras.src.trainers.data_adapters.py_dataset_adapter import PyDataset
from numpy._typing import NDArray

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from annflux.tools.core import AnnFluxState

logger = logging.getLogger("annflux_server")


def linear_retraining_logic(state: AnnFluxState, status_callback):
    labeled_indices = state.labeled_indices
    if labeled_indices is None or len(labeled_indices) == 0:
        return
    logger.info(f"{len(labeled_indices)=}")
    balance = True
    test_labels = state.label_array_test[state.labeled_test_indices]
    train_val_labels = state.label_array[labeled_indices]
    no_label_for_labeled_idx = np.where(
        train_val_labels == None  # noqa
    )[0]
    if len(no_label_for_labeled_idx) > 0:
        raise RuntimeError(
            f"no label for idx {np.array(labeled_indices)[no_label_for_labeled_idx]}"
        )

    features = state.features
    train_val_features = features[labeled_indices]
    weights_path = os.path.join(state.annflux_folder, "linear.weights.h5")
    test_features = features[state.labeled_test_indices]

    model2 = linear_train_func(
        train_val_features,
        test_features,
        train_val_labels,
        test_labels,
        weights_path,
        balance,
        status_callback,
    )

    state.g_quick_status = "recomputing features"
    state.features = model2.predict(features)

    return weights_path


def linear_train_func(
    train_val_features: NDArray,
    test_features: NDArray,
    train_val_labels: List[List[str]],
    test_labels: List[List[str]],
    weights_path: str,
    balance: bool,
    status_callback: keras.src.callbacks.Callback,
) -> Model:
    binarizer = MultiLabelBinarizer()

    binarizer.fit(train_val_labels)
    targets = binarizer.transform(train_val_labels)
    print(f"{train_val_labels=}")
    print(f"{len(binarizer.classes_)=}")

    test_targets = binarizer.transform(test_labels)

    x_train, x_val, y_train, y_val = train_test_split(
        train_val_features, targets, test_size=0.10, random_state=42
    )
    # make the linear model
    features_size = train_val_features.shape[1]
    input_ = Input(shape=(features_size,))
    dense = input_
    features_ = Dense(features_size, activation="relu", name="features")(dense)
    features_ = Lambda(lambda x: l2_normalize(x, axis=1))(features_)
    predictions = Dense(len(binarizer.classes_), activation="sigmoid")(features_)
    # for predictions, to train
    model = Model(inputs=[input_], outputs=[predictions])
    # to compute features
    model2 = Model(inputs=[input_], outputs=[features_])
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=0.000001, verbose=1
    )
    model.summary()

    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.1), metrics=["accuracy"])

    checkpointer = ModelCheckpoint(
        weights_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=True,
    )

    model.fit(
        x=BalanceSequence(x_train, y_train, 1024, balance=balance),
        batch_size=1024,
        validation_data=(x_val, y_val),
        epochs=200,
        verbose=1,
        callbacks=[
            reduce_lr,
            checkpointer,
            EarlyStopping(patience=5),
            status_callback,
        ],
    )
    model.load_weights(weights_path)
    test_predictions = model.predict(test_features)
    acc_test = accuracy_score(test_targets, (test_predictions > 0.5).astype(int))
    print(f"linear from features accuracy = {acc_test}")
    return model2


class BalanceSequence(PyDataset):
    def __init__(self, x_set, y_set, batch_size, balance: bool = False):  # noqa
        self.x, self.y = np.array(x_set), np.array(y_set)
        class_counts = Counter(np.argmax(self.y, axis=1))
        self.classes_ = list(class_counts.keys())
        if balance:
            self.class_weights = None
        else:
            self.class_weights = np.array(
                [class_counts[x_] for x_ in self.classes_], dtype=float
            )
            self.class_weights /= self.class_weights.sum()
        self.class_to_indices: Dict[int, List[int]] = defaultdict(lambda: [])
        for i_, class_ in enumerate(np.argmax(self.y, axis=1)):
            self.class_to_indices[class_].append(i_)
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        indices_batch = []
        for _ in range(self.batch_size):
            class_ = np.random.choice(self.classes_, p=self.class_weights)
            indices_batch.append(np.random.choice(self.class_to_indices[class_]))

        # print(type(self.x[indices_batch]))
        return self.x[indices_batch], self.y[indices_batch]



class DummyCallBack(Callback):
    def __init__(self):
        super().__init__()

    def __call__(self, epoch, logs=None):
        print(epoch)

    def on_epoch_end(self, epoch, logs=None):
        print(epoch)


def _test_linear_train():
    features = np.load("/mnt/big/indeed/lepisea/annflux/datarepo/resultset-20251030124355-7a3d7e2f/last_full.npz")["lastFull"]
    data = pandas.read_csv("/mnt/big/indeed/lepisea/annflux/annflux.csv")
    test_indices = np.where(data["in_test"]==1)[0]
    train_val_indices = np.where(data["in_test"]==0)[0]

    labeled_indices = np.where(data["labeled"]==1)[0]
    train_val_indices = sorted(list(set(train_val_indices).intersection(set(labeled_indices))))
    test_indices = sorted(list(set(test_indices).intersection(set(labeled_indices))))

    train_val_features = features[train_val_indices]
    test_features = features[test_indices]
    label_true = data["label_true"]
    train_val_labels = [[y_ for y_ in x_.split(",") if y_.endswith("ae") and len(y_.split()) == 1] for x_ in label_true[train_val_indices]]
    test_labels = [[y_ for y_ in x_.split(",") if y_.endswith("ae") and len(y_.split()) == 1] for x_ in label_true[test_indices]]
    linear_train_func(train_val_features, test_features, train_val_labels, test_labels, "tmp.weights.h5", False, DummyCallBack())


if __name__ == '__main__':
    _test_linear_train()