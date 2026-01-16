import os
from typing import Any

import pandas
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame


def m():
    accuracy_avg, recall_avg, t = compute_averages("experiments/papbig_knnexp_5_step_500_strategy_fre_strat_linear_at_(0, 5, 10, 15)")
    # accuracy_avg2, recall_avg2, t = compute_averages("experiments/step_500_strategy_fre_strat_linear_at_1")
    # accuracy_avg3, recall_avg3, t = compute_averages("experiments/step_500_strategy_fre_strat_linear_at_(1, 5, 10)")
    # accuracy_avg4, recall_avg4, t = compute_averages("experiments/papbig_step_500_strategy_fre_strat_linear_at_(0, 5, 10, 15)")
    plt.subplot(221)
    plt.plot(t.labeled_train_set_size, accuracy_avg, label="linear at (0, 5, 10, 15)")
    # plt.plot(t.labeled_train_set_size, accuracy_avg2, label="linear at 1")
    # plt.plot(t.labeled_train_set_size, accuracy_avg3, label="linear at (1, 5, 10)")
    # plt.plot(t.labeled_train_set_size, accuracy_avg4, label="linear at (1, 5, 10)")
    # plt.plot(t2.labeled_train_set_size, t2.accuracy, label="step - linear at 10 - frestrat 2")
    # plt.plot(t3.labeled_train_set_size, t3.accuracy, label="step - linear at 10 - frestrat")
    # plt.plot(t4.labeled_train_set_size, t4.accuracy, label="step - linear at 10 - frestrat")
    # plt.plot(t4.accuracy, label="frestrat - linear")
    plt.legend()

    plt.subplot(222)
    plt.plot(t.labeled_train_set_size, recall_avg, label="Recall")
    # plt.plot(t.labeled_train_set_size, recall_avg2, label="Recall")
    # plt.plot(t.labeled_train_set_size, recall_avg3  , label="Recall")
    # plt.plot(t.labeled_train_set_size, recall_avg4  , label="Recall")
    # plt.plot(t2.labeled_train_set_size, t2.recall, label="Recall")
    # plt.plot(t3.labeled_train_set_size, t3.recall, label="Recall")
    # plt.plot(t4.labeled_train_set_size, t4.recall, label="Recall")

    plt.legend()
    plt.show()


def compute_averages(path) -> tuple[DataFrame, Any, Any]:
    recalls = []
    accuracies = []
    for seed in range(42, 42 + 5):
        path = f"{path}_seed={seed}.csv"
        if os.path.exists(path):
            t = pandas.read_csv(path)
            recalls.append(t.recall)
            accuracies.append(t.accuracy)

    recall_avg = np.mean(np.vstack(recalls), axis=0)
    accuracy_avg = np.mean(np.vstack(accuracies), axis=0)
    return accuracy_avg, recall_avg, t


if __name__ == '__main__':
    m()