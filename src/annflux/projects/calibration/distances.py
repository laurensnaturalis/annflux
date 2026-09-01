import pandas
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


def m():
    t = pandas.read_csv("/mnt/big/indeed/bombus-plant-intel/annflux/annflux.csv")

    t_labeled = t[(t.labeled == 1) & (~pandas.isna(t.min_distance))]
    t_labeled["correct"] = t.apply(
        lambda row_: row_.label_predicted == row_.label_true, axis=1
    )
    t_labeled["inv_min_distance"] = (
        t_labeled["min_distance"].max() - t_labeled["min_distance"]
    )

    print(len(t_labeled))

    print(
        t_labeled[
            [
                "min_distance",
                "label_predicted",
                "label_true",
                "correct",
                "inv_min_distance",
            ]
        ].sort_values("min_distance")
    )

    fpr, tpr, thresholds = roc_curve(
        t_labeled.correct, t_labeled.min_distance.max() - t_labeled.min_distance
    )

    print(
        roc_auc_score(
            t_labeled.correct, t_labeled.min_distance.max() - t_labeled.min_distance
        )
    )
    import numpy as np
    hist, edges = np.histogram(t_labeled.min_distance, bins="auto")
    print(hist, edges)
    plt.scatter(t_labeled.min_distance, t_labeled.correct)
    plt.show()

    # plt.plot(fpr, tpr, label="ROC")
    # plt.show()
    #
    # plt.plot(thresholds, tpr, label="ROC")
    # plt.show()
    import numpy as np

    print(thresholds)
    thresholds = thresholds[1:]
    thresholds = thresholds.max() - thresholds
    plt.plot(thresholds, (1 - fpr)[1:], label="ROC")
    plt.xlabel("Distance")
    plt.ylabel("'prob' = 1 - fpr")
    plt.show()


if __name__ == "__main__":
    m()
