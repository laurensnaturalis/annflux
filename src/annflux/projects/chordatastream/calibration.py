import pandas
import numpy as np


def m():
    # t = pandas.read_csv("/mnt/big/indeed/chordata_adb/annflux/annflux.csv")
    # t = pandas.read_csv("/mnt/big/indeed/diopsis_quality/annflux/annflux.csv")
    t = pandas.read_csv("/home/lhogeweg/annflux/data/streetsurfacevis/annflux/annflux.csv", dtype={"scores_predicted": str})

    # print(t.label_predicted)
    # print(t.label_true)
    # print(t.scores_predicted)

    correct = []
    probability = []
    labels_predicted_ = []
    for _, row in t[
        ~pandas.isna(t.label_true) & ~pandas.isna(t.label_predicted)
    ].iterrows():
        labels_true = row.label_true.split(",")
        labels_predicted = row.label_predicted.split(",")
        scores_predicted = map(float, row.scores_predicted.split(","))

        for label_predicted, score_predicted in zip(labels_predicted, scores_predicted):
            correct.append(label_predicted in labels_true)
            labels_predicted_.append(label_predicted)
            probability.append(score_predicted)

    ece = 0
    for bin_center in np.arange(0.025, 1, 0.05):
        selection = np.where(
            (probability >= bin_center - 0.05 / 2)
            & (probability < bin_center + 0.05 / 2)
        )[0]
        bin_calibration = np.mean(np.array(correct)[selection])
        if np.abs(bin_center - (1 - 0.025)) < 1e-6:
            print(bin_center, bin_calibration)
        ece += len(selection) / len(probability) * np.abs   (bin_center - bin_calibration) if not np.isnan(bin_calibration) else 0
    print("ECE", ece)

    for label in set(labels_predicted_):
        ece = 0
        for bin_center in np.arange(0.025, 1, 0.05):
            selection = np.where(
                (probability >= bin_center - 0.05 / 2)
                & (probability < bin_center + 0.05 / 2)
                & (np.array(labels_predicted_) == label)
            )[0]
            bin_calibration = np.mean(np.array(correct)[selection])
            if np.abs(bin_center - (1 - 0.025)) < 1e-6:
                print(label, "last bin CE [0.95-1.00]", np.round(bin_calibration, 2))
            ece += (
                len(selection) / len(probability) * np.abs(bin_center - bin_calibration)
                if not np.isnan(bin_calibration)
                else 0
            )
        sel_probs = np.array(probability)[np.array(labels_predicted_) == label]


        print(label, "ECE", np.round(ece, 4), "% predictions > 95%:", np.round(len(np.where(sel_probs > 0.95)[0]) / len(sel_probs), 2))

    # for bin_center in np.arange(0.955, 1, 0.005):
    #     selection = np.where(
    #         (probability >= bin_center - 0.005 / 2)
    #         & (probability < bin_center + 0.005 / 2)
    #     )[0]
    #     bin_calibration = np.mean(np.array(correct)[selection])
    #     print(bin_center - 0.005 / 2, bin_center + 0.005 / 2, bin_calibration)


if __name__ == "__main__":
    m()
