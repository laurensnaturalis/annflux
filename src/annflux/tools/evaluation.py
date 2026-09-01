import numpy as np
from sklearn.metrics import accuracy_score


def compute_ece(
    table,
    half_bin_width=0.025,
    start=0.0,
    probability_name="combined_score",
    label_true_name="label_true",
    label_predicted_name="label_predicted",
):
    table.to_csv("ece.csv", index=False)
    bin_centers = np.arange(
        start + half_bin_width, 1 + half_bin_width, 2 * half_bin_width
    )
    scores = []
    num = []
    has_data = []
    for bin_center in bin_centers:
        selection = table[
            np.logical_and(
                bin_center - half_bin_width < table[probability_name],
                table[probability_name] < bin_center + half_bin_width,
            )
        ]
        # print(selection[probability_name].values)
        # print(selection[label_true_name].values)
        # print(selection[label_predicted_name].values)
        scores.append(
            accuracy_score(
                selection[label_true_name].values,
                selection[label_predicted_name].values,
            )
            if len(selection) > 0
            else 0
        )
        has_data.append(len(selection) > 0)
        num.append(len(selection))
        # print(bin_center, )
    # print(np.array(num) * np.abs(np.array(scores) - np.array(bin_centers)))
    errors = np.array(num) * np.abs(np.array(scores) - np.array(bin_centers))
    ece = errors.sum() / len(table)
    last_bin_ece = errors[-1] / num[-1]
    #
    bla = table.sort_values(by=probability_name)
    correctcum = bla.apply(
        lambda row_: row_[label_true_name] == row_[label_predicted_name], axis=1
    ).values.cumsum()
    numcum = np.arange(1, len(correctcum) + 1)
    # print(correctcum / numcum)
    tmp_ = np.where((correctcum / numcum) >= 0.5)[0]
    if len(tmp_) > 0:
        tmp_ = tmp_[0]
    else:
        tmp_ = None
    # print(tmp_)
    fifty_threshold = bla[probability_name].values[tmp_]
    #
    return (
        ece,
        bin_centers,
        half_bin_width,
        num,
        scores,
        last_bin_ece,
        has_data,
        fifty_threshold,
    )
