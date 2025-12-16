import copy
from collections import defaultdict
from typing import Dict, Set

import faiss
import numpy as np
import pandas
from numpy._typing import NDArray
from sklearn.metrics import pairwise_distances

from annflux.shared import AnnfluxSource


def fast_density_peak_clustering(source_or_folder: str | AnnfluxSource, show=False):
    """
    Applies fast density peak clustering (TODO: ref)

    Modifies the data state on disk

    TODO: apply to original features instead of embeddings
    """
    if not isinstance(source_or_folder, AnnfluxSource):
        source = AnnfluxSource(source_or_folder)
    else:
        source = source_or_folder

    data = pandas.read_csv(
        source.data_state_path, dtype={"label_predicted": str, "score_true": float}
    )


    t_ = data
    features = t_[["e_0", "e_1"]].values

    print("features.shape", features.shape)
    knn_index = faiss.index_factory(
        features.shape[1],
        "Flat",
        {"inner": faiss.METRIC_INNER_PRODUCT, "l2": faiss.METRIC_L2}["l2"],
    )

    features *= 1 - 1e-2 * np.random.rand(features.shape[0], features.shape[1])

    knn_index.train(features)
    knn_index.add(features)

    result = knn_index.search(features, k=15)
    knn_indices = result[1]
    distances = result[0]

    for i_ in range(len(knn_indices)):
        if knn_indices[i_, 0] != i_:
            for j_ in range(knn_indices.shape[1]):
                if knn_indices[i_, j_] == i_:
                    knn_indices[i_, j_] = knn_indices[i_, 0]
                    knn_indices[i_, 0] = i_
                    distances[i_, 0] *= 0.99
                    break

    knn_indices = knn_indices[:, 1:]

    distances = distances[:, 1:]

    knn_densities = 1.0 / distances[:, -1]
    print(knn_densities.shape)
    p1 = set(range(len(features)))

    local_density_peaks, child_to_parent_non_ldp = find_local_density_peak(
        knn_densities, p1, knn_indices, distances
    )
    child_to_parent_ldp, child_to_parent_ldp_depth = fast_find_parent_node_ldp(
        np.array(list(local_density_peaks)), knn_densities, features
    )
    for parent in child_to_parent_non_ldp.values():
        if isinstance(parent, type(np.array)):
            print(parent)
        assert parent in local_density_peaks or parent in child_to_parent_non_ldp.keys()
    for child, parent in child_to_parent_non_ldp.items():
        # print(child, parent)
        assert knn_densities[child] <= knn_densities[parent]
    child_to_ldp_parent = {}
    child_to_depth = {}
    for child, parent in child_to_parent_non_ldp.items():
        if parent in local_density_peaks:
            child_to_ldp_parent[child] = parent
            child_to_depth[child] = 0
        else:
            child_ = child
            parent_ = child_to_parent_non_ldp[child]
            d_ = 0
            while parent_ not in local_density_peaks:
                parent_ = child_to_parent_non_ldp[child_]
                child_ = parent_
                d_ += 1
                if d_ > 100:
                    print(child, child_, parent_)
                    exit(1)
                # print(d_)
            child_to_ldp_parent[child] = parent_
            child_to_depth[child] = -d_

    print("len(child_to_ldp_parent)", len(child_to_ldp_parent))
    combined = copy.deepcopy(child_to_parent_non_ldp)
    combined.update(child_to_parent_ldp)
    print("len(combined)", len(combined))
    combined_depth = {}
    num_children: list[int] = [
        0,
    ] * len(combined)
    for child, parent in combined.items():
        child_ = child
        parent_ = combined[child]
        if parent_ is not None:
            num_children[parent_] += 1
        d_ = 0
        while parent_ is not None:
            parent_ = combined[child_]
            if parent_ is not None:
                num_children[parent_] += 1
            child_ = parent_
            d_ += 1
            if d_ > 100:
                print(child, child_, parent_)
                exit(1)
            # print(d_)
        combined_depth[child] = d_
    print("max(combined_depth.values())", max(combined_depth.values()))
    print("max(num_children)", max(num_children))
    local_density_peaks_set = set(local_density_peaks)
    indices_ = range(len(data))
    data["dp_is_ldp"] = [int(r in local_density_peaks_set) for r in indices_]
    data["dp_depth"] = [num_children[r] for r in indices_]
    data["dp_depth"] = data["dp_depth"].max() - data["dp_depth"]
    data["dp_depth_alt"] = [combined_depth[r] for r in indices_]
    data["dp_parent"] = [
        int(combined[r]) if combined[r] is not None else -1 for r in indices_
    ]

    display_order = [
        None,
    ] * len(data)

    for i, pos in enumerate(np.argsort(data["dp_depth"].values)):
        display_order[pos] = i

    data["display_order"] = display_order
    print(np.where(data["dp_depth"].values == 0)[0][0])
    print(np.argsort(data["dp_depth"].values))
    print(f"Writing to {source.data_state_path=}")
    data.to_csv(source.data_state_path, index=False)

    if show:
        import matplotlib.pyplot as plt

        species_true = data.label_true

        if features.shape[1] != 2:
            raise NotImplementedError
        from matplotlib.pyplot import cm

        plt.subplot(221)
        plt.title(f"|ldp| = {len(local_density_peaks)}")
        color = iter(cm.rainbow(np.linspace(0, 1, len(local_density_peaks)))) # ty: ignore
        ldp_parent_to_children = defaultdict(lambda: [])
        for child, parent in child_to_ldp_parent.items():
            ldp_parent_to_children[parent].append(child)
        for ldp in sorted(local_density_peaks):
            children = ldp_parent_to_children[ldp]
            c = next(color).reshape(1, -1)
            p = plt.scatter(features[children, 0], features[children, 1], c=c)
            p = plt.scatter(
                features[ldp : ldp + 1, 0], features[ldp : ldp + 1, 1], c=c, marker="x"
            )  # c=p.get_facecolors()[0].reshape(1,-1), marker="x")
            # print(p.get_facecolors()[0])
        plt.subplot(222)

        color = iter(cm.rainbow(np.linspace(0, 1, len(set(species_true))))) # ty: ignore
        for species in set(species_true):
            sel = np.where(species_true == species)[0]
            centroid = np.mean(features[sel], axis=0)
            # print(centroid)
            c = next(color).reshape(1, -1)
            p = plt.scatter(features[sel, 0], features[sel, 1], c=c)
            plt.annotate(species, centroid)

        plt.subplot(223)

        children = ldp_parent_to_children[list(local_density_peaks)[0]]
        global_to_local = dict(list(zip(children, range(len(children)))))
        x = features[children, 0]
        y = [child_to_depth[x_] for x_ in children]
        for c, child in enumerate(children):
            if child_to_parent_non_ldp[child] in children:
                local_parent_index = global_to_local[child_to_parent_non_ldp[child]]
                plt.plot(
                    [x[c], x[local_parent_index]],
                    [y[c], y[local_parent_index]],
                    c="k",
                    lw=0.5,
                )
        plt.scatter(x, y)

        plt.subplot(224)

        ldp_parent_to_children = defaultdict(lambda: [])
        for child, parent in child_to_parent_ldp.items():
            ldp_parent_to_children[parent].append(child)
        children = np.array(list(sorted(local_density_peaks)))
        global_to_local = dict(list(zip(children, range(len(children)))))
        x = features[children, 0]
        y = [child_to_parent_ldp_depth[x_] for x_ in children]
        color = iter(cm.rainbow(np.linspace(0, 1, len(local_density_peaks)))) # ty: ignore
        for c, child in enumerate(children):
            if child_to_parent_ldp[child] in children:
                local_parent_index = global_to_local[child_to_parent_ldp[child]]
                plt.plot(
                    [x[c], x[local_parent_index]],
                    [y[c], y[local_parent_index]],
                    c="k",
                    lw=0.5,
                )
        plt.scatter(x, y, c=[next(color) for _ in range(len(x))])

        plt.show()


def find_local_density_peak(
    densities: NDArray, p1: Set[int], knn_indices: NDArray, knn_distances: NDArray
):
    local_density_peaks: Set[int] = set()
    child_to_parent: Dict[int, int] = {}
    delta = [
        None,
    ] * len(densities)
    for i in p1.copy():
        if i not in p1:
            continue
        nn_i = knn_indices[i]
        nn_densities = densities[nn_i]
        if np.all(densities[i] >= nn_densities):
            local_density_peaks.add(i)
            # p1 -= set(nn_i)
            # print("ldp")
        else:
            # from NN with higher density pick the one with the smallest distance to i as parent
            i_nn_has_higher_density = np.where(nn_densities > densities[i])[0]
            min_d = np.inf
            min_i_nn = None
            for i_nn in i_nn_has_higher_density:
                if knn_distances[i, i_nn] < min_d:
                    min_i_nn = i_nn
                    min_d = knn_distances[i, i_nn]
            if min_i_nn is None:
                print(knn_distances[i])
                print("oemba")
            j = knn_indices[i, min_i_nn]
            child_to_parent[i] = int(j)
            delta[i] = min_d
    return local_density_peaks, child_to_parent


def fast_find_parent_node_ldp(local_density_peaks, densities, features):
    sort_ = np.argsort(densities[local_density_peaks])
    sorted_ldp = local_density_peaks[sort_]
    pwd = pairwise_distances(features[sorted_ldp])
    child_to_parent = {}
    child_to_parent_global = {}
    for i_ in range(len(sorted_ldp) - 1):
        parent_i = np.argmin(pwd[i_, i_ + 1 :]) + (i_ + 1)
        child_to_parent_global[sorted_ldp[i_]] = sorted_ldp[parent_i]
        child_to_parent[i_] = parent_i
    child_to_parent[len(local_density_peaks) - 1] = None  # root
    child_to_parent_global[sorted_ldp[len(local_density_peaks) - 1]] = None  # root
    #
    child_to_depth = {}
    for child, parent in child_to_parent.items():
        # traverse to root
        child_ = child
        parent_ = child_to_parent[child]
        d_ = 0
        while parent_ is not None:
            parent_ = child_to_parent[child_]
            child_ = parent_
            d_ += 1
        child_to_depth[sorted_ldp[child]] = -d_

    return child_to_parent_global, child_to_depth


if __name__ == "__main__":
    fast_density_peak_clustering("/mnt/big/indeed/mollusca")
