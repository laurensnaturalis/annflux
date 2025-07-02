from collections import Counter

import pandas
import numpy as np

from annflux.shared import AnnfluxSource


def get_descendants(parent_to_child, start_node) -> set[int]:
    descendants = set()
    stack = [start_node]
    while stack:
        current = stack.pop()
        descendants.add(current)
        for child in parent_to_child.get(current, []):
            stack.append(child)
    return descendants


def m(project_folder: str):
    source = AnnfluxSource(project_folder)
    t = pandas.read_csv(
        source.data_state_path, dtype={"label_predicted": str, "score_true": float}
    )
    t["dp_parent"] = t["dp_parent"].fillna(-1).astype(int)
    # t.to_parquet("annflux.pq")
    #
    # t = pandas.read_parquet("annflux.pq")
    t["num_children"] = t["dp_depth"].max() - t["dp_depth"]

    print(t["num_children"].min(), t["num_children"].max())

    # print(t[t["num_children"] == t["num_children"].min()].dp_is_ldp)
    t["dp_cluster"] = None
    t["num_children_alt"] = None
    child_to_parent = dict(zip(range(len(t)), t["dp_parent"]))

    parent_to_children = {}
    for child, parent in child_to_parent.items():
        if parent not in parent_to_children:
            parent_to_children[parent] = []
        parent_to_children[parent].append(child)
    #
    for r_, row in t.iterrows():
        t.at[r_, "num_children_alt"] = len(parent_to_children.get(r_, []))
    # assign merge clusters
    n = 0
    dp_idx = set(t[t.dp_is_ldp == 1].index.values)
    for r_, row in t[t.dp_is_ldp == 1].sort_values("dp_depth").iterrows():
        has_dp_children = (
            len(set(sorted(parent_to_children.get(r_, []))).intersection(dp_idx)) > 0
        )
        # print(n, row["num_children"], len(children), has_dp_children, len(descendants))
        if not has_dp_children:
            t.loc[sorted(get_descendants(parent_to_children, r_)), "dp_cluster"] = n
        #
        n += 1
        for child2 in parent_to_children[child_to_parent[r_]]:
            child2_descendants = get_descendants(parent_to_children, child2)
            if len(set(child2_descendants).intersection(dp_idx)) == 0:
                t.loc[
                    sorted(
                        list(child2_descendants)
                        + [
                            child2,
                        ]
                    ),
                    "dp_cluster",
                ] = n
        n += 1

        #
    # merge
    cluster_counts = t.groupby("dp_cluster").size().reset_index(name="counts")
    print(f"{cluster_counts['counts'].sum()=}")
    print(len(t))
    print(t["num_children_alt"].sum())
    target_num_clusters = np.sqrt(len(t)) / 2

    # for r_, row in t[pandas.isna(t["dp_cluster"])].iterrows():
    #     parent = row.dp_parent
    #     while parent != -1:
    #         parent_idx = int(row.dp_parent)
    #         row = t.iloc[parent_idx]
    #         print(row["dp_is_ldp"], parent_idx, len(set(sorted(parent_to_children.get(parent_idx, []))).intersection(dp_idx)) > 0, row["dp_cluster"])
    #         parent = row.dp_parent
    #
    #
    #     break
    print(f"{len(cluster_counts)=}")
    loop = 0
    strategy = "merge_tail"
    largest_cluster_overall = None
    new_num_clusters = None
    sum_tail_factor = 0.9
    tail_factor = 0.25
    reduced_in_loops = []

    while len(t["dp_cluster"].unique()) > target_num_clusters:
        reduced_in_loop = 0
        for _, cluster in cluster_counts.sort_values("counts").iterrows():
            # print(cluster)
            cluster_rows = t[t["dp_cluster"] == cluster["dp_cluster"]]
            num_clusters = 1
            # find ancestor for which descendants have more than one cluster
            if len(cluster_rows.index.values) > 0:
                parent = cluster_rows.index.values[0]
            else:
                continue
            done = False
            while num_clusters == 1:
                parent = child_to_parent[parent]

                descendants_idx = sorted(get_descendants(parent_to_children, parent))
                counts_per_cluster = Counter(t.iloc[descendants_idx]["dp_cluster"])

                num_clusters = len(counts_per_cluster)
                # print(cluster["dp_cluster"], parent, num_clusters)
                if num_clusters > 1:
                    local_common_clusters_counts = [
                        x_
                        for x_ in counts_per_cluster.most_common()
                        if x_[0] is not None
                    ]
                    if strategy == "merge_second_largest":
                        largest_cluster = None
                        second_largest_cluster = None
                        third_largest_cluster = None
                        for cluster2, count_ in local_common_clusters_counts:
                            if cluster2 is not None:
                                if largest_cluster is None:
                                    largest_cluster = cluster2
                                elif second_largest_cluster is None:
                                    second_largest_cluster = cluster2
                                elif third_largest_cluster is None:
                                    third_largest_cluster = cluster2
                                    break
                        if (
                            second_largest_cluster is not None
                            and third_largest_cluster is not None
                        ):
                            # print(third_largest_cluster)
                            from_ = np.where(t["dp_cluster"] == third_largest_cluster)[
                                0
                            ][0]
                            to_ = second_largest_cluster
                        elif second_largest_cluster is not None:
                            # print(second_largest_cluster)
                            from_ = np.where(t["dp_cluster"] == second_largest_cluster)[
                                0
                            ][0]
                            to_ = largest_cluster
                        else:
                            print(counts_per_cluster)
                            continue
                    elif strategy == "merge_tail":
                        sum_ = 0
                        tail = []
                        start_tail = local_common_clusters_counts[
                            int(tail_factor * len(local_common_clusters_counts)) :
                        ]
                        # tail = [x_[0] for x_ in start_tail]
                        for cluster_, count_ in reversed(start_tail):
                            tail.append(cluster_)
                            sum_ += count_
                            if sum_ >= sum_tail_factor * start_tail[0][1]:
                                break

                        if tail[0] == largest_cluster_overall:
                            tail = tail[1:]
                        if len(tail) < 2:
                            continue

                        # if tail[0]
                        # print(tail[1:])
                        to_ = tail[0]
                        # if to_ == largest_cluster_overall:
                        #     continue
                        from_ = np.where(t["dp_cluster"].isin(set(tail[1:])))[0]
                    else:
                        raise NotImplementedError

                    reduced_in_loop += 1
                    t.loc[from_, "dp_cluster"] = to_
                    new_num_clusters = len(t["dp_cluster"].unique())
                    recent_cluster_counts = Counter(t["dp_cluster"]).most_common()[:10]
                    largest_cluster_overall = recent_cluster_counts[0][0]
                    print(
                        "merging",
                        len(from_),
                        to_,
                        new_num_clusters,
                        recent_cluster_counts,
                        loop,
                    )
                    if new_num_clusters <= target_num_clusters:
                        done = True
                        break
            if done:  # break the for loop, the while condition breaks the whole loop
                break
        cluster_counts = t.groupby("dp_cluster").size().reset_index(name="counts")
        if reduced_in_loop < 5:
            sum_tail_factor *= 1.1
            tail_factor *= 0.9
            reduced_in_loops.append(reduced_in_loop)
        if len(reduced_in_loops) > 5:
            if np.all(np.array(reduced_in_loops[-5:]) == 0):
                break
        print(f"{reduced_in_loop=}, {sum_tail_factor=}")
        loop += 1
    # for _, row in cluster_counts.iterrows():
    #     print(row)

    print(len(t["dp_cluster"].unique()))

    cluster_counts = Counter(t["dp_cluster"]).most_common()

    order_map = {
        x_: i_
        for i_, x_ in enumerate([x_[0] for x_ in cluster_counts if x_[0] is not None])
    }

    t["dp_cluster"] = t["dp_cluster"].apply(lambda x_: order_map.get(x_))

    # most needed points
    # TODO: use original features, not embedding
    t["dp_most_needed"] = t["dp_cluster"].max() + 1
    for rank_, (_, row) in enumerate(t.groupby("dp_cluster").size().reset_index(name="counts").sort_values("counts", ascending=False).iterrows()):
        # pick the point with the lowest depth
        t.at[t[t["dp_cluster"]==row.dp_cluster]["dp_depth"].idxmin(), "dp_most_needed"] = rank_
    #

    t.to_csv(source.data_state_path, index=False)


if __name__ == "__main__":
    m("/mnt/big/indeed/mollusca")
