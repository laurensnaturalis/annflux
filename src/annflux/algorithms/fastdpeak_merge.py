from collections import Counter

import pandas
import numpy as np


def get_descendants(parent_to_child, start_node) -> set[int]:
    descendants = set()
    stack = [start_node]
    while stack:
        current = stack.pop()
        descendants.add(current)
        for child in parent_to_child.get(current, []):
            stack.append(child)
    return descendants


def m():
    # t = pandas.read_csv("indeed.csv")
    # t["dp_parent"] = t["dp_parent"].fillna(-1).astype(int)
    # t.to_parquet("annflux.pq")

    t = pandas.read_parquet("annflux.pq")
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
        descendants = get_descendants(parent_to_children, r_)
        # print(n, row["num_children"], len(children), has_dp_children, len(descendants))
        if not has_dp_children:
            t.loc[sorted(descendants), "dp_cluster"] = n
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
    new_cluster_counts = len(cluster_counts)
    loop = 0
    while len(t["dp_cluster"].unique()) > target_num_clusters:
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

                descendants = get_descendants(parent_to_children, parent)

                counts_per_cluster = Counter(t.iloc[sorted(descendants)]["dp_cluster"])
                largest_cluster = counts_per_cluster.most_common(1)[0][0]
                num_clusters = len(counts_per_cluster)
                # print(cluster["dp_cluster"], parent, num_clusters)
                if num_clusters > 1:
                    # assign to largest cluster
                    print(
                        "merging",
                        largest_cluster,
                        cluster["dp_cluster"],
                        new_cluster_counts - 1,
                        len(t["dp_cluster"].unique()),
                        loop
                    )
                    t.loc[cluster_rows.index, "dp_cluster"] = largest_cluster
                    new_cluster_counts -= 1
                    if new_cluster_counts <= target_num_clusters:
                        done = True
                        break
            if done:  # break the for loop, the while condition breaks the whole loop
                break
        cluster_counts = t.groupby("dp_cluster").size().reset_index(name="counts")
        loop += 1
    # for _, row in cluster_counts.iterrows():
    #     print(row)

    print(len(t["dp_cluster"].unique()))

    t.to_csv(
        "/mnt/big/indeed/diopsis-hazehorst-apr5-6/annflux/annflux.csv", index=False
    )


if __name__ == "__main__":
    m()
