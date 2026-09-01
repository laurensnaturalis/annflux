def get_full_path(child_to_parent_edges: list[tuple[str, str]], leaf: str):
    parent_map = {}
    for child, parent in child_to_parent_edges:
        parent_map[child] = parent if parent != "null" else None

    path = []
    current_node = leaf
    while current_node is not None:
        path.append(current_node)
        current_node = parent_map.get(current_node)

    path.reverse()
    return path
