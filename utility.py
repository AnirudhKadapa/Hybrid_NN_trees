import torch

def build_paths(depth, device=None):
    num_leaves = 2 ** depth
    leaves = torch.arange(num_leaves, device=device)

    path_nodes = torch.zeros(num_leaves, depth, device=device)
    path_dirs = torch.zeros(num_leaves, depth, device=device)

    node = torch.zeros(num_leaves, dtype=torch.long, device=device)

    for level in range(depth):
        direction = (leaves >> (depth - level - 1)) & 1

        path_nodes[:, level] = node
        path_dirs[:, level] = direction

        node = 2 * node + 1 + direction

    return path_nodes, path_dirs

def build_path_masks(depth, device=None):
    num_leaves = 2 ** depth
    num_nodes = 2 ** depth - 1

    leaves = torch.arange(num_leaves, device=device)  # (L,)

    path_left = torch.zeros(num_nodes, num_leaves, device=device)
    path_right = torch.zeros(num_nodes, num_leaves, device=device)

    node = torch.zeros(num_leaves, dtype=torch.long, device=device)

    for level in range(depth):
        directions = (leaves >> (depth - level - 1)) & 1  # 0 = left, 1 = right

        leaf_ids = torch.arange(num_leaves, device=device)

        path_left[node[directions == 0], leaf_ids[directions == 0]] = 1.0
        path_right[node[directions == 1], leaf_ids[directions == 1]] = 1.0

        node = 2 * node + 1 + directions

    return path_left, path_right

def smooth_step(z):
    t = (z + 0.5).clamp(0, 1)
    return t * t * (3.0 - 2.0 * t)


if __name__=="__main__":
    node, dir = build_paths(3)
    print(node)
    print()
    print(dir)
    