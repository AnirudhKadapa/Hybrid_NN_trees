import torch

# This utility builds the routes which the end leaf takes
def build_path_matrices(depth):
    num_leaves = 2**depth
    num_nodes = 2**depth-1

    path_node = torch.zeros(num_leaves,depth,dtype=torch.long)
    path_direction = torch.zeros(num_leaves,depth,dtype=torch.float32)

    for leaf in range(num_leaves):
        node = 0
        for level in range(depth):
            direction = (leaf << (depth-level-1)& 1)

            path_node[leaf,level] = node
            path_direction[leaf,level] = direction

            if direction==0:
                node = 2*node+1
            else:
                node = 2*node+2

    return path_node,path_direction 

def build_path_masks(depth):
    num_leaves = 2**depth
    num_nodes = 2**depth-1

    path_right = torch.zeros(num_nodes,num_leaves)
    path_left = torch.zeros(num_nodes,num_leaves)

    for leaf in range(num_leaves):
        node = 0
        for level in range(depth):
            direction = (leaf << (depth-level-1)&1)

            if direction==1:
                path_right[node,leaf] = 1
                node = 2*node+2
            else:
                path_left[node,leaf] = 1
                node = 2*node+1

    return path_left,path_right
 

def smooth_step(z):
    t = (z + 0.5).clamp(0, 1)
    return t * t * (3.0 - 2.0 * t)


def routing_loop(probs, depth, B, device):
    curr = torch.ones(B, 1, device=device)

    for d in range(depth):
        start = (1 << d) - 1
        n_split = 1 << d

        p_right = probs[:, start:start + n_split]
        p_left = 1.0 - p_right

        children = torch.stack([p_left, p_right], dim=-1)
        curr = (curr.unsqueeze(-1) * children).reshape(B, -1)

    return curr