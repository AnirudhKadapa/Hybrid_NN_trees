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
 