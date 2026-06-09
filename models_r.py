"""
Here we are going to add vectorized version of the model with improvement versions of NAT and other comparison models like xgboost etc.
"""

import torch 
import torch.nn as nn
from utility import build_paths, smooth_step


# Single NAT tree (Fully Independent)
class SinglularNATtree(nn.Module):
    def __init__(self, input_dim, depth, num_classes):
        super().__init__()

        self.num_nodes = 2**depth -1
        self.num_leaves = 2**depth
        self.node_weights = nn.Linear(input_dim, self.num_nodes,bias=True) 
        self.leaf_logits = nn.Parameter(torch.randn(self.num_leaves,num_classes)*0.01)

        path_nodes,path_dirs = build_paths(depth, device=None)

        self.register_buffer("path_nodes",path_nodes)
        self.register_buffer("path_dirs",path_dirs)
        
    def forward(self,x):
        node_logits = self.node_weights(x) # Broadcast(B, input_dim)x(input_dim, num_nodes)-> (B, num_nodes)
        p_right = smooth_step(node_logits)
        path_p = p_right[:, self.path_nodes]    # (B, num_leaves, depth)

        path_probs = torch.where(self.path_dirs.unsqueeze(0), path_p, 1.0 - path_p)      # (1, num_leaves, depth)
        leaf_probs = path_probs.prod(dim=-1) 
        out = leaf_probs @ self.leaf_logits     # (B, num_classes)

        return out

# NAT TREE shared weights(shared)
class SharedWeightSingularTree(nn.Module):
    def __init__(self,input_dim, depth, num_classes):
        super().__init__()

        self.num_leaves = 2**depth
        self.num_nodes = 2**depth -1
        self.node_weights = nn.Linear(input_dim,1, bias=True)
        self.leaf_logits = nn.Parameter(torch.randn(self.num_leaves,num_classes)*0.01)
        
        path_nodes,path_dirs = build_paths(depth, device=None)
        self.register_buffer("path_nodes",path_nodes)
        self.register_buffer("path_dirs",path_dirs)

    def forward(self,x):
        node_logits = self.node_weights(x)
        p_right = smooth_step(node_logits)
        path_p = p_right.expand(-1, self.num_leaves, self.depth)
        path_probs = torch.where(self.path_dirs.unsqueeze(0), path_p, 1.0 - path_p)   # (1, num_leaves, depth)
        leaf_probs = path_probs.prod(dim=-1)
        out = leaf_probs @ self.leaf_logits  # (B, num_classes)

        return out
    
# oblivious Singular NAT Tree
class LevelSharedSingularTree(nn.Module):
    def __init__(self,input_dim, depth, num_classes):
        super().__init__()

        self.num_leaves = 2**depth
        self.num_nodes = 2**depth-1
        self.node_weights = nn.Linear(input_dim, depth, bias=True)
        self.leaf_logits = nn.Parameter(torch.randn(self.num_leaves,num_classes)*0.01)

        path_nodes,path_dirs = build_paths(depth,device=None)
        self.register_buffer("path_nodes",path_nodes)
        self.register_buffer("path_dirs",path_dirs)

    def forward(self,x):
        node_logits = self.node_weights(x)
        p_right = smooth_step(node_logits)
        path_p = p_right.unsqueeze(1).expand(-1, self.num_leaves, -1)  # (B, num_leaves, depth)
        path_probs = torch.where(self.path_dirs.unsqueeze(0), path_p, 1.0-path_p)  # (1, num_leaves, depth)
        leaf_probs = path_probs.prod(dim=-1)  
        out = leaf_probs @ self.leaf_logits   

        return out

     




# oblivious NAT -Has same weight at the same level in a single tree and different across parallel trees
class Oblivious_NAT(nn.Module):
    def __init__(self, input_dim, n_trees, depth, num_classes):
        super().__init__()


# NAT fully independent


# NAT shared- shared across all parallel trees and independent for the leaves