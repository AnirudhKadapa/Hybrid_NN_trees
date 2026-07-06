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


# oblivious NAT -Has same weight at the same level in a singular tree and different across parallel trees
# oblivious NAT layer



# NAT fully independent Layer
class FullyIndependentNATLayer(nn.Module):
    def __init__(self,input_dim, n_trees, depth ,output_dim):
        super().__init__()
        self.num_nodes = 2**depth-1
        self.num_leaves = 2**depth

        self.node_weights = nn.Parameter(torch.randn(n_trees,self.num_nodes,input_dim)*0.01)
        self.node_bias = nn.Parameter(torch.randn(n_trees,self.num_nodes))

        self.leaf_logits = nn.Paramater(torch.randn(n_trees, self.num_leaves, output_dim)*0.01)

        path_nodes, path_dir = build_paths(depth)
        self.register_buffer("path_nodes",path_nodes.long())
        self.register_buffer("path_dir",path_dir.long())
    
    def forward(self,x):
        node_logits = torch.einsum("bi,tni->btn",x,self.node_weights) + self.node_bias.unsqueeze(0)
        eps =1e-6
        probs = smooth_step(node_logits).clamp(eps,1-eps)
        '''
        probs shape -> (batch, n_trees, num_nodes)
        '''
        log_probs_left = torch.log(probs)
        log_probs_right = torch.right(1.0-probs)

        '''
        log_all_probs -> (batch, num_trees, num_nodes, 2)
        So, for log_path_probs -> (batch, num_trees, num_leaves, depth)
        probs_all_log_paths -> (batch, num_trees, num_leaves)
        '''
        log_all_probs = torch.stack([log_probs_left,log_probs_right],dim=-1)
        log_path_probs = log_all_probs[:,:,self.path_nodes,self.path_dir]
        prob_all_log_paths = log_path_probs.sum(dim=-1)

        prob_paths = torch.exp(prob_all_log_paths) 

        tree_outputs = torch.einsum("btl,tlo->bto",prob_paths,self.leaf_logits)

        return tree_outputs


# NAT shared- shared across all parallel trees and independent for the leaves
