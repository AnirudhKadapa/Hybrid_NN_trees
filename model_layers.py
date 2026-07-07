"""
Here we are going to add vectorized version of the model with improvement versions of NAT and other comparison models like xgboost etc.
"""

import torch 
import torch.nn as nn
from torch.amp import autocast
import math
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
    def __init__(self,input_dim, n_trees, depth):
        super().__init__()
        self.depth = depth
        self.num_nodes = 2**depth-1
        self.num_leaves = 2**depth

        self.node_weights = nn.Parameter(torch.randn(n_trees,self.num_nodes,input_dim)*0.01)
        self.node_bias = nn.Parameter(torch.zeros(n_trees,self.num_nodes))

        self.leaf_logits = nn.Parameter(torch.randn(n_trees, self.num_leaves)*0.01)

        path_nodes, path_dir = build_paths(depth)
        self.register_buffer("path_nodes",path_nodes.long())
        self.register_buffer("path_dir",path_dir.long())
    
    def _leaf_probs(self,x):
        with autocast(device_type=x.device.type,enabled=False):
            x32 = x.float()
            W32 = self.node_weights.float()
            b32 = self.node_bias.float()
            node_logits = torch.einsum("bi,tni->btn",x32,W32) + b32.unsqueeze(0)
            eps =1e-6
            probs = smooth_step(node_logits).clamp(eps,1-eps)
            
            # probs shape -> (batch, n_trees, num_nodes)
            log_probs_left = torch.log(probs)
            log_probs_right = torch.log(1.0-probs)

            '''
            log_all_probs -> (batch, num_trees, num_nodes, 2)
            log_path_probs -> (batch, num_trees, num_leaves, depth)
            probs_all_log_paths -> (batch, num_trees, num_leaves)
            '''
            log_all_probs = torch.stack([log_probs_left,log_probs_right],dim=-1)
            log_path_probs = log_all_probs[:,:,self.path_nodes,self.path_dir]
            prob_all_log_paths = log_path_probs.sum(dim=-1)

            return torch.exp(prob_all_log_paths)

    @torch.no_grad()
    def leaf_entropy(self,x,batch_size=8192):
        batch_sum = None
        n = x.size(0)
        for i in range(0,n,batch_size):
            # _leaf_prob shape -> (batch, trees, leaves)
            leaf_p = self._leaf_probs(x[i:i+batch_size])
            leaf_sum = leaf_p.sum(0)
            if batch_sum is None:
                batch_sum = leaf_sum #shape -> (Trees, leaves)
            else:
                batch_sum.add_(leaf_sum)
            
        probs = batch_sum / n
        probs = probs/probs.sum(-1,keepdim=True).clamp(min=1e-8)

        # H = -(probs * probs.clamp(min=1e-12).log()).sum(-1) 
        H = torch.special.entr(probs).sum(-1)
        H_max = math.log(self.num_leaves)

        if H_max == 0:
            return  0.0
        
        return (H.mean()/H_max).item()

    def forward(self,x):
        prob_paths = self._leaf_probs(x)
        tree_outputs = torch.einsum("btl,tl->bt",prob_paths,self.leaf_logits)
        return tree_outputs
    


# NAT shared- shared across all parallel trees and independent for the leaves
