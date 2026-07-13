import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from utility import  build_path_masks, smooth_step, build_paths

class SoftDecisionTree(nn.Module):
    def __init__(self, input_dim, depth,num_classes):
        super().__init__()
        self.num_treenodes = 2**depth -1
        self.num_leaves = 2**depth
        self.node_weights = nn.Linear(input_dim,self.num_treenodes)
        self.leaves_logits = nn.Parameter(torch.randn(self.num_leaves,num_classes)*0.05)
    
    def forward(self,x):
        batch_size = x.shape[0]
        decision_probs = torch.sigmoid(self.node_weights(x))
        path_probs = x.new_ones(batch_size,1)
        node_index = 0

        for i in range(self.depth):
            at_level = 2**i
            curr_probs = []

            for j in range(at_level):
                p = decision_probs[:,node_index].unsqueeze(1)
                parent_path = path_probs[:,j].unsqueeze(1)
                left = p * parent_path
                right = (1-p)* parent_path
                curr_probs.append(left)
                curr_probs.append(right)
                node_index+=1

            path_probs = torch.cat(curr_probs,dim=1)

        output = torch.einsum("bl,lc->bc",path_probs,self.leaves_logits)
        return output


class VectorizedSDT(nn.Module):
    def __init__(self,input_dim, depth, num_classes):
        super().__init__()
        self.num_leaves = 2**depth
        self.numtree_nodes = 2**depth-1

        self.node_weights = nn.Linear(input_dim,self.numtree_nodes)
        self.leaf_weights = nn.Parameter(torch.randn(self.num_leaves,num_classes)*0.05)

        path_left,path_right = build_path_masks(depth)

        self.register_buffer("path_left",path_left)
        self.register_buffer("path_right",path_right)

    def forward(self,x):
        decision_logits = self.node_weights(x)
        log_p_right = F.logsigmoid(decision_logits)
        log_p_left = F.logsigmoid(-decision_logits)

        log_path_probs = (
            log_p_left @ self.path_left
            + log_p_right @ self.path_right
        )

        path_probs = torch.exp(log_path_probs)

        output = path_probs @ self.leaf_logits

        return output
    

class FastNATLinearSharedLayer(nn.Module):
    def __init__(self, input_dim, n_nodes, depth=6, dropout=0.0):
        super().__init__()

        self.n_nodes = n_nodes
        self.depth = depth
        self.n_internal = 2 ** depth - 1
        self.n_leaves = 2 ** depth

        self.bn = nn.BatchNorm1d(input_dim)
        self.W_shared = nn.Parameter(torch.randn(self.n_internal, input_dim) * (2.0 / input_dim) ** 0.5)
        self.b_shared = nn.Parameter(torch.zeros(self.n_internal))
        self.leaves = nn.Parameter(torch.randn(n_nodes, self.n_leaves) * 0.1)
        self.drop = nn.Dropout(dropout)

        path_left, path_right = build_path_masks(depth)

        self.register_buffer("path_left", path_left)
        self.register_buffer("path_right", path_right)

    def forward(self, x):
        x_bn = self.bn(x)

        logits = F.linear(x_bn, self.W_shared, self.b_shared)
        probs = smooth_step(logits)
        eps = 1e-6
        probs = probs.clamp(eps, 1.0 - eps)

        log_p_right = torch.log(probs)
        log_p_left = torch.log(1.0 - probs)

        log_path_probs = (
            log_p_left @ self.path_left
            + log_p_right @ self.path_right
        )

        path_probs = torch.exp(log_path_probs)
        out = path_probs @ self.leaves.T

        return self.drop(out)

# Fully independent Soft Differentiable Tree with Smooth Step
class FullyIndependentSDT(nn.Module):
    def __init__(self, input_dim, depth, num_classes, device):
        self.num_leaves = 2**depth
        self.num_nodes = 2**depth - 1 

        self.node_weights = nn.Linear(input_dim, self.num_nodes, bias=True, device= device)
        self.leaf_weights = nn.Parameter(torch.randn(self.num_leaves, num_classes, device=device) * 0.01)

        path_nodes, path_dirs = build_paths(depth,device)

        self.register_buffer("path_nodes",path_nodes.long())
        self.register_buffer("path_dirs",path_dirs.long())

    def forward(self,x):
        decision_logits = self.node_weights(x)
        eps = 1e-6
        probs = smooth_step(decision_logits).clamp(eps,1-eps)
        
        log_probs_right = torch.log(probs)
        log_probs_left = torch.log(1.0-probs)

        log_probs = torch.stack([log_probs_left,log_probs_right],dim=-1)

        path_log_probs = log_probs[:,self.path_nodes,self.path_dirs]
        log_path_probs = path_log_probs.sum(dim=-1)
        path_probs = torch.exp(log_path_probs)
        out = path_probs @ self.leaf_weights
        
        return self.drop(out)

# fast SDT using smooth function
class FastSDT(nn.Module):
    def __init__(self,input_dim, depth, num_classes):
        super().__init__()

        self.num_nodes = 2**depth-1
        self.num_leaves = 2**depth

        self.node_weights = nn.Linear(input_dim,self.num_nodes,bias=True)
        self.leaf_logits = nn.Parameter(torch.randn(self.num_leaves,num_classes)*0.01)
        
        '''
        both path_nodes and path_dirs have the sanme shape -> (num_leaves, depth)
        '''
        path_nodes, path_dirs = build_paths(depth)
        self.register_buffer("path_nodes",path_nodes.long())
        self.register_buffer("path_dirs",path_dirs.long())

    def forward(self,x):
        node_logits = self.node_weights(x)
        eps = 1e-6
        probs = smooth_step(node_logits).clamp(eps,1-eps)

        '''
        node_logits -> (B,num_nodes)
        probs -> (B, num_nodes)
        '''
        log_probs_left = torch.log(probs)
        log_probs_right = torch.log(1.0-probs)
   
        '''
        log_all_probs -> (B,num_nodes,2)
        
        (left,right): multiple batches, num_nodes across the row and column (left, right)

        '''
        log_all_probs = torch.stack([log_probs_left,log_probs_right],dim=-1) 

        # path_log_probs -> (B, num_leaves, depth)
        path_log_probs = log_all_probs[:,self.path_nodes,self.path_dirs]
        log_probs_path = path_log_probs.sum(dim=-1)
        prob_paths = torch.exp(log_probs_path)

        return prob_paths @ self.leaf_logits
        
# Fast SDT with smooth function with shared weights across levels
class ObliviousWeghtSharedNAT(nn.Module):
    def __init__(self,input_dim, depth,num_classes):
        super().__init__()
        self.depth = depth
        self.num_leaves = 2**depth
        self.num_nodes = 2**depth-1

        self.node_weights = nn.Linear(input_dim,depth)
        self.leaf_weights = nn.Linear(self.num_leaves, num_classes, bias=False)

        _, path_dir = build_paths(depth)

        self.register_buffer("path_dir",path_dir.float())

    def forward(self,x):
        node_logits = self.node_weights(x)
        eps = 1e-6
        probs = smooth_step(node_logits).clamp(eps,1-eps)

        log_prob_left = torch.log(probs)
        log_prob_right = torch.log(1.0-probs)

        log_all_paths = torch.cat([log_prob_left,log_prob_right],dim=-1) # shape -> (batch, depth, 2)
        level_idx = torch.arange(self.depth).unsqueeze(0)

        path_log_probs = log_all_paths[:, level_idx, self.path_dirs]
        leaf_log_probs = path_log_probs.sum(-1) 

        leaf_probs = torch.exp(leaf_log_probs)
        logits = self.leaf_weights(leaf_probs)
        return logits




