import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utility import build_path_masks

class PlainNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,hidden_dim), nn.ReLU(),nn.Dropout(dropout),
            nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.Dropout(dropout),
            nn.Linear(hidden_dim,n_classes)
        )
    def forward(self,x):
        return self.net(x)
    
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
        decision_probs = torch.sigmoid(self.node_weights(x))
        log_p_right = F.logsigmoid(decision_probs)
        log_p_left = F.logsigmoid(-decision_probs)

        log_path_probs = (
            log_p_left @ self.path_left
            + log_p_right @ self.path_right
        )

        path_probs = torch.exp(log_path_probs)

        output = path_probs @ self.leaf_logits

        return output







            
