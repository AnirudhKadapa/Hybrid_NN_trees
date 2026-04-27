import torch 
import torch.nn as nn
import numpy as np

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
    




            
