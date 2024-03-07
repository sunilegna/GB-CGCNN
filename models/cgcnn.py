'''
Below codes are simplified version of original codes from https://github.com/txie-93/cgcnn
  Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties
  Xie, Tian and Grossman, Jeffrey C.
'''

from torch_scatter import scatter
import torch.nn as nn
import torch
from torch_geometric.nn.aggr import AttentionalAggregation

class CGConv(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len):
        super(CGConv, self).__init__()
        
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()
        
    def forward(self, x, edge_index, edge_attr):
        # N, M = edge_index.shape
        xi = x[edge_index[0]]
        xj = x[edge_index[1]]
        
        total_nbr_fea = torch.cat([xi, xj, edge_attr], dim=-1)
        
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=-1)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        
        nbr_message = nbr_filter * nbr_core
        nbr_sumed = scatter(nbr_message, edge_index[0], dim=0, reduce='sum')
        nbr_sumed = self.bn2(nbr_sumed)
        return nbr_sumed
        
class GbGraphConvNet(nn.Module):
    def __init__(self, orig_atom_fea_len, nbr_fea_len, atom_fea_len=64):
        super(GbGraphConvNet, self).__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.conv = CGConv(atom_fea_len, nbr_fea_len)   
        self.attention_nn = torch.nn.Linear(atom_fea_len, 1)
        self.GA = AttentionalAggregation(self.attention_nn)     
        self.fc_out = nn.Linear(atom_fea_len, 1)
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(p=0.2)
        self.silu = nn.SiLU()

    def forward(self, graphs, lower_f):
        atom_fea = self.embedding(graphs.x)
        if lower_f is not None:
            atom_fea = atom_fea + lower_f
        atom_fea = self.conv(atom_fea, graphs.edge_index, graphs.edge_attr)
        atom_fea_ = self.silu(atom_fea)
        crys_fea = self.GA(atom_fea_, graphs.batch)  
        # crys_fea = self.dropout(crys_fea)           
        crys_fea = self.fc_out(crys_fea)
              
        return atom_fea, crys_fea
    
    @classmethod
    def get_model(cls, orig_atom_fea_len, nbr_fea_len, model_config):
        model = GbGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    atom_fea_len=model_config["atom_fea_len"])
        return model
