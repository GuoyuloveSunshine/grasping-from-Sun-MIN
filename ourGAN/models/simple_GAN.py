import torch
import torch.nn as nn
from models.skeleton_gnn import Node2EdgeLayer, Edge2NodeLayer, Node2NodeLayer
from models.enc_and_dec import calculate_adj_matrix

"""
Inputs:
    node_feats - Tensor with node features of shape [batch_size, num_joints, c_in]-N*23*3   - normalized position information
    edge_feats - Tensor with edge features of shape [batch_size, num_offset + one_base, features]- N*23*1  - normalized offset information
    adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections.
                Shape: [batch_size, num_nodes, num_nodes]

    input - form [batch_size, num_joint, c_in] - N * num_joint *(x, y, z)
"""
class Discriminator(nn.Module):
    def __init__(self, args, topology):
        super(Discriminator, self).__init__()
        self.c_in_node = args.position_feature
        self.c_out_node = args.position_feature
        self.c_in_edge = args.offset_feature
        self.c_out_edge = args.offset_feature
        self.c_in_conv = args.position_feature + args.offset_feature
        self.c_out_conv = args.position_feature + args.offset_feature
        self.adj_matrix = calculate_adj_matrix(topology)
        self.len = len(topology)
        # network module
        self.layer1 = nn.Conv1d(self.c_in_conv, self.c_out_conv, padding = 1, kernel_size=4, stride=2)
        self.layer2 = nn.ModuleList()  # conv1d
        self.layer2.append(Node2NodeLayer(self.c_in_node, self.c_out_node))
        self.layer2.append(Node2EdgeLayer(self.c_in_node, self.c_in_edge, self.c_out_edge))
        self.layer3 = nn.ModuleList()
        self.layer3.append(Node2NodeLayer(self.c_in_node, self.c_out_node))
        self.layer3.append(Edge2NodeLayer(self.c_in_edge, self.c_in_node, self.c_out_node))
        self.layer3.append(Node2EdgeLayer(self.c_in_node, self.c_in_edge, self.c_out_edge))
        self.layer3.append(nn.Linear(self.c_out_node*2, self.c_out_node))
        self.layer3.append(nn.ReLU())
        self.layer4 = nn.Linear(self.c_in_conv, 1)

    def forward(self, input):

        input = input.permute(1, 2, 0) # (num_joint, xyz+1, batch_size)
        input = self.layer1(input) # (num_joint, feature_size, batch_size/2)
        input = input.permute(2, 0, 1)  # (batch_size/2, num_joint, feature_size+1)

        batch_size, num_joints, _ = input.shape
        offset = input[:, :, -1]
        input = input[:, :, :-1]
        adj_matrix = torch.cat(batch_size*[self.adj_matrix]).reshape((batch_size, self.len, self.len))
        
        # first layer
        input = self.layer2[0](input, adj_matrix) # (batch_size, num_joint, xyz)
        offset = self.layer2[1](input, offset, adj_matrix) # (batch_size, num_joint, offset)
        
        #seconde layer
        feature_node1 = self.layer3[0](input, adj_matrix) # (batch_size, num_joint, xyz)
        feature_node2 = self.layer3[1](input, offset, adj_matrix) # (batch_size, num_joint, xyz)
        offset = self.layer3[2](input, offset, adj_matrix) # (batch_size, num_joint, offset)
        input = torch.concat([feature_node1, feature_node2], dim = 2)  # (batch_size, num_joint, xyz*2)
        input = input.reshape(batch_size * num_joints, -1)
        input = self.layer3[4](self.layer3[3](input))
        # print("input shape: ", input.shape)
        input = input.reshape(batch_size, num_joints, self.c_out_node) # (batch_size, num_joint, xyz)
        input = torch.concat([offset, input], dim = 2) # (batch_size, num_joint, xyz + offset)
        input = torch.concat([offset, input], dim = 2).reshape(-1, self.c_in_conv) # (batch_size, num_joint, xyz + offset)
        input = self.layer4(input)
        input = torch.sigmoid(input) 
        return input