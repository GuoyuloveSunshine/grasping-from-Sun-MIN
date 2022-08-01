from turtle import forward
import torch
from torch import nn
from models.skeleton_gnn import Node2EdgeLayer, Edge2NodeLayer, Node2NodeLayer

"""
Inputs:
    node_feats - Tensor with node features of shape [batch_size, num_joints, c_in]-N*23*3   - normalized position information
    edge_feats - Tensor with edge features of shape [batch_size, num_offset, features]- N*23*1  - normalized offset information
    adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections.
                Shape: [batch_size, num_nodes, num_nodes]

    input - form [batch_size, num_joint, c_in] - N * num_joint *(x, y, z)
    offset - form [batch_size, num_joint, joint_len] - N * num_joint * offset
"""
def calculate_adj_matrix(topology):
    adj = torch.zeros((len(topology), len(topology)))
    for i,j in enumerate(topology):
        if i == 0: #the root not in
            continue
        adj[j,i] = 1
    return adj

class Encoder(nn.Module):
    def __init__(self, args, topology):
        super(Encoder, self).__init__()
        self.c_in_node = args.position_feature
        self.c_out_node = args.position_feature
        self.c_in_edge = args.offset_feature
        self.c_out_edge = args.offset_feature
        self.c_in_conv = args.position_feature + args.offset_feature
        self.c_out_conv = args.position_feature + args.offset_feature
        self.adj_matrix = calculate_adj_matrix(topology)
        self.len = len(topology)
        # network module
        self.layers = nn.ModuleList()
        self.module = nn.ModuleList()
        self.module.append(Node2NodeLayer(self.c_in_node, self.c_out_node))
        self.module.append(Edge2NodeLayer(self.c_in_edge, self.c_in_node, self.c_out_node))
        self.module.append(Node2EdgeLayer(self.c_in_node, self.c_in_edge, self.c_out_edge))
        self.module.append(nn.Linear(self.c_out_node*2, self.c_out_node))
        self.module2 = nn.ModuleList()
        self.module2.append(Node2NodeLayer(self.c_in_node, self.c_out_node))
        self.module2.append(Edge2NodeLayer(self.c_in_edge, self.c_in_node, self.c_out_node))
        self.module2.append(Node2EdgeLayer(self.c_in_node, self.c_in_edge, self.c_out_edge))
        self.module2.append(nn.Linear(self.c_out_node*2, self.c_out_node))

        # network layers
        self.layers.append(nn.Conv1d(self.c_in_conv, self.c_out_conv, padding = 1, kernel_size=4, stride=2)) # kernel_size = 4
        self.layers.append(self.module)
        self.layers.append(nn.Conv1d(self.c_in_conv, self.c_out_conv, padding = 1, kernel_size=4, stride=2))
        self.layers.append(self.module2)
        self.layers.append(nn.Conv1d(self.c_in_conv, self.c_out_conv, padding = 1, kernel_size=4, stride=2))

    def forward(self, input, offset):
        # print("input: ", input.shape)
        # print("offset: ", offset.shape)
        for i, layer in enumerate(self.layers):
            if i%2 ==0:
                input = torch.concat([input, offset], dim = 2) # (batch_size, num_joint, xyz + length)
                input = input.permute(1, 2, 0) # (num_joint, feature_size, batch_size)
                input = layer(input) # (num_joint, feature_size, batch_size/2)
                input = input.permute(2, 0, 1)  # (batch_size/2, num_joint, feature_size)
            else:
                # print(i, input.shape)
                input_node = input[:, :, :-1] # (batch_size/2, num_joint, xyz)
                input_edge = input[:, :, -1] # (batch_size/2, num_joint, offset)
                batch_size, num_joints, _ = input.shape
                # print(self.adj_matrix)
                adj_matrix = torch.cat(batch_size*[self.adj_matrix]).reshape((batch_size, self.len, self.len))
                # print(adj_matrix.shape)
                feature_node1 = layer[0](input_node, adj_matrix) # (batch_size/2, num_joint, xyz)
                feature_node2 = layer[1](input_node, input_edge, adj_matrix) # (batch_size/2, num_joint, xyz)

                offset = layer[2](input_node, input_edge, adj_matrix) # (batch_size/2, num_joint, offset)
                input = torch.concat([feature_node1, feature_node2], dim = 2)  # (batch_size/2, num_joint, xyz*2)
                
                input = input.reshape(batch_size * num_joints, -1)
                input = layer[3](input).reshape(batch_size, num_joints, self.c_out_node) # (batch_size/2, num_joint, xyz) 
        if input.shape[2] == self.c_out_conv:
            # print(input.shape)
            input = input[:, :, :-self.c_out_edge]
        return input

class Decoder(nn.Module):
    def __init__(self, args, topology):
        super(Decoder, self).__init__()
        self.c_in_node = args.position_feature
        self.c_out_node = args.position_feature
        self.c_in_edge = args.offset_feature
        self.c_out_edge = args.offset_feature
        self.c_in_conv = args.position_feature + args.offset_feature
        self.c_out_conv = args.position_feature + args.offset_feature
        self.adj_matrix = calculate_adj_matrix(topology)
        self.len = len(topology)
        # network module
        self.layers = nn.ModuleList()
        self.module = nn.ModuleList()
        self.module.append(Node2NodeLayer(self.c_in_node, self.c_out_node))
        self.module.append(Edge2NodeLayer(self.c_in_edge, self.c_in_node, self.c_out_node))
        self.module.append(Node2EdgeLayer(self.c_in_node, self.c_in_edge, self.c_out_edge))
        self.module.append(nn.Linear(self.c_out_node*2, self.c_out_node))
        self.module2 = nn.ModuleList()
        self.module2.append(Node2NodeLayer(self.c_in_node, self.c_out_node))
        self.module2.append(Edge2NodeLayer(self.c_in_edge, self.c_in_node, self.c_out_node))
        self.module2.append(Node2EdgeLayer(self.c_in_node, self.c_in_edge, self.c_out_edge))
        self.module2.append(nn.Linear(self.c_out_node*2, self.c_out_node))

        # network layers
        self.layers.append(nn.ConvTranspose1d(self.c_in_conv, self.c_out_conv, padding = 1, kernel_size=4, stride=2)) # kernel
        self.layers.append(self.module)
        self.layers.append(nn.ConvTranspose1d(self.c_in_conv, self.c_out_conv, padding = 1, kernel_size=4, stride=2))
        self.layers.append(self.module2)
        self.layers.append(nn.ConvTranspose1d(self.c_in_conv, self.c_out_conv, padding = 1, kernel_size=4, stride=2))

    def forward(self, input, offset):
        for i, layer in enumerate(self.layers):
            if i%2 ==0:
                input = torch.concat([input, offset], dim = 2) # (batch_size, num_joint, xyz + length)
                input = input.permute(1, 2, 0) # (num_joint, feature_size, batch_size)
                input = layer(input) # (num_joint, feature_size, batch_size/2)
                input = input.permute(2, 0, 1)  # (batch_size/2, num_joint, feature_size)
            else:
                # print(i, layer)
                input_node = input[:, :, :-1] # (batch_size/2, num_joint, xyz)
                input_edge = input[:, :, -1] # (batch_size/2, num_joint, offset)
                batch_size, num_joints, _ = input.shape

                adj_matrix = torch.cat(batch_size*[self.adj_matrix]).reshape((batch_size, self.len, self.len))
                feature_node1 = layer[0](input_node, adj_matrix) # (batch_size/2, num_joint, xyz)
                feature_node2 = layer[1](input_node, input_edge, adj_matrix) # (batch_size/2, num_joint, xyz)

                offset = layer[2](input_node, input_edge, adj_matrix) # (batch_size/2, num_joint, offset)
                input = torch.concat([feature_node1, feature_node2], dim = 2)  # (batch_size/2, num_joint, xyz*2)
                
                input = input.reshape(batch_size * num_joints, -1)
                input = layer[3](input).reshape(batch_size, num_joints, self.c_out_node) # (batch_size/2, num_joint, xyz) 
        if input.shape[2] == self.c_out_conv:
            input = input[:, :, :-self.c_out_edge]
        return input

