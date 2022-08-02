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
        self.len = len(topology)
        self.device = torch.device(args.cuda_device if (torch.cuda.is_available()) else 'cpu')
        self.c_in_node = args.position_feature
        self.c_out_node = args.position_feature
        self.c_in_edge = args.offset_feature
        self.c_out_edge = args.offset_feature
        self.c_in_conv = (args.position_feature + args.offset_feature)*self.len
        self.c_out_conv = (args.position_feature + args.offset_feature)*self.len
        self.adj_matrix = calculate_adj_matrix(topology).to(self.device)
        
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
        self.layer4 = nn.Linear(self.c_in_conv, 32)
        self.layer5 = nn.ReLU()
        self.layer6 = nn.Linear(32, 1)

    def forward(self, input):
        # print("dis input: ", input.shape)
        # first layer
        batch_size, frame, num_joint, in_feature = input.shape # (batch_size, frame, num_joint, xyz + length)
        input = input.permute(0, 2, 3, 1) # (batch_size, num_joint, feature_size, frame)
        input = input.reshape((batch_size, -1, frame)) # (batch_size, num_joint*feature_size, frame)
        input = self.layer1(input) # (batch_size, num_joint * feature_size, ?)
        input = input.permute(0, 2, 1)  # (batch_size, frame/2, num_joint * feature_size)
        input = input.reshape((batch_size, -1, num_joint, in_feature)) # (batch_size, frame/2, num_joint, feature_size)
        # print("after first layer: ", input.shape)

        # second layer
        offset = input[... , -1].unsqueeze(-1) # (batch_size, frame/2, num_joint, offset)
        input = input[... , :-1] # (batch_size, frame/2, num_joint, xyz)
        batch_size, frame, _,__ = input.shape
        adj_matrix = torch.cat(batch_size*frame*[self.adj_matrix]).reshape((batch_size, frame, self.len, self.len))

        input = self.layer2[0](input, adj_matrix) # (batch_size, frame/2, num_joint, xyz)
        offset = self.layer2[1](input, offset, adj_matrix) # (batch_size, frame/2, num_joint, offset)
    
        # third layer
        feature_node1 = self.layer3[0](input, adj_matrix) # (batch_size, frame/2, num_joint, xyz)
        feature_node2 = self.layer3[1](input, offset, adj_matrix) # (batch_size, frame/2, num_joint, xyz)
        offset = self.layer3[2](input, offset, adj_matrix) # (batch_size, num_joint, offset)
        input = torch.concat([feature_node1, feature_node2], dim = -1)  # (batch_size, frame/2, num_joint, xyz*2)
        input = self.layer3[3](input)
        input = self.layer3[4](input) # (batch_size, frame/2, num_joint, xyz)
        # print("after third layer: ", input.shape)

        input = torch.concat([offset, input], dim = -1) # (batch_size, frame/2, num_joint, xyz + offset)
        input = input.reshape((batch_size, frame, -1)) # (batch_size, frame/2, num_joint * xyz + offset)
        input = self.layer4(input) # (batch_size, frame/2, 32)
        input = self.layer5(input) # ReLU
        input = self.layer6(input)
        input = torch.sigmoid(input) 
        return input