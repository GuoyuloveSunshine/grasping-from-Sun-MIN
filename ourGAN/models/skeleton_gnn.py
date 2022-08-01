
import torch
from torch import nn


"""
Inputs:
    node_feats - Tensor with node features of shape [batch_size, num_joints, c_in]-N*23*3   - normalized position information
    edge_feats - Tensor with edge features of shape [batch_size, num_offset + one_base, features]- N*23*1  - normalized offset information
    adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections.
                Shape: [batch_size, num_nodes, num_nodes]
"""

class Node2NodeLayer(nn.Module):
    
    def __init__(self, c_in_node, c_out_node):
        super().__init__()
        self.parent_linear = nn.Linear(c_in_node, c_out_node)
        self.recurrent = nn.Linear(c_in_node, c_out_node)
        self.children_linear = nn.Linear(c_in_node, c_out_node)
        self.c_in_node = c_in_node
        self.c_out_node = c_out_node
        
    def forward(self, node_features, adj_matrix, aggr='mean'):
        # We assume that the diagonal of adj_matrix is empty
        
        # Mean aggregation (try sum maybe ?)
        num_children = adj_matrix.sum(dim=-1, keepdims=True)
        num_parents = adj_matrix.transpose(1, 2).sum(dim=-1, keepdims=True)
        num_neighbours = num_children + num_parents + 1

        batch_size, num_joints, c_in_node = node_features.shape
        node_features = node_features.reshape(batch_size * num_joints, c_in_node)
        
        # Parents
        parent_features = self.parent_linear(node_features).reshape(batch_size, num_joints, self.c_out_node)
        parent_features = torch.bmm(adj_matrix.transpose(1, 2), parent_features)
            
        # Recurrent
        recurrent_features = self.recurrent(node_features).reshape(batch_size, num_joints, self.c_out_node)
        
        # Children
        children_features = self.children_linear(node_features).reshape(batch_size, num_joints, self.c_out_node)
        children_features = torch.bmm(adj_matrix, children_features)

        # Mean features (possible improvements : other aggregation, weighted sum)
        if aggr=='mean':
            node_features = (parent_features + children_features + recurrent_features) / num_neighbours
        elif aggr=='sum':
            node_features = parent_features + children_features + recurrent_features
        
        # Possibly other activation function
        node_features = torch.sigmoid(node_features)
        
        # shape (batch_size, num_joint, c_out_node)
        return node_features

class Node2EdgeLayer(nn.Module):
    
    def __init__(self, c_in_node, c_in_edge, c_out_edge):
        super().__init__()
        
        self.parent_linear = nn.Linear(c_in_node, c_out_edge)
        self.recurrent = nn.Linear(c_in_edge, c_out_edge)
        self.children_linear = nn.Linear(c_in_node, c_out_edge)
        
        self.c_in_node = c_in_node
        self.c_in_edge = c_in_edge
        self.c_out_edge = c_out_edge
        
    def forward(self, node_features, edge_features, adj_matrix, aggr='mean'):
        
        # node_features [batch_size, num_joints, c_in_node]
        # edge_features [batch_size, num_joints, c_in_edge]
        
        # Each node has only one parent edge (parent_edge_feautres)
        # Nodes can have several children edges bmm(adj, children_edge_features)
        
        # For each node, the edge of same index is the parent edge
        # For each node, the adjacency matrix gives the indices of children nodes
        
        # Mean aggregation (try sum maybe ?)
        num_children = adj_matrix.sum(dim=-1, keepdims=True)
        num_parents = adj_matrix.transpose(1, 2).sum(dim=-1, keepdims=True)
        num_neighbours = num_children + num_parents + 1

        batch_size, num_joints, _ = node_features.shape
        node_features = node_features.reshape(batch_size * num_joints, self.c_in_node)
        edge_features = edge_features.reshape(batch_size * num_joints, self.c_in_edge)
        
        # Children
        children_features = self.children_linear(node_features).reshape(batch_size, num_joints, self.c_out_edge)

        # Recurrent
        recurrent_features = self.recurrent(edge_features).reshape(batch_size, num_joints, self.c_out_edge)
        
        # Parents
        parent_features = self.parent_linear(node_features).reshape(batch_size, num_joints, self.c_out_edge)
        parent_features = torch.bmm(adj_matrix.transpose(1, 2), parent_features)

        # Mean features (possible improvements : other aggregation, weighted sum)
        if aggr=='mean':
            edge_features = (parent_features + children_features + recurrent_features) / 3
        elif aggr=='sum':
            edge_features = parent_features + children_features + recurrent_features
        
        # Possibly other activation function
        edge_features = torch.sigmoid(edge_features)
        
        # shape (batch_size, num_joint, c_out_edge)
        return edge_features

class Edge2NodeLayer(nn.Module):
    def __init__(self, c_in_edge, c_in_node, c_out_node):
        super().__init__()
        self.c_in_edge = c_in_edge
        self.c_in_node = c_in_node
        self.c_out_node = c_out_node
        
        self.parent_linear = nn.Linear(c_in_edge, c_out_node)
        self.recurrent = nn.Linear(c_in_node, c_out_node)
        self.children_linear = nn.Linear(c_in_edge, c_out_node)
        
    def forward(self, node_features, edge_features, adj_matrix, aggr='mean'):
        
        # Mean aggregation (try sum maybe ?)
        num_children = adj_matrix.sum(dim=-1, keepdims=True)
        num_parents = adj_matrix.transpose(1, 2).sum(dim=-1, keepdims=True)
        num_neighbours = num_children + num_parents + 1

        batch_size, num_joints, _ = node_features.shape
        node_features = node_features.reshape(batch_size * num_joints, self.c_in_node)
        edge_features = edge_features.reshape(batch_size * num_joints, self.c_in_edge)
        
        # Children
        children_features = self.children_linear(edge_features).reshape(batch_size, num_joints, self.c_out_node)
        children_features = torch.bmm(adj_matrix, children_features)
        # Recurrent
        recurrent_features = self.recurrent(node_features).reshape(batch_size, num_joints, self.c_out_node)
        
        # Parents
        parent_features = self.parent_linear(edge_features).reshape(batch_size, num_joints, self.c_out_node)
        

        # Mean features (possible improvements : other aggregation, weighted sum)
        if aggr=='mean':
            res = (parent_features + children_features + recurrent_features) / num_neighbours
        elif aggr=='sum':
            res = parent_features + children_features + recurrent_features
        
        # Possibly other activation function
        res = torch.sigmoid(res)
        

        # shape (batch_size, num_joint, c_out_node)
        return res