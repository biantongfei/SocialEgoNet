import math
import numpy as np
from collections import OrderedDict
import networkx as nx

import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch_geometric.nn import GCN, GATConv
from torch_geometric.utils import add_self_loops
import torchvision.models.video as models
from torch_geometric.data import Data, Batch

from Dataset import get_inputs_size, coco_body_point_num, head_point_num, hands_point_num, harper_body_point_num, \
    halpe_body_point_num
from graph import Graph, ConvTemporalGraphical
from MSG3D.msg3d import Model as MsG3d
from DGSTGCN.dgstgcn import Model as DG_Model
from constants import intention_classes, attitude_classes, action_classes, device, dtype

intention_class_num = len(intention_classes)
attitude_class_num = len(attitude_classes)
action_class_num = len(action_classes)


class ChainClassifier(nn.Module):
    def __init__(self, framework, in_feature_size=16):
        super(ChainClassifier, self).__init__()
        super().__init__()
        self.framework = framework
        self.intention_head = nn.Sequential(nn.ReLU(),
                                            nn.Linear(in_feature_size, intention_class_num)
                                            )

        self.attitude_head = nn.Sequential(
            nn.BatchNorm1d(in_feature_size + intention_class_num),
            nn.ReLU(),
            nn.Linear(in_feature_size + intention_class_num, attitude_class_num)
        )
        self.action_head = nn.Sequential(
            nn.BatchNorm1d(in_feature_size + intention_class_num + attitude_class_num),
            nn.ReLU(),
            nn.Linear(in_feature_size + intention_class_num + attitude_class_num, jpl_action_class_num)
        )

    def forward(self, y):
        y1 = self.intention_head(y)
        y2 = self.attitude_head(torch.cat((y, y1), dim=1))
        y3 = self.action_head(torch.cat((y, y1, y2), dim=1))
        return y1, y2, y3


class GNN(nn.Module):
    def __init__(self, body_part, framework, sequence_length, frame_sample_hop, keypoint_hidden_dim, time_hidden_dim,
                 fc_hidden1, fc_hidden2):
        super(GNN, self).__init__()
        super().__init__()
        self.body_part = body_part
        self.input_size = get_inputs_size(body_part)
        self.framework = framework
        self.sequence_length = sequence_length
        self.frame_sample_hop = frame_sample_hop
        self.keypoint_hidden_dim = keypoint_hidden_dim
        self.time_hidden_dim = self.keypoint_hidden_dim * time_hidden_dim
        self.fc_hidden1 = fc_hidden1
        self.fc_hidden2 = fc_hidden2
        if body_part[0]:
            self.GCN_body = GCN(in_channels=3, hidden_channels=self.keypoint_hidden_dim, num_layers=3)
        if body_part[1]:
            self.GCN_head = GCN(in_channels=3, hidden_channels=self.keypoint_hidden_dim, num_layers=3)
        if body_part[2]:
            self.GCN_hand = GCN(in_channels=3, hidden_channels=self.keypoint_hidden_dim, num_layers=3)
        self.gcn_attention = nn.Linear(int(self.keypoint_hidden_dim * self.input_size / 3), 1)
        self.lstm = nn.LSTM(math.ceil(self.input_size / 3) * self.keypoint_hidden_dim,
                            hidden_size=self.time_hidden_dim, num_layers=3, bidirectional=True,
                            batch_first=True)
        self.fc_input_size = self.time_hidden_dim * 2
        self.lstm_attention = nn.Linear(self.fc_input_size, 1)
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, self.fc_hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(self.fc_hidden1),
            nn.Linear(self.fc_hidden1, self.fc_hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(self.fc_hidden2),
        )
        self.classifier = ChainClassifier(framework, self.fc_hidden2)

    def forward(self, data):
        x_list = []
        if self.body_part[0]:
            x_body, edge_index_body, batch_body = data[0][0].to(dtype=dtype, device=device), data[1][0].to(
                device=device), data[2][0].to(device)
            x_body = self.GCN_body(x=x_body, edge_index=edge_index_body, batch=batch_body)
            x_body = x_body.view(-1, int(self.sequence_length / self.frame_sample_hop), self.keypoint_hidden_dim * (
                self.body_point_num))
            x_list.append(x_body)
        if self.body_part[1]:
            x_head, edge_index_head, batch_head = data[0][1].to(dtype=dtype, device=device), data[1][1].to(device), \
                data[2][1].to(device)
            x_head = self.GCN_head(x=x_head, edge_index=edge_index_head, batch=batch_head)
            x_head = x_head.view(-1, int(self.sequence_length / self.frame_sample_hop),
                                 self.keypoint_hidden_dim * head_point_num)
            x_list.append(x_head)
        if self.body_part[2]:
            x_hand, edge_index_hand, batch_hand = data[0][2].to(dtype=dtype, device=device), data[1][2].to(device), \
                data[2][2].to(device)
            x_hand = self.GCN_hand(x=x_hand, edge_index=edge_index_hand, batch=batch_hand)
            x_hand = x_hand.view(-1, int(self.sequence_length / self.frame_sample_hop),
                                 self.keypoint_hidden_dim * hands_point_num)
            x_list.append(x_hand)
        x = torch.cat(x_list, dim=2)
        x = x.view(-1, int(self.sequence_length / self.frame_sample_hop),
                   self.keypoint_hidden_dim * int(self.input_size / 3))
        gcn_attention_weights = nn.Softmax(dim=1)(self.gcn_attention(x))
        x = x * gcn_attention_weights

        on, _ = self.time_model(x)
        on = on.view(on.shape[0], on.shape[1], 2, -1)
        x = (torch.cat([on[:, :, 0, :], on[:, :, 1, :]], dim=-1))
        attention_weights = nn.Softmax(dim=1)(self.lstm_attention(x))
        x = torch.sum(x * attention_weights, dim=1)
        x = self.fc(x)
        return self.classifier(x)
