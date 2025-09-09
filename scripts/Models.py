import torch
from torch import nn
from torch_geometric.nn import GCN

from constants import (intention_classes, attitude_classes, action_classes, coco_body_point_num, face_point_num,
                       hands_point_num)

intention_class_num = len(intention_classes)
attitude_class_num = len(attitude_classes)
action_class_num = len(action_classes)


class ChainClassifier(nn.Module):
    def __init__(self, in_feature_size):
        super().__init__()
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
            nn.Linear(in_feature_size + intention_class_num + attitude_class_num, action_class_num)
        )

    def forward(self, y):
        y1 = self.intention_head(y)
        y2 = self.attitude_head(torch.cat((y, y1), dim=1))
        y3 = self.action_head(torch.cat((y, y1, y2), dim=1))
        return y1, y2, y3


class SocialEgoNet(nn.Module):
    def __init__(self,
                 sequence_length,
                 gcn_hidden_dim,
                 gcn_num_layers,
                 lstm_hidden_dim,
                 lstm_num_layers,
                 fc1_hidden,
                 fc2_hidden):
        super().__init__()
        self.sequence_length = sequence_length
        self.gcn_hidden_dim = gcn_hidden_dim
        self._init_gcn(gcn_num_layers)
        self.gcn_output_dim = gcn_hidden_dim * (coco_body_point_num + face_point_num + hands_point_num)
        self.gcn_attention = nn.Linear(self.gcn_output_dim, 1)
        self.time_model = nn.LSTM(self.gcn_output_dim, hidden_size=lstm_hidden_dim, num_layers=lstm_num_layers,
                            bidirectional=True, batch_first=True)
        self.lstm_attention = nn.Linear(lstm_hidden_dim * 2, 1)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, fc1_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(fc1_hidden),
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(fc2_hidden),
        )
        self.classifier = ChainClassifier(fc2_hidden)

    def _init_gcn(self, gcn_num_layers):
        self.GCN_body = GCN(in_channels=3, hidden_channels=self.gcn_hidden_dim, num_layers=gcn_num_layers)
        self.GCN_head = GCN(in_channels=3, hidden_channels=self.gcn_hidden_dim, num_layers=gcn_num_layers)
        self.GCN_hand = GCN(in_channels=3, hidden_channels=self.gcn_hidden_dim, num_layers=gcn_num_layers)

    def _apply_gcn(self, gcn, x, edge_index, batch, num_points):
        return gcn(x=x, edge_index=edge_index, batch=batch).view(-1, self.sequence_length,
                                                                 self.gcn_hidden_dim * num_points)

    def forward(self, data):
        print(data.body.x.device)
        print(data.body.edge_index.shape)
        print(data.body.batch.shape)
        x_body = self._apply_gcn(self.GCN_body, data.body.x, data.body.edge_index, data.body.batch, coco_body_point_num)
        x_face = self._apply_gcn(self.GCN_face, data.face.x, data.face.edge_index, data.face.batch, face_point_num)
        x_hand = self._apply_gcn(self.GCN_hand, data.hands.x, data.hands.edge_index, data.hands.batch, hands_point_num)
        x = torch.cat([x_body, x_face, x_hand], dim=2)
        x = x.view(-1, self.sequence_length, self.gcn_output_dim)
        x = x * nn.Softmax(dim=1)(self.gcn_attention(x))
        on, _ = self.lstm(x)
        on = on.view(on.shape[0], on.shape[1], 2, -1)
        x = (torch.cat([on[:, :, 0, :], on[:, :, 1, :]], dim=-1))
        x = torch.sum(x * nn.Softmax(dim=1)(self.lstm_attention(x)), dim=1)
        x = self.fc(x)
        x = self.classifier(x)
        return x
