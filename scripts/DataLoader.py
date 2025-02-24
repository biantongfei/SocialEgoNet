from sympy.categories import Object
from torch.utils.data import DataLoader
import torch
from torch_geometric.data import Data, Batch

from constants import coco_body_point_num, face_point_num, coco_body_l_pair, face_l_pair, hand_l_pair, device, dtype

body_edge_index = torch.Tensor(coco_body_l_pair).t().to(dtype=torch.int64, device=device)
face_edge_index = torch.Tensor(face_l_pair).t().to(dtype=torch.int64, device=device)
hands_edge_index = torch.Tensor(hand_l_pair).t().to(dtype=torch.int64, device=device)
body_l_pair_num = len(coco_body_l_pair)
head_l_pair_num = len(face_l_pair)
hand_l_pair_num = len(hand_l_pair)


class WholebodyPoseData(Object):
    def __init__(self, **kwargs):
        super().__init__()
        self.body = None
        self.face = None
        self.hands = None


class JPL_P4S_DataLoader(DataLoader):
    def __init__(self, dataset, batch_size, sequence_length):
        super().__init__(dataset=dataset, batch_size=batch_size, num_workers=0)
        self.sequence_length = sequence_length
        self.collate_fn = self.gcn_collate_fn


    def gcn_collate_fn(self, data):
        body_pose_graph_data = []
        face_pose_graph_data = []
        hand_pose_graph_data = []
        int_label = []
        att_label = []
        act_label = []
        for d in data:
            for i in d[0].size[0]:
                body_pose_graph_data.append(
                    Data(x=d[0][i][:coco_body_point_num].to(dtype=dtype, device=device), edge_index=body_edge_index))
                face_pose_graph_data.append(Data(
                    x=d[0][i][coco_body_point_num:coco_body_point_num + face_point_num].to(dtype=dtype, device=device),
                    edge_index=face_edge_index))
                hand_pose_graph_data.append(
                    Data(x=d[0][i][coco_body_point_num + face_point_num:].to(dtype=dtype, device=device),
                         edge_index=hands_edge_index))
            int_label.append(d[1][0])
            att_label.append(d[1][1])
            act_label.append(d[1][2])
        whole_body_pose_data = WholebodyPoseData()
        whole_body_pose_data.body = Batch.from_data_list(body_pose_graph_data)
        whole_body_pose_data.face = Batch.from_data_list(face_pose_graph_data)
        whole_body_pose_data.hand = Batch.from_data_list(hand_pose_graph_data)
        labels = (
            torch.Tensor(int_label).to(device), torch.Tensor(att_label).to(device), torch.Tensor(act_label).to(device))
        return whole_body_pose_data, labels
