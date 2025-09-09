from torch.utils.data import DataLoader
import torch
from torch_geometric.data import Data, Batch

from constants import coco_body_point_num, head_point_num, coco_body_l_pair, head_l_pair, hand_l_pair, device, dtype

body_edge_index = torch.Tensor(coco_body_l_pair).t().to(dtype=torch.int64)
head_edge_index = torch.Tensor(head_l_pair).t().to(dtype=torch.int64) - coco_body_point_num
hands_edge_index = torch.Tensor(hand_l_pair).t().to(dtype=torch.int64) - (
        coco_body_point_num + head_point_num)
body_l_pair_num = len(coco_body_l_pair)
head_l_pair_num = len(head_l_pair)
hand_l_pair_num = len(hand_l_pair)


class WholebodyPoseData:
    def __init__(self):
        super().__init__()
        self.body = self.head = self.hands = None


class JPL_Social_DataLoader(DataLoader):
    def __init__(self, dataset, batch_size, sequence_length):
        super().__init__(dataset=dataset, batch_size=batch_size, num_workers=0, collate_fn=self.gcn_collate_fn,
                         pin_memory=True)
        self.sequence_length = sequence_length

    def gcn_collate_fn(self, data):
        body_pose_graph_data = []
        head_pose_graph_data = []
        hand_pose_graph_data = []
        int_label = []
        att_label = []
        act_label = []
        for d in data:
            for i in range(self.sequence_length):
                frame = d[0][i]
                body_pose_graph_data.append(
                    Data(x=frame[:coco_body_point_num].to(dtype), edge_index=body_edge_index))
                head_pose_graph_data.append(
                    Data(x=frame[coco_body_point_num:coco_body_point_num + head_point_num].to(dtype),
                         edge_index=head_edge_index))
                hand_pose_graph_data.append(
                    Data(x=frame[coco_body_point_num + head_point_num:].to(dtype), edge_index=hands_edge_index))
            int_label.append(d[1][0])
            att_label.append(d[1][1])
            act_label.append(d[1][2])
        whole_body_pose_data = WholebodyPoseData()
        whole_body_pose_data.body = Batch.from_data_list(body_pose_graph_data)
        whole_body_pose_data.head = Batch.from_data_list(head_pose_graph_data)
        whole_body_pose_data.hands = Batch.from_data_list(hand_pose_graph_data)
        labels = (
            torch.Tensor(int_label).to(dtype), torch.Tensor(att_label).to(dtype), torch.Tensor(act_label).to(dtype))
        return whole_body_pose_data, labels
