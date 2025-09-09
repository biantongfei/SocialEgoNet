import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from constants import coco_body_point_num, head_point_num, hands_point_num


def filter_not_interacting_sample(att_y_true, att_y_output):
    mask = (att_y_true != 2)
    return att_y_true[mask], att_y_output[mask]


class JPL_Social_Dataset(Dataset):
    def __init__(self, data_path, sequence_length):
        super().__init__()
        self.files = os.listdir(data_path)
        self.files = [i for i in self.files if 'ori_' in i]
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.features, self.labels = [], []
        for file in self.files: self.get_graph_data_from_file(file)

    def get_graph_data_from_file(self, file):
        with open(os.path.join(self.data_path, file), 'r') as f:
            feature_json = json.load(f)
        if not feature_json['frames']:
            return None
        first_id = feature_json['frames'][0]['frame_id'] if feature_json['frames'] else -1
        if first_id == -1:
            return
        frame_width, frame_height = feature_json['frame_size'][0], feature_json['frame_size'][1]
        video_frame_num = len(feature_json['frames'])
        index, frame_num = 0, 0
        x_tensor = torch.empty((self.sequence_length, coco_body_point_num + head_point_num + hands_point_num, 3))
        while frame_num < self.sequence_length:
            if index == video_frame_num:
                x_tensor[frame_num] = x
                frame_num += 1
            else:
                frame = feature_json['frames'][index]
                if frame['frame_id'] > first_id + frame_num:
                    x_tensor[frame_num] = x
                    frame_num += 1
                else:
                    index += 1
                    if frame['frame_id'] - first_id > self.sequence_length:
                        break
                    else:
                        frame_feature = np.array(frame['keypoints'])
                        frame_feature[:, :2] = 2 * (frame_feature[:, :2] / [frame_width, frame_height] - 0.5)
                        x = torch.tensor(frame_feature)
                        x_tensor[frame_num] = x
                        frame_num += 1
        if frame_num == 0:
            return
        label = feature_json['intention_class'], feature_json['attitude_class'], feature_json['action_class']
        self.features.append(x_tensor)
        self.labels.append(label)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def __len__(self):
        return len(self.features)
