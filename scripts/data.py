import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from constants import coco_body_point_num, face_point_num, hands_point_num


def get_first_id(feature_json):
    for frame in feature_json['frames']:
        return frame['frame_id']
    return -1


def filter_not_interacting_sample(att_y_true, att_y_output):
    mask = (att_y_true != 2)
    return att_y_true[mask], att_y_output[mask]


class JPLP4S_Dataset(Dataset):
    def __init__(self, data_path, sequence_length):
        super().__init__()
        self.files = os.listdir(data_path)
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.features, self.labels = [], []
        for file in self.files:
            self.get_graph_data_from_file(file)

    def get_graph_data_from_file(self, file):
        if file.endswith('json'):
            with open(self.data_path + file, 'r') as f:
                print(file)
                feature_json = json.load(f)
                f.close()
            frame_width, frame_height = feature_json['frame_size'][0], feature_json['frame_size'][1]
            video_frame_num = len(feature_json['frames'])
            if video_frame_num == 0:
                return
            first_id = get_first_id(feature_json)
            if first_id == -1:
                return
            index = 0
            x_tensor = torch.empty((self.sequence_length, coco_body_point_num + face_point_num + hands_point_num, 3))
            frame_num = 0
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
                            frame_feature[:, 0] = 2 * (frame_feature[:, 0] / frame_width - 0.5)
                            frame_feature[:, 1] = 2 * (frame_feature[:, 1] / frame_height - 0.5)
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
