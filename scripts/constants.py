import torch
import numpy as np
import random

batch_size = 128
learning_rate = 1e-2


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device('cpu')
dtype = torch.float32
intention_classes = ['Interacting', 'Interested', 'Not_Interested']
attitude_classes = ['Positive', 'Negative', 'Not_Interacting']
action_classes = ['Handshake', 'Hug', 'Pet', 'Wave', 'Punch', 'Throw', 'Point', 'Gaze', 'Leave',
                  'No_Response']
coco_body_point_num = 23
face_point_num = 68
hands_point_num = 42
jpl_video_fps = 30

coco_body_l_pair = [[0, 1], [0, 2], [1, 3], [2, 4],  # Head
                    [5, 7], [7, 9], [6, 8], [8, 10],  # Body
                    [5, 6], [11, 12], [5, 11], [6, 12],
                    [0, 5], [0, 6],
                    [11, 13], [12, 14], [13, 15], [14, 16],
                    [15, 17], [15, 18], [15, 19], [16, 20], [16, 21], [16, 22]]
face_l_pair = [[23, 24], [24, 25], [25, 26], [26, 27], [27, 28], [28, 29], [29, 30], [30, 31], [31, 32],
               [32, 33], [33, 34], [34, 35], [35, 36], [36, 37], [37, 38], [38, 39],  # jawline
               [40, 41], [41, 42], [42, 43], [43, 44], [44, 50],  # right eyebrow
               [45, 46], [46, 47], [47, 48], [48, 49], [45, 50],  # left eyebrow
               [50, 51], [51, 52], [52, 53], [53, 56], [54, 55], [55, 56], [56, 57], [57, 58],  # nose
               [59, 60], [60, 61], [61, 62], [62, 63], [63, 64], [64, 59],  # right eye
               [65, 66], [66, 67], [67, 68], [68, 69], [69, 70], [70, 65],  # left eye
               [71, 72], [72, 73], [73, 74], [74, 75], [75, 76], [76, 77], [77, 78], [78, 79], [79, 80], [80, 81],
               [81, 82], [82, 71], [71, 83], [82, 83], [83, 84], [84, 85], [85, 86], [86, 87], [87, 77],
               [87, 88], [88, 89], [89, 90], [90, 83]]
hand_l_pair = [[91, 92], [92, 93], [93, 94], [94, 95], [91, 96], [96, 97], [97, 98], [98, 99], [91, 100],
               [100, 101], [101, 102],
               [102, 103], [91, 104], [104, 105], [105, 106], [106, 107], [91, 108], [108, 109], [109, 110],
               [110, 111], [112, 113],
               [113, 114], [114, 115], [115, 116], [112, 117], [117, 118], [118, 119], [119, 120], [112, 121],
               [121, 122], [122, 123],
               [123, 124], [112, 125], [125, 126], [126, 127], [127, 128], [112, 129], [129, 130], [130, 131],
               [131, 132]]
