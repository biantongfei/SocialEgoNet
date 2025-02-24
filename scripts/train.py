from Models import SocialEgoNet
from DataLoader import JPL_P4S_DataLoader
from constants import device
from data import JPLP4S_Dataset, filter_not_interacting_sample

import torch
from torch.nn import functional
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import f1_score
from tqdm import tqdm
import argparse
import yaml

parser = argparse.ArgumentParser(description='Train SocialEgoNet on JPL-P4S')
parser.add_argument('--cfg', type=str, required=True)

args = parser.parse_args()

with open(args.cfg, "r") as f:
    config = yaml.safe_load(f)

data_path = config['data']['path']
sequence_length = config['data']['sequence_length']
# trainset = JPLP4S_Dataset(data_path + 'train/', sequence_length)
# valset = JPLP4S_Dataset(data_path + 'validation/', sequence_length)
# testset = JPLP4S_Dataset(data_path + 'test/', sequence_length)

trainset = JPLP4S_Dataset(data_path + 'mixed/coco_wholebody/', sequence_length)
# valset = JPLP4S_Dataset(data_path + 'mixed/coco_wholebody/', sequence_length)
# testset = JPLP4S_Dataset(data_path + 'mixed/coco_wholebody/', sequence_length)
print(len(trainset))

batch_size = config['train']['batch_size']
socialegonet = SocialEgoNet(sequence_length=sequence_length,
                            gcn_hidden_dim=config['model']['gcn_hidden_dim'],
                            gcn_num_layers=config['model']['gcn_num_layers'],
                            lstm_hidden_dim=config['model']['lstm_hidden_dim'],
                            lstm_num_layers=config['model']['lstm_num_layers'],
                            fc1_hidden=config['model']['fc1_hidden'],
                            fc2_hidden=config['model']['fc2_hidden'],
                            )
socialegonet.to(device)
if config['train']['optimizer'] == 'adam':
    optimizer = torch.optim.Adam(socialegonet.parameters(), lr=config['train']['learning_rate'])
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
train_loader = JPL_P4S_DataLoader(dataset=trainset, batch_size=batch_size, sequence_length=sequence_length)
val_loader = JPL_P4S_DataLoader(dataset=trainset, batch_size=batch_size, sequence_length=sequence_length)
for epoch in range(config['train']['num_epochs']):
    socialegonet.train()
    print('Training')
    train_loader = tqdm(train_loader, dynamic_ncols=True)
    for inputs, (int_labels, att_labels, act_labels) in train_loader:
        int_labels, att_labels, act_labels = int_labels.to(dtype=torch.long, device=device), att_labels.to(
            dtype=torch.long, device=device), act_labels.to(dtype=torch.long, device=device)
        int_outputs, att_outputs, act_outputs = socialegonet(inputs)
        loss_1 = functional.cross_entropy(int_outputs, int_labels)
        loss_2 = functional.cross_entropy(att_outputs, att_labels)
        loss_3 = functional.cross_entropy(act_outputs, act_labels)
        total_loss = loss_1 + loss_2 + loss_3
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()
    print('Validating')
    int_y_true, int_y_pred, att_y_true, att_y_pred, act_y_true, act_y_pred = [], [], [], [], [], []
    socialegonet.eval()
    for inputs, (int_labels, att_labels, act_labels) in val_loader:
        int_labels, att_labels, act_labels = int_labels.to(dtype=torch.int64, device=device), att_labels.to(
            dtype=torch.int64, device=device), act_labels.to(dtype=torch.int64, device=device)
        int_outputs, att_outputs, act_outputs = socialegonet(inputs)
        int_outputs = torch.softmax(int_outputs, dim=1)
        _, pred = torch.max(int_outputs, dim=1)
        int_y_true += int_labels.tolist()
        int_y_pred += pred.tolist()
        att_outputs = torch.softmax(att_outputs, dim=1)
        att_labels, att_outputs = filter_not_interacting_sample(att_labels, att_outputs)
        _, pred = torch.max(att_outputs, dim=1)
        att_y_true += att_labels.tolist()
        att_y_pred += pred.tolist()
        act_outputs = torch.softmax(act_outputs, dim=1)
        _, pred = torch.max(act_outputs, dim=1)
        act_y_true += act_labels.tolist()
        act_y_pred += pred.tolist()
    result_str = 'validating--> epoch: %d, ' % epoch
    int_y_true, int_y_pred = torch.Tensor(int_y_true), torch.Tensor(int_y_pred)
    int_acc = int_y_pred.eq(int_y_true).sum().float().item() / int_y_pred.size(dim=0)
    int_f1 = f1_score(int_y_true, int_y_pred, average='weighted')
    result_str += 'int_acc: %.2f, int_f1: %.2f, ' % (int_acc * 100, int_f1 * 100)
    att_y_true, att_y_pred = torch.Tensor(att_y_true), torch.Tensor(att_y_pred)
    att_acc = att_y_pred.eq(att_y_true).sum().float().item() / att_y_pred.size(dim=0)
    att_f1 = f1_score(att_y_true, att_y_pred, average='weighted')
    result_str += 'att_acc: %.2f, att_f1: %.2f, ' % (att_acc * 100, att_f1 * 100)
    act_y_true, act_y_pred = torch.Tensor(act_y_true), torch.Tensor(act_y_pred)
    act_acc = act_y_pred.eq(act_y_true).sum().float().item() / act_y_pred.size(dim=0)
    act_f1 = f1_score(act_y_true, act_y_pred, average='weighted')
    result_str += 'act_acc: %.2f%%, act_f1: %.2f ' % (act_acc * 100, act_f1 * 100)
    print(result_str)

print('Testing')
test_loader = JPL_P4S_DataLoader(dataset=trainset, sequence_length=sequence_length, batch_size=batch_size)
int_y_true, int_y_pred, att_y_true, att_y_pred, act_y_true, act_y_pred = [], [], [], [], [], []
socialegonet.eval()
for inputs, (int_labels, att_labels, act_labels) in test_loader:
    int_labels, att_labels, act_labels = int_labels.to(device), att_labels.to(device), act_labels.to(device)
    int_outputs, att_outputs, act_outputs = socialegonet(inputs)
    int_outputs = torch.softmax(int_outputs, dim=1)
    _, pred = torch.max(int_outputs, dim=1)
    int_y_true += int_labels.tolist()
    int_y_pred += pred.tolist()
    att_outputs = torch.softmax(att_outputs, dim=1)
    att_labels, att_outputs = filter_not_interacting_sample(att_labels, att_outputs)
    _, pred = torch.max(att_outputs, dim=1)
    att_y_true += att_labels.tolist()
    att_y_pred += pred.tolist()
    act_outputs = torch.softmax(act_outputs, dim=1)
    _, pred = torch.max(act_outputs, dim=1)
    act_y_true += act_labels.tolist()
    act_y_pred += pred.tolist()
result_str = ''
int_y_true, int_y_pred = torch.Tensor(int_y_true), torch.Tensor(int_y_pred)
int_acc = int_y_pred.eq(int_y_true).sum().float().item() / int_y_pred.size(dim=0)
int_f1 = f1_score(int_y_true, int_y_pred, average='weighted')
result_str += 'int_acc: %.2f, int_f1: %.2f, ' % (int_acc * 100, int_f1 * 100)
att_y_true, att_y_pred = torch.Tensor(att_y_true), torch.Tensor(att_y_pred)
att_acc = att_y_pred.eq(att_y_true).sum().float().item() / att_y_pred.size(dim=0)
att_f1 = f1_score(att_y_true, att_y_pred, average='weighted')
result_str += 'att_acc: %.2f, att_f1: %.2f, ' % (att_acc * 100, att_f1 * 100)
act_y_true, act_y_pred = torch.Tensor(act_y_true), torch.Tensor(act_y_pred)
act_acc = act_y_pred.eq(act_y_true).sum().float().item() / act_y_pred.size(dim=0)
act_f1 = f1_score(act_y_true, act_y_pred, average='weighted')
result_str += 'act_acc: %.2f, act_f1: %.2f, ' % (act_acc * 100, act_f1 * 100)
