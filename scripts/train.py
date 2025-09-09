from Models import SocialEgoNet
from DataLoader import JPL_Social_DataLoader
from constants import device
from data import JPL_Social_Dataset, filter_not_interacting_sample

import torch
from torch.nn import functional
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import f1_score
from tqdm import tqdm
import argparse
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description='Train SocialEgoNet on JPL-Social')
    parser.add_argument('--cfg', type=str, required=True)
    return parser.parse_args()


def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def get_dataloaders(config):
    data_path = config['data']['path']
    sequence_length = config['data']['sequence_length']
    trainset = JPL_Social_Dataset(data_path + 'train/', sequence_length)
    # valset = JPL_Social_Dataset(data_path + 'validation/', sequence_length)
    valset = JPL_Social_Dataset(data_path + 'test/', sequence_length)
    testset = JPL_Social_Dataset(data_path + 'test/', sequence_length)
    print('JPL_Social Dataset Size, trainset:%d, valset:%d, testset:%d' % (len(trainset), len(valset), len(testset)))

    batch_size = config['train']['batch_size']
    train_loader = JPL_Social_DataLoader(trainset, batch_size=batch_size, sequence_length=sequence_length)
    val_loader = JPL_Social_DataLoader(valset, batch_size=batch_size, sequence_length=sequence_length)
    test_loader = JPL_Social_DataLoader(testset, batch_size=batch_size, sequence_length=sequence_length)

    return train_loader, val_loader, test_loader


def create_model(config):
    model = SocialEgoNet(
        sequence_length=config['data']['sequence_length'],
        gcn_hidden_dim=config['model']['gcn_hidden_dim'],
        gcn_num_layers=config['model']['gcn_num_layers'],
        lstm_hidden_dim=config['model']['lstm_hidden_dim'],
        lstm_num_layers=config['model']['lstm_num_layers'],
        fc1_hidden=config['model']['fc1_hidden'],
        fc2_hidden=config['model']['fc2_hidden'],
    ).to(device)
    return model


@torch.no_grad()
def evaluate_model(epoch, model, dataloader, task):
    model.eval()

    metrics = {"int": {"true": [], "pred": []}, "att": {"true": [], "pred": []}, "act": {"true": [], "pred": []}}

    for inputs, (int_labels, att_labels, act_labels) in tqdm(dataloader, dynamic_ncols=True, desc=task + " Evaluating"):
        int_labels, att_labels, act_labels = int_labels.to(device), att_labels.to(device), act_labels.to(device)
        int_outputs, att_outputs, act_outputs = model(inputs)

        metrics["int"]["true"].extend(int_labels.tolist())
        metrics["int"]["pred"].extend(torch.argmax(torch.softmax(int_outputs, dim=1), dim=1).tolist())

        att_labels, att_outputs = filter_not_interacting_sample(att_labels, att_outputs)

        metrics["att"]["true"].extend(att_labels.tolist())
        metrics["att"]["pred"].extend(torch.argmax(torch.softmax(att_outputs, dim=1), dim=1).tolist())

        metrics["act"]["true"].extend(act_labels.tolist())
        metrics["act"]["pred"].extend(torch.argmax(torch.softmax(act_outputs, dim=1), dim=1).tolist())

    results = {}
    for key in metrics:
        y_true, y_pred = torch.tensor(metrics[key]["true"]), torch.tensor(metrics[key]["pred"])
        acc = (y_pred == y_true).sum().item() / len(y_true)
        f1 = f1_score(y_true, y_pred, average='weighted')
        results[key] = {"acc": acc * 100, "f1": f1 * 100}

    print(
        f"{task} Results -> epoch: {epoch}, int_acc: {results['int']['acc']:.2f}, int_f1: {results['int']['f1']:.2f}, "
        f"att_acc: {results['att']['acc']:.2f}, att_f1: {results['att']['f1']:.2f}, "
        f"act_acc: {results['act']['acc']:.2f}, act_f1: {results['act']['f1']:.2f}")

    return results


def main():
    args = parse_args()
    config = load_config(args.cfg)

    train_loader, val_loader, test_loader = get_dataloaders(config)
    socialegonet = create_model(config)
    if config['train']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(socialegonet.parameters(), lr=config['train']['learning_rate'])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(config['train']['num_epochs']):
        socialegonet.train()
        train_loader = tqdm(train_loader, dynamic_ncols=True)
        for inputs, (int_labels, att_labels, act_labels) in train_loader:
            int_labels, att_labels, act_labels = int_labels.to(device), att_labels.to(device), act_labels.to(device)
            int_outputs, att_outputs, act_outputs = socialegonet(inputs)
            loss_1 = functional.cross_entropy(int_outputs, int_labels)
            loss_2 = functional.cross_entropy(att_outputs, att_labels)
            loss_3 = functional.cross_entropy(act_outputs, act_labels)
            total_loss = loss_1 + loss_2 + loss_3
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        evaluate_model(epoch, socialegonet, val_loader, 'validation')

    print("Testing model on test set...")
    evaluate_model(socialegonet, test_loader, 'Test')

    torch.save(socialegonet.state_dict(), f"weights/socialegonet_jpl.pt")
    print(f"Model saved at weights/socialegonet_jpl.pt")


if __name__ == '__main__':
    main()
