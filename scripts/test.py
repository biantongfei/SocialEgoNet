from Models import SocialEgoNet
from DataLoader import JPL_Social_DataLoader
from constants import device
from data import JPL_Social_Dataset, filter_not_interacting_sample

import torch

from sklearn.metrics import f1_score
from tqdm import tqdm
import argparse
import yaml


def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def evaluate(model, dataloader):
    """ 评估模型，计算准确率和F1分数 """
    model.eval()
    metrics = {"int": ([], []), "att": ([], []), "act": ([], [])}

    with torch.no_grad():
        for inputs, (int_labels, att_labels, act_labels) in tqdm(dataloader, dynamic_ncols=True, desc="Testing"):
            int_labels, att_labels, act_labels = (int_labels.to(device),
                                                  att_labels.to(device),
                                                  act_labels.to(device))

            int_outputs, att_outputs, act_outputs = model(inputs)

            for key, outputs, labels in zip(["int", "att", "act"],
                                            [int_outputs, att_outputs, act_outputs],
                                            [int_labels, att_labels, act_labels]):
                outputs = torch.softmax(outputs, dim=1)
                if key == "att":
                    labels, outputs = filter_not_interacting_sample(labels, outputs)
                _, preds = torch.max(outputs, dim=1)

                metrics[key][0].extend(labels.tolist())
                metrics[key][1].extend(preds.tolist())

    return {key: (f1_score(torch.tensor(y_true), torch.tensor(y_pred), average="weighted"),
                  (torch.tensor(y_pred) == torch.tensor(y_true)).float().mean().item() * 100)
            for key, (y_true, y_pred) in metrics.items()}


def main():
    parser = argparse.ArgumentParser(description="Test SocialEgoNet on JPL-P4S")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
    parser.add_argument("--check_point", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    config = load_config(args.cfg)

    # 加载数据集
    testset = JPL_Social_Dataset(config["data"]["path"] + "test/", config["data"]["sequence_length"])
    test_loader = JPL_Social_DataLoader(dataset=testset, sequence_length=config["data"]["sequence_length"],
                                        batch_size=config["train"]["batch_size"])

    # 加载模型
    model = SocialEgoNet(sequence_length=config["data"]["sequence_length"], **config["model"])
    model.load_checkpoint(args.check_point)
    model.to(device)

    print("Testing...")
    results = evaluate(model, test_loader)

    # 打印评估结果
    for key in ["int", "att", "act"]:
        print(f"{key}_acc: {results[key][1]:.2f}, {key}_f1: {results[key][0]:.2f}")


if __name__ == '__main__':
    main()
