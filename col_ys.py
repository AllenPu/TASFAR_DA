import numpy as np
import json
from network import ANN
import torch
from dataset import CalHouseDataset
from torch.utils.data import DataLoader


def cal_var(lst):
    """
    Args:
        lst: a list of prediction
    Returns:
        variance of lst
    """
    lst = np.array(lst)
    return np.var(lst)


def col_ys():
    """
    Collect source_y, target_y, and target_label
    """
    net = ANN(input_size=8, output_size=1, hidden_sizes=[128, 128, 64], dropout=0.2)
    net.load_state_dict(torch.load('./model/pretrained_model.pt'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    source_dataset = CalHouseDataset(domain_index='rich')
    source_dataloader = DataLoader(source_dataset, batch_size=1, shuffle=False)
    source_y = []
    for j, data in enumerate(source_dataloader):
        x, label, sample_id = data
        x = x.to(device)
        # Calculate Variance
        net.train()
        pred_list = []
        for i in range(20):
            pred = net(x)
            pred_list.append(pred.item())
        var = cal_var(pred_list)
        # Calculate prediction for std (|pred-label|)
        net.eval()
        prediction = net(x).item()
        source_y.append([var, prediction, label.item()])

    target_dataset = CalHouseDataset(domain_index='poor')
    target_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=False)
    target_y = {}
    target_label = {}
    for j, data in enumerate(target_dataloader):
        x, label, sample_id = data
        x = x.to(device)
        # Calculate Variance
        net.train()
        pred_list = []
        for i in range(20):
            pred = net(x)
            pred_list.append(pred.item())
        var = cal_var(pred_list)
        # Calculate prediction for std (|pred-label|)
        net.eval()
        prediction = net(x).item()
        target_y[sample_id.item()] = (var, prediction)
        target_label[sample_id.item()] = label.item()

    with open('./data/source_y.json', 'w') as fp:
        json.dump(source_y, fp)
    with open('./data/target_y.json', 'w') as fp:
        json.dump(target_y, fp)
    with open('./data/target_label.json', 'w') as fp:
        json.dump(target_label, fp)


if __name__ == '__main__':
    col_ys()
