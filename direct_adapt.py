import numpy as np
import json
from network import ANN
import torch
from dataset import CalHouseDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def cal_var(lst):
    """
    Args:
        lst: a list of prediction
    Returns:
        variance of lst
    """
    lst = np.array(lst)
    return np.var(lst)


def adapt():
    """
    Collect source_y, target_y, and target_label
    """
    net = ANN(input_size=8, output_size=1, hidden_sizes=[128, 128, 64], dropout=0.2)
    #
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.load_state_dict(torch.load('./model/pretrained_model.pt', map_location=torch.device(device)))
    net.to(device)
    #
    target_dataset = CalHouseDataset(domain_index='poor')
    target_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=False)
    target_x = []
    target_y = []
    target_label = []
    for j, data in enumerate(target_dataloader):
        x, label, sample_id = data
        x = x.to(device)
        # Calculate Variance
        net.train()
        y_list = []
        #pred_list = []
        #pred_list.append(pred.iemt())
        pred = net(x)
        target_x.append(sample_id.item())
        target_y.append(label.item())
        target_label.append(pred.item())
        
    plt.plot(target_x, target_y, label='labels')
    plt.plot(target_x, target_label, label='preds')
    plt.legend()
    plt.show()
        # Calculate prediction for std (|pred-label|)
        #net.eval()
    
    
    
    #with open('./data/source_y.json', 'w') as fp:
    #    json.dump(source_y, fp)
    #with open('./data/target_y.json', 'w') as fp:
    #    json.dump(target_y, fp)
    #with open('./data/target_label.json', 'w') as fp:
    #    json.dump(target_label, fp)


if __name__ == '__main__':
    adapt()
