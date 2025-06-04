import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from dataloader import train_dataset
from trainer import Trainer
from utils import dataset_split


def main():
    with open('config.yaml', encoding='UTF-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    print('Configuration...')
    print(cfg)

    name_list = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
    root_path = '/home/Dataset/DCASE2020_Task2_dataset/dev_data'

    device_num = cfg['gpu_num']
    device = torch.device(f'cuda:{device_num}')

    print('training dataset loading...')
    dataset = train_dataset(root_path, name_list)

    train_ds, valid_ds = dataset_split(dataset, split_ratio=cfg['split_ratio'])

    train_dataloader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    valid_dataloader = DataLoader(valid_ds, batch_size=cfg['batch_size'])

    trainer = Trainer(device=device, alpha=cfg['alpha'], beta=cfg['beta'],
                      m=cfg['m'], s=cfg['s'], epochs=cfg['epoch'],
                      class_num=cfg['num_classes'], lr=cfg['lr'])

    trainer.train(train_dataloader, valid_dataloader, cfg['save_path'])


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    random_seed(2025)
    torch.set_num_threads(4)
    main()