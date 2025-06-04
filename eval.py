from sklearn import metrics
import torch
from torch.utils.data import DataLoader
import yaml

from dataloader import test_dataset
from model.net import MSMTgramMFN
from losses import ASDLoss


def evaluator(net, test_loader, criterion, beta, device):
    net.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x_wavs, x_mels, id_labels, type_labels, AN_N_labels in test_loader:
            x_wavs = x_wavs.to(device)
            x_mels = x_mels.to(device)
            id_labels = id_labels.to(device)
            type_labels = type_labels.to(device)
            AN_N_labels = AN_N_labels.to(device)

            id_logits, type_logits, _ = net(x_wavs, x_mels, id_labels, type_labels)
            id_loss = criterion(id_logits, id_labels)
            type_loss = criterion(type_logits, type_labels)
            score = beta * type_loss + (1 - beta) * id_loss

            y_pred.extend(score.tolist())
            y_true.extend(AN_N_labels.tolist())

    auc = metrics.roc_auc_score(y_true, y_pred)
    pauc = metrics.roc_auc_score(y_true, y_pred, max_fpr=0.1)
    return auc, pauc


def main():
    with open('config.yaml', encoding='UTF-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    print(cfg)

    beta = cfg['beta']
    device_num = cfg['gpu_num']
    device = torch.device(f'cuda:{device_num}')

    net = MSMTgramMFN(num_classes=cfg['num_classes'], m=cfg['m'], s=cfg['s']).to(device)
    net.load_state_dict(torch.load(cfg['save_path']))
    net.eval()

    criterion = ASDLoss(reduction='none').to(device)

    name_list = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
    root_path = '/home/Dataset/DCASE2020_Task2_dataset/dev_data'

    avg_AUC = 0.
    avg_pAUC = 0.

    for i in range(len(name_list)):
        test_ds = test_dataset(root_path, name_list[i], name_list)
        test_dataloader = DataLoader(test_ds, batch_size=1)

        AUC, PAUC = evaluator(net, test_dataloader, criterion, beta, device)
        avg_AUC += AUC
        avg_pAUC += PAUC
        print(f"{name_list[i]} - AUC: {AUC:.5f}, pAUC: {PAUC:.5f}")

    avg_AUC = avg_AUC / len(name_list)
    avg_pAUC = avg_pAUC / len(name_list)

    print(f"Average AUC: {avg_AUC:.5f},  Average pAUC: {avg_pAUC:.5f}")


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()