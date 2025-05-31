import torch
from dataloader import test_dataset
from torch.utils.data import DataLoader
from sklearn import metrics
from model.net import TASTgramMFN
from losses import ASDLoss
import yaml


def evaluator(net, test_loader, criterion, device):
    net.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x_wavs, x_mels, labels, AN_N_labels in test_loader:
            x_wavs, x_mels, labels, AN_N_labels = x_wavs.to(device), x_mels.to(device), labels.to(device), AN_N_labels.to(device)

            logits, _ = net(x_wavs, x_mels, labels, train=False)
            score = criterion(logits, labels)

            y_pred.extend(score.tolist())
            y_true.extend(AN_N_labels.tolist())

    auc = metrics.roc_auc_score(y_true, y_pred)
    pauc = metrics.roc_auc_score(y_true, y_pred, max_fpr=0.1)
    return auc, pauc


def main():
    with open('config.yaml', encoding='UTF-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    print(cfg)

    device_num = cfg['gpu_num']
    device = torch.device(f'cuda:{device_num}')

    net = TASTgramMFN(num_classes=cfg['num_classes'], m=cfg['m'], mode=cfg['mode']).to(device)
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

        AUC, PAUC = evaluator(net, test_dataloader, criterion, device)
        avg_AUC += AUC
        avg_pAUC += PAUC
        print(f"{name_list[i]} - AUC: {AUC:.5f}, pAUC: {PAUC:.5f}")

    avg_AUC = avg_AUC / len(name_list)
    avg_pAUC = avg_pAUC / len(name_list)

    print(f"Average AUC: {avg_AUC:.5f},  Average pAUC: {avg_pAUC:.5f}")


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()