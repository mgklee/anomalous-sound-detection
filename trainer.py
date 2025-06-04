import torch
from torch.amp import autocast
from tqdm import tqdm

from losses import ASDLoss
from model.net import MSMTgramMFN
from utils import get_accuracy, mixup_data, noisy_arcmix_criterion


class Trainer:
    def __init__(self, device, alpha, beta, m, s, epochs=300, class_num=41, lr=1e-4):
        self.device = device
        self.epochs = epochs
        self.alpha = alpha
        self.beta = beta
        self.net = MSMTgramMFN(num_classes=class_num, use_arcface=True, m=m, s=s).to(self.device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=0.1*float(lr))
        self.criterion = ASDLoss().to(self.device)
        self.test_criterion = ASDLoss(reduction='none').to(self.device)

    def train(self, train_loader, valid_loader, save_path):
        num_steps = len(train_loader)
        min_val_loss = 1e10

        for epoch in tqdm(range(self.epochs), total=self.epochs):
            sum_loss = 0.
            sum_accuracy = 0.

            for x_wavs, x_mels, id_labels, type_labels in tqdm(train_loader, total=num_steps):
                self.net.train()

                x_wavs = x_wavs.to(self.device)
                x_mels = x_mels.to(self.device)
                id_labels = id_labels.to(self.device)
                type_labels = type_labels.to(self.device)

                with autocast('cuda'):
                    mixed_x_wavs, mixed_x_mels, y_id, y_type, lam = mixup_data(x_wavs, x_mels, id_labels, type_labels,
                                                                               self.device, alpha=self.alpha)
                    id_logits, type_logits, _ = self.net(mixed_x_wavs, mixed_x_mels, id_labels, type_labels)
                    id_loss = noisy_arcmix_criterion(self.criterion, id_logits, y_id[0], y_id[1], lam)
                    type_loss = noisy_arcmix_criterion(self.criterion, type_logits, y_type[0], y_type[1], lam)
                    loss = self.beta * type_loss + (1 - self.beta) * id_loss

                sum_accuracy += get_accuracy(id_logits, id_labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                sum_loss += loss.item()
            self.scheduler.step()

            avg_loss = sum_loss / num_steps
            avg_accuracy = sum_accuracy / num_steps

            valid_loss, valid_accuracy = self.valid(valid_loader)

            if min_val_loss > valid_loss:
                min_val_loss = valid_loss
                lr = self.scheduler.get_last_lr()[0]
                print("model has been saved!")
                print(f'lr: {lr:.7f} | EPOCH: {epoch} | Train_loss: {avg_loss:.5f} | Train_accuracy: {avg_accuracy:.5f} | Valid_loss: {valid_loss:.5f} | Valid_accuracy: {valid_accuracy:.5f}')
                torch.save(self.net.state_dict(), save_path)

    def valid(self, valid_loader):
        self.net.eval()

        num_steps = len(valid_loader)
        sum_loss = 0.
        sum_accuracy = 0.

        for (x_wavs, x_mels, id_labels, type_labels) in valid_loader:
            x_wavs = x_wavs.to(self.device)
            x_mels = x_mels.to(self.device)
            id_labels = id_labels.to(self.device)
            type_labels = type_labels.to(self.device)

            id_logits, type_logits, _ = self.net(x_wavs, x_mels, id_labels, type_labels)
            id_loss = self.criterion(id_logits, id_labels)
            type_loss = self.criterion(type_logits, type_labels)
            loss = self.beta * type_loss + (1 - self.beta) * id_loss
            sum_loss += loss.item()
            sum_accuracy += get_accuracy(id_logits, id_labels)

        avg_loss = sum_loss / num_steps
        avg_accuracy = sum_accuracy / num_steps
        return avg_loss, avg_accuracy