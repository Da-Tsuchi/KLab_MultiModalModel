import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class LossCounter():
    def __init__(self):
        self.losses = {'train':[], 'val':[]}

    def add(self, phase, loss):
        self.losses[phase].append(loss)

    def plot_loss(self, result_dir):
        # Plot the loss values.
        fig = plt.figure()
        plt.plot(self.losses['train'], label='Train')
        plt.plot(self.losses['val'], label='Val')

        # Set the title and axis labels.
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Show the plot.
        plt.savefig(os.path.join(result_dir, "loss.png"))

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2,pad_token_id=0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.pad_token_id = pad_token_id

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='mean', ignore_index=self.pad_token_id)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss