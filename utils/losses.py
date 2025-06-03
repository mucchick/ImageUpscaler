import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):
    """Loss Function"""
    def __init__(self, epsilon=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.epsilon)
        return loss.mean()