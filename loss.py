import torch
from torch.nn import functional as F

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, mask):
        pred = pred.flatten()
        mask = mask.flatten().float()
        
        intersect = (mask * pred).sum()
        dice_score = (2*intersect + self.smooth) / (pred.sum() + mask.sum() + self.smooth)
        dice_loss = 1 - dice_score
        return dice_loss
    
class DiceLossWithLogits(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, mask):
        prob = torch.sigmoid(pred)
        mask = mask.float()

        prob = prob.flatten()
        mask = mask.flatten()

        intersection = (prob * mask).sum()
        dice_score = (2 * intersection + self.smooth) / (prob.sum() + mask.sum() + self.smooth)
        dice_loss = 1 - dice_score
        return dice_loss