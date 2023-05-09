import torch
from torch import nn

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
        self.eps = 1e-7

    def forward(self, logits, y_true):
        logits = torch.sigmoid(logits)

        # Flatten the input tensors
        logits = logits.view(-1)        
        y_true = y_true.view(-1)
        # Calculate the confusion matrix
        intersection = (logits * y_true).sum()
        union = logits.sum() + y_true.sum() - intersection

        # Calculate the IoU and return the complement as the loss
        iou = intersection / (union + self.eps)
        return 1 - iou

class BCEDiceLoss(nn.Module):

    """
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: loss.
    """

    def __init__(self, bce_weight=0.5, weight=None, smooth=.0, class_weights=[1]):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.eps = 1e-7
        self.smooth = smooth
        self.class_weights = class_weights
        self.nll = torch.nn.BCEWithLogitsLoss(weight=weight)

    def forward(self, logits, y_true):
        loss = self.bce_weight * self.nll(logits, y_true)
        if self.bce_weight < 1.:
            dice_loss = 0.
            batch_size, num_classes = logits.shape[:2]
            logits = torch.sigmoid(logits)
            for c in range(num_classes):
                iflat = logits[:, c,...].view(batch_size, -1)
                tflat = y_true[:, c,...].view(batch_size, -1)
                intersection = (iflat * tflat).sum()
                
                w = self.class_weights[c]
                dice_loss += w * ((2. * intersection + self.smooth) /
                                 (iflat.sum() + tflat.sum() + self.smooth + self.eps))
            loss -= (1 - self.bce_weight) * torch.log(dice_loss)

        return loss
