import numpy as np
from torch.utils.data import Dataset
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """_summary_: Updates the average meter with the new value and the number of samples
        Args:
            val (_type_): value
            n (int, optional):  Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """_summary_

    Args:
        output (tensor): output of the model
        target (_type_): target
        topk (tuple, optional): topk. Defaults to (1,).

    Returns:
        float: accuracy
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class ExpertDatasetTensor(Dataset):
    """Generic dataset with expert predictions and labels and images"""

    def __init__(self, images, targets, exp_preds):
        self.images = images
        self.targets = np.array(targets)
        self.exp_preds = np.array(exp_preds)

    def __getitem__(self, index):
        """Take the index of item and returns the image, label, expert prediction and index in original dataset"""
        label = self.targets[index]
        image = self.images[index]
        expert_pred = self.exp_preds[index]
        return torch.FloatTensor(image), label, expert_pred

    def __len__(self):
        return len(self.targets)
