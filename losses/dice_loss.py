"""Losses for binary pneumothorax segmentation."""
import torch
from torch import nn
import torch.nn.functional as F

class PositiveDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__(); self.smooth=smooth
    def forward(self, logits, targets):
        _validate_inputs(logits, targets)
        pos=targets.flatten(1).sum(1)>0
        if not pos.any(): return logits.sum()*0.0
        p=torch.sigmoid(logits[pos]).flatten(1); t=targets[pos].flatten(1)
        inter=(p*t).sum(1)
        dice=(2*inter+self.smooth)/(p.sum(1)+t.sum(1)+self.smooth)
        return 1-dice.mean()

class NegativeTopKLoss(nn.Module):
    def __init__(self, topk_fraction=0.02):
        super().__init__(); self.topk_fraction=topk_fraction
    def forward(self, logits, targets):
        _validate_inputs(logits, targets)
        neg=targets.flatten(1).sum(1)==0
        if not neg.any(): return logits.sum()*0.0
        x=logits[neg].flatten(1)
        k=max(1,int(x.shape[1]*self.topk_fraction))
        hard=torch.topk(x,k=k,dim=1).values
        return F.binary_cross_entropy_with_logits(hard,torch.zeros_like(hard))

class SegmentationLoss(nn.Module):
    def __init__(self,bce_weight=1.0,dice_weight=1.0,negative_weight=0.5,
                 smooth=1.0,pos_weight=1.0,topk_fraction=0.02):
        super().__init__()
        self.bce_weight=bce_weight; self.dice_weight=dice_weight; self.negative_weight=negative_weight
        self.register_buffer("pos_weight",torch.tensor([float(pos_weight)],dtype=torch.float32))
        self.dice=PositiveDiceLoss(smooth); self.negative=NegativeTopKLoss(topk_fraction)
    def forward(self,logits,targets):
        _validate_inputs(logits,targets)
        bce=F.binary_cross_entropy_with_logits(logits,targets,pos_weight=self.pos_weight)
        return self.bce_weight*bce+self.dice_weight*self.dice(logits,targets)+self.negative_weight*self.negative(logits,targets)

def _validate_inputs(logits,targets):
    if logits.shape!=targets.shape:
        raise ValueError(f"shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}")
    if not logits.is_floating_point() or not targets.is_floating_point():
        raise TypeError("logits and targets must be floating point")
