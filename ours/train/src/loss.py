#!/usr/bin/env python
# encoding: utf-8

import math
import time
import numpy as np
import torch
import torchvision
import torch.nn.functional as F

from torch import nn
from torch.nn import Parameter


def _pairwise_distances(embeddings, squared=False, p=2):
    assert p == 1 or p == 2

    if p == 2:
        dot_product = torch.mm(embeddings, embeddings.transpose(0, 1))
        square_norm = torch.diag(dot_product)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = torch.clamp(distances, min=0.0)

        if not squared:
            mask = torch.eq(distances, 0.0).float()
            distances = distances + mask * 1e-6
            distances = torch.sqrt(distances)
            distances = distances * (1.0 - mask)

    elif p == 1:
        abs = torch.abs(embeddings.unsqueeze(0) - embeddings.unsqueeze(1))
        distances = torch.sum(abs, dim=2)

    return distances


class NormalizationLayer(torch.nn.Module):
    def __init__(self, normalize_scale=1.0, learn_scale=True):
        super(NormalizationLayer, self).__init__()
        self.norm_s = float(normalize_scale)
        if learn_scale:
            self.norm_s = torch.nn.Parameter(torch.FloatTensor((self.norm_s,)))

    def forward(self, x):
        features = self.norm_s * x / torch.norm(x, dim=1, keepdim=True).expand_as(x)
        return features


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=1, p=2, squared=True, soft=False):
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.squared = squared
        self.soft = soft

    def forward(self, x1, x2):
        pairwise_dist = _pairwise_distances(torch.cat([x1, x2]), p=self.p, squared=self.squared)

        labels = torch.Tensor(list(range(x1.shape[0])) + list(range(x2.shape[0])))
        mask_anchor_positive = self.__get_anchor_positive_triplet_mask(labels).float()
        anchor_positive_dist = torch.mul(mask_anchor_positive, pairwise_dist)
        hardest_positive_dist = torch.max(anchor_positive_dist, dim=1, keepdim=True)[0]

        mask_anchor_negative = self.__get_anchor_negative_triplet_mask(labels).float()
        max_anchor_negative_dist = torch.max(pairwise_dist, dim=1, keepdim=True)[0]
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
        hardest_negative_dist = torch.min(anchor_negative_dist, dim=1, keepdim=True)[0]

        if self.soft:
            triplet_loss = torch.log(hardest_positive_dist - hardest_negative_dist + self.margin)
        else:
            triplet_loss = hardest_positive_dist - hardest_negative_dist + self.margin
        triplet_loss = torch.max(triplet_loss, torch.zeros_like(triplet_loss))
        triplet_loss = torch.mean(triplet_loss)

        return triplet_loss

    def __get_anchor_positive_triplet_mask(self, labels):
        indices_equal = torch.eye(labels.shape[0]).bool()
        indices_not_equal = ~indices_equal

        labels_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        mask = indices_not_equal.__and__(labels_equal).cuda()

        return mask

    def __get_anchor_negative_triplet_mask(self, labels):
        labels_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).bool()
        mask = ~labels_equal.cuda()

        return mask
