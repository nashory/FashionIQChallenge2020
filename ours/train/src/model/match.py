import math
import string
import random
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M
from torch.autograd import Variable

from src.model.text_only import TextOnlyModel
from src.model.tirg import TIRG


class MatchBase(object):
    def update(self, output, input):
        '''
        input = (
            (x_c, c_c, data['c_id']),
            (x_t, c_t, data['t_id']),
            (we, we_key, text),
            (ie)
        )
        '''
        # assign input
        x1 = self.model['norm'](output[0])  # manipulated
        x2 = self.model['norm'](output[1])  # target
        ie = self.model['norm'](input[3])  # target

        # loss
        loss = 0.
        loss += 1.0 - F.cosine_similarity(x1, ie)
        loss += 1.0 - F.cosine_similarity(x2, ie)
        loss /= 2.
        loss = loss.mean()

        # backward
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # return log
        log_data = dict()
        log_data['loss'] = float(loss.data)
        return log_data

class MatchTIRG(MatchBase, TIRG):
    def __init__(self, **kwargs):
        super(MatchTIRG, self).__init__(**kwargs)

class MatchTextOnly(MatchBase, TextOnlyModel):
    def __init__(self, **kwargs):
        super(MatchTextOnly, self).__init__(**kwargs)
