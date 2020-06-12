import math

import torch
import torch.nn as nn
import torchvision.models as M
import torch.nn.functional as F

import src.model.resnet as resnet
from src.model.base import ImageEncoderTextEncoderBase
from src.loss import (NormalizationLayer,
                      BatchHardTripletLoss)


class TextOnlyModel(ImageEncoderTextEncoderBase):
    def __init__(self, args, **kwargs):
        super(TextOnlyModel, self).__init__(**kwargs)

        self.args = args
        self.texts = kwargs.get('texts')
        self.text_method = kwargs.get('text_method')
        normalize_scale = args.normalize_scale

        self.model['criterion'] = BatchHardTripletLoss()
        self.model['norm'] = NormalizationLayer(learn_scale=True,
                                                normalize_scale=normalize_scale)
        self.model = nn.ModuleDict(self.model)
        
        # optimizer
        self.opt = torch.optim.AdamW(
            self.get_config_optim(args.lr),
            lr=args.lr,
            betas=(0.55, 0.999)
        )

    def get_config_optim(self, lr):
        params = []
        for k, v in self.model.items():
            if k == 'backbone':
                params.append({'params': v.parameters(), 'lr': lr * 0.1})
            else:
                params.append({'params': v.parameters(), 'lr': lr})
        return params

    def adjust_lr(self, lr):
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr

    def save(self, path, state={}):
        state['state_dict'] = dict()
        for k, v in self.model.items():
            state['state_dict'][k] = v.state_dict()
        state['texts'] = self.texts
        torch.save(state, path)

    def load(self, path):
        state_dict = torch.load(path)['state_dict']
        for k, v in state_dict.items():
            self.model[k].load_state_dict(v)

    def get_original_image_feature(self, x):
        '''
        x = image
        '''
        x = self.extract_image_feature(x)
        return self.model['norm'](x)

    def get_manipulated_image_feature(self, x):
        '''
        x[0] = (x_c, c_c, data['c_id'])
        x[1] = (we, w_key, text)
        '''
        if self.text_method == 'swem':
            x = self.extract_text_feature(x[1][0])
        else:
            x = self.extract_text_feature(x[1][2])
        return self.model['norm'](x)

    def update(self, output, input):
        # assign input
        x1 = self.model['norm'](output[0])
        x2 = self.model['norm'](output[1])

        # loss
        loss = self.model['criterion'](x1, x2)

        # backward
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # return log
        log_data = dict()
        log_data['loss'] = float(loss.data)
        return log_data

    def forward(self, x):
        '''
        data = (
            (x_c, c_c, data['c_id']),
            (x_t, c_t, data['t_id']),
            (we, we_key, text) 
        )
        '''
        if self.text_method == 'swem':
            x_f = self.extract_text_feature(x[2][0])
        else:
            x_f = self.extract_text_feature(x[2][2])
        x_c = self.extract_image_feature(x[0][0])
        x_t = self.extract_image_feature(x[1][0])
        return (x_f, x_t, x_c)
