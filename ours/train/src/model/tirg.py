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

import src.model.resnet as resnet
from src.model.fusion import ConCatModule
from src.model.base import ImageEncoderTextEncoderBase
from src.loss import (NormalizationLayer,
                      BatchHardTripletLoss)
 

class TIRG(ImageEncoderTextEncoderBase):
    """The TIRG model.
    The method is described in
    Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays.
    "Composing Text and Image for Image Retrieval - An Empirical Odyssey"
    CVPR 2019. arXiv:1812.07119
    """

    def __init__(self, args, **kwargs):
        super(TIRG, self).__init__(**kwargs)

        self.args = args
        self.texts = kwargs.get('texts')
        self.text_method = kwargs.get('text_method')
        normalize_scale = args.normalize_scale

        self.model['criterion'] = BatchHardTripletLoss()

        self.w = nn.Parameter(torch.FloatTensor([1.0, 10.0, 1.0, 1.0]))
        self.model['norm'] = NormalizationLayer(learn_scale=True,
                                                normalize_scale=normalize_scale)
                                                
        self.model['gated_feature_composer'] = torch.nn.Sequential(
            ConCatModule(),
            nn.BatchNorm1d(self.out_feature_image + self.out_feature_text),
            nn.ReLU(),
            nn.Linear(self.out_feature_image + self.out_feature_text, self.out_feature_image + self.out_feature_text),
            nn.BatchNorm1d(self.out_feature_image + self.out_feature_text),
            nn.ReLU(),
            nn.Linear(self.out_feature_image + self.out_feature_text, self.out_feature_image),
        )

        self.model['res_info_composer'] = torch.nn.Sequential(
            ConCatModule(),
            nn.BatchNorm1d(self.out_feature_image + self.out_feature_text),
            nn.ReLU(),
            nn.Linear(self.out_feature_image + self.out_feature_text, self.out_feature_image + self.out_feature_text),
            nn.ReLU(),
            nn.Linear(self.out_feature_image + self.out_feature_text, self.out_feature_image),
        )
        self.model = nn.ModuleDict(self.model)

        # optimizer
        self.opt = torch.optim.AdamW(
            self.get_config_optim(args.lr),
            lr=args.lr,
            betas=(0.55, 0.999)
        )

    def compose_img_text(self, imgs, texts):
        image_features = self.extract_image_feature(imgs)
        text_features = self.extract_text_feature(texts)
        return self.compose_image_text_features(image_features, text_features)

    def compose_image_text_features(self, image_features, text_features):
        f1 = self.model['gated_feature_composer']((image_features, text_features))
        f2 = self.model['res_info_composer']((image_features, text_features))
        f = torch.sigmoid(f1) * image_features * self.w[0] + f2 * self.w[1]
        return f

    def get_config_optim(self, lr):
        params = []
        for k, v in self.model.items():
            if k == 'backbone':
                params.append({ 'params': v.parameters(), 'lr': lr, 'lrp': float(self.args.lrp)})
            else:
                params.append({ 'params': v.parameters(), 'lr': lr, 'lrp': 1.0 })
        return params

    def adjust_lr(self, lr):
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr * param_group['lrp']

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
            x = self.compose_img_text(x[0][0], x[1][0])
        else:
            x = self.compose_img_text(x[0][0], x[1][2])
        return self.model['norm'](x)

    def update(self, output, input):
        '''
        input = (
            (x_c, c_c, data['c_id']),
            (x_t, c_t, data['t_id']),
            (we, we_key, text)
        )
        '''

        # assign input
        x1 = self.model['norm'](output[0])  # manipulated
        x2 = self.model['norm'](output[1])  # target

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
            x_f = self.compose_img_text(x[0][0], x[2][0])
        else:
            x_f = self.compose_img_text(x[0][0], x[2][2])
        x_c = self.extract_image_feature(x[0][0])
        x_t = self.extract_image_feature(x[1][0])
        return (x_f, x_t, x_c)
