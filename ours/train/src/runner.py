import os
import json

import torch
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np

from irbench.irbench import IRBench
from irbench.evals.eval_helper import EvalHelper


class Trainer(object):
    def __init__(self,
                 args,
                 data_loader,
                 model,
                 summary_writer):
        self.args=args
        self.data_loader=data_loader
        self.model=model
        self.summary_writer=summary_writer

        self.processed_images = 0
        self.global_step = 0

    def __adjust_lr__(self, epoch, warmup=True):
        lr = self.args.lr * self.args.batch_size / 16.0
        if warmup:
            warmup_images = 10000
            lr = min(self.processed_images * lr / float(warmup_images), lr)
        for e in self.args.lr_decay_steps:
            if epoch >= e:
                lr *= self.args.lr_decay_factor
        self.model.adjust_lr(lr)
        self.cur_lr = lr

    def __logging__(self, log_data):
        msg = f'[Train][{self.args.expr_name}]'
        msg += f'[Epoch: {self.epoch}]'
        msg += f'[Lr:{self.cur_lr:.6f}]'
        log_data['lr'] = self.cur_lr
        for k, v in log_data.items():
            if not self.summary_writer is None:
                self.summary_writer.add_scalar(k, v, self.global_step)
            if isinstance(v, float):
                msg += f' {k}:{v:.6f}'
            else:
                msg += f' {k}:{v}'
        print(msg)

    def train(self, epoch):
        self.epoch = epoch
        self.model.train()

        for bidx, input in enumerate(tqdm(self.data_loader, desc='Train')):
            self.global_step += 1
            self.processed_images += input[0][0].size(0)
            self.__adjust_lr__(epoch, warmup=self.args.warmup)

            # data
            input[0][0] = Variable(input[0][0].cuda())      # input[0] = (x_c, c_c, data['c_id'])
            input[0][1] = Variable(input[0][1].cuda())
            input[1][0] = Variable(input[1][0].cuda())      # input[1] = (x_t, c_t, data['t_id'])
            input[1][1] = Variable(input[1][1].cuda())
            input[2][0] = Variable(input[2][0].cuda())      # input[2] = (we, w_key, text)
            input[3] = Variable(input[3].cuda())            # input[3] = (ie)

            # forward and update
            output = self.model(input)
            log_data = self.model.update(output, input)
            if (bidx % self.args.print_freq) == 0:
                self.__logging__(log_data)


class Evaluator(object):
    def __init__(self,
                 args,
                 data_loader,
                 model,
                 summary_writer,
                 eval_freq):

        self.args=args
        self.data_loader=data_loader
        self.model=model
        self.summary_writer=summary_writer
        self.eval_freq=eval_freq
        self.best_score = 0.
        self.repo_path = os.path.join('./repo', args.expr_name)
        os.makedirs(self.repo_path, exist_ok=True)
        self.targets = list(self.data_loader.keys())

    def test(self, epoch):
        # initialize irbench.
        ir_config = {}
        ir_config['srch_method'] = 'bf'
        ir_config['srch_libs'] = None
        ir_config['desc_type'] = 'global'
        irbench = IRBench(ir_config)

        # test.
        self.epoch = epoch
        model = self.model.eval()
        r10 = 0.
        r50 = 0.
        r10r50 = 0.
        for target, data_loader in self.data_loader.items():

            irbench.clean()
            eval_helper = EvalHelper()

            # add index features.
            data_loader.dataset.set_mode('index')
            for bidx, input in enumerate(tqdm(data_loader, desc='Index')):
                input[0] = Variable(input[0].cuda())      # input[0] = (x, image_id)
                data = input[0]
                image_id = input[1]

                with torch.no_grad():
                    output = model.get_original_image_feature(data)

                for i in range(output.size(0)):
                    _iid = image_id[i]
                    _feat = output[i].squeeze().cpu().numpy()
                    irbench.feed_index([_iid, _feat])

            # add query features && GT.
            data_loader.dataset.set_mode('query')
            for bidx, input in enumerate(tqdm(data_loader, desc='Query')):
                """
                input[0] = (x_c, c_c, data['c_id'])
                input[1] = (x_t, c_t, data['t_id'])
                input[2] = (we, w_key, text)
                """
                input[0][0] = Variable(input[0][0].cuda())
                input[2][0] = Variable(input[2][0].cuda())
                data = (input[0], input[2])

                with torch.no_grad():
                    output = model.get_manipulated_image_feature(data)

                for i in range(output.size(0)):
                    # query
                    _qid = input[2][1][i]
                    _feat = output[i].squeeze().cpu().numpy()
                    irbench.feed_query([_qid, _feat])

                    # GT
                    _w_key = input[2][1][i]
                    _tid = input[1][2][i]
                    eval_helper.feed_gt([_w_key, [_tid]])

            # get score.
            res = irbench.search_all(top_k=50)
            res = irbench.render_result(res)
            eval_helper.feed_rank_from_dict(res)
            score = eval_helper.evaluate(metric=['top_k_acc'], kappa=[10, 50])
            print(f'Target: {target}')
            print(score)
            _r10 = score[0][str(10)]['top_k_acc']
            _r50 = score[0][str(50)]['top_k_acc']
            _r10r50 = 0.5 * (score[0][str(10)]['top_k_acc'] + score[0][str(50)]['top_k_acc'])
            r10 += _r10
            r50 += _r50
            r10r50 += _r10r50
            if (bidx % self.args.print_freq) == 0 and self.summary_writer is not None:
                self.summary_writer.add_scalar(f'{target}/R10', _r10, epoch)
                self.summary_writer.add_scalar(f'{target}/R50', _r50, epoch)
                self.summary_writer.add_scalar(f'{target}/R10R50', _r10r50, epoch)

        # mean score.
        r10r50 /= len(self.data_loader)
        r10 /= len(self.data_loader)
        r50 /= len(self.data_loader)
        print(f'Overall>> R10:{r10:.4f}\tR50:{r50:.4f}\tR10R50:{r10r50:.4f}')

        # logging
        if (bidx % self.args.print_freq) == 0 and self.summary_writer is not None:
            self.summary_writer.add_scalar(f'overall/R10', r10, epoch)
            self.summary_writer.add_scalar(f'overall/R50', r50, epoch)
            self.summary_writer.add_scalar(f'overall/R10R50', r10r50, epoch)

        # save checkpoint
        cur_score = r10r50
        if (cur_score > self.best_score):
            self.best_score = cur_score
            with open(os.path.join(self.repo_path, 'args.json'), 'w', encoding='utf-8') as f:
                json.dump(vars(self.args), f, indent=4, ensure_ascii=False)
            state = { 'score': self.best_score }
            if len(self.data_loader) == 3:
                self.model.save(os.path.join(self.repo_path, 'best_model.pth'), state)
            elif len(self.data_loader) == 1:
                target = list(self.data_loader.keys())[0]
                self.model.save(os.path.join(self.repo_path, f'best_model_{target}.pth'), state)
            else:
                raise OSError('Something is wrong!!')
