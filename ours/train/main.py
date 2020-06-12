import os
import sys
import time
import random
import argparse

from pprint import pprint
import torch

from irbench.irbench import IRBench
from irbench.evals.eval_helper import EvalHelper


# init
def init_env():
    # load argsuration.
    state = {k: v for k, v in args._get_kwargs()}
    pprint(state)
    args.lr_decay_steps = [int(x) for x in args.lr_decay_steps.strip().split(',')]

    # if use cuda.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # Random seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manualSeed)
        torch.backends.cudnn.benchmark = True  # speed up training.


def main():
    # init
    init_env()

    # train/val data loader
    from src.dataset import FashionIQTrainValDataset, FashionIQTestDataset
    train_dataset = FashionIQTrainValDataset(
        data_root=args.data_root,
        image_size=args.image_size,
        split='train',
        target=args.target,
    )
    train_loader = train_dataset.get_loader(batch_size=args.batch_size)

    if (args.target == 'all') or (args.target is None):
        targets = ['dress', 'toptee', 'shirt']
    else:
        targets = [args.target]

    test_loader = dict()
    for target in targets:
        test_dataset = FashionIQTestDataset(
            data_root=args.data_root,
            image_size=args.image_size,
            split='val',
            target=target,
        )
        test_loader[target] = test_dataset.get_loader(batch_size=args.batch_size)

    # model
    if args.method == 'text-only':
        from src.model.text_only import TextOnlyModel
        model = TextOnlyModel(args=args,
                              backbone=args.backbone,
                              texts=train_dataset.get_all_texts(),
                              text_method=args.text_method,
                              fdims=args.fdims,
                              fc_arch='A',
                              init_with_glove=False)
    elif args.method == 'tirg':
        from src.model.tirg import TIRG
        model = TIRG(args=args,
                     backbone=args.backbone,
                     texts=train_dataset.get_all_texts(),
                     text_method=args.text_method,
                     fdims=args.fdims,
                     fc_arch='B',
                     init_with_glove=True)
    elif args.method == 'match-text-only':
        from src.model.match import MatchTextOnly
        args.text_method = 'lstm'
        model = MatchTextOnly(args=args,
                              backbone=args.backbone,
                              texts=train_dataset.get_all_texts(),
                              text_method=args.text_method,
                              fdims=args.fdims,
                              fc_arch='A',
                              init_with_glove=False)
    elif args.method == 'match-tirg':
        from src.model.match import MatchTIRG
        model = MatchTIRG(args=args,
                          backbone=args.backbone,
                          texts=train_dataset.get_all_texts(),
                          text_method=args.text_method,
                          fdims=args.fdims,
                          fc_arch='B',
                          init_with_glove=True)
    else:
        raise NotImplementedError()

    # cuda
    model = model.cuda()

    # summary writer
    from tensorboardX import SummaryWriter
    log_path = os.path.join('logs', args.expr_name)
    os.makedirs(log_path, exist_ok=True)
    summary_writer = SummaryWriter(log_path)

    # trainer
    from src.runner import Trainer, Evaluator
    trainer = Trainer(args=args,
                      data_loader=train_loader,
                      model=model,
                      summary_writer=summary_writer)

    evaluator = Evaluator(args=args,
                          data_loader=test_loader,
                          model=model,
                          summary_writer=summary_writer,
                          eval_freq=1)

    # train/test for N-epochs.
    for epoch in range(args.epochs):
        epoch = epoch + 1
        trainer.train(epoch)
        evaluator.test(epoch)

    print('Congrats! You just finished training.')


if __name__ == '__main__':
    # args for region
    parser = argparse.ArgumentParser('Train')

    # Common options.
    parser.add_argument('--gpu_id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--manualSeed', type=int, default=int(time.time()), help='manual seed')
    parser.add_argument('--warmup', action="store_true", help="warmup?")
    parser.add_argument('--expr_name', default='devel', type=str, help='experiment name')
    parser.add_argument('--data_root', required=True, type=str)
    parser.add_argument('--text_method',
                        default='lstm',
                        choices=['lstm','swem','lstm-gru'],
                        type=str)
    parser.add_argument('--fdims', default=2048, type=int, help='output feature dimensions')

    # common training parameters
    parser.add_argument('--method', default='tirg', type=str, help='method')
    parser.add_argument('--target', default='all', type=str, help='target (dress | shirt | toptee)')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--print_freq', default=1, type=int)
    parser.add_argument('--batch_size', default=32, type=int, help='train batchsize')
    parser.add_argument('--image_size', default=224, type=int, help='image size (default: 16)')
    parser.add_argument('--backbone', default='resnet152', type=str)
    parser.add_argument('--normalize_scale', default=5.0, type=float)
    parser.add_argument('--lr', default=0.00011148, type=float, help='initial learning rate')
    parser.add_argument('--lrp', default=0.48, type=float, help='lrp')
    parser.add_argument('--lr_decay_factor', default=0.4747, type=float)
    parser.add_argument('--lr_decay_steps', default="10,20,30,40,50,60,70", type=str)

    ## parse and save args.
    args, _ = parser.parse_known_args()

    ## train
    main()
