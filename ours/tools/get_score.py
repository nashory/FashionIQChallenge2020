'''get_score.py
'''
import os
import sys
sys.path.append('../train')
import pickle
import time
import random
import json
import argparse
import easydict

from pprint import pprint
from tqdm import tqdm
import numpy as np

import torch
from torch.autograd import Variable

from datetime import datetime


TOP_K = 50

# init
def init_env():
    # load argsuration.
    state = {k: v for k, v in args._get_kwargs()}
    pprint(state)

    # if use cuda.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    # Random seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manualSeed)
        torch.backends.cudnn.benchmark = True           # speed up training.

        
score = dict()
hyperopt = dict()

def main(args):
    # init
    init_env()
        
    date_key = str(datetime.now().strftime("%Y%m%d%H%M%S"))[2:]

    # load model
    print(f'Load model: {args.expr_name}')
    root_path = f'../train/repo/{args.expr_name}'
    with open(os.path.join(root_path, 'args.json'), 'r') as f:
        largs = json.load(f)
        largs = easydict.EasyDict(largs)
        pprint(largs)
        image_size = largs.image_size
        texts = torch.load(os.path.join(root_path, 'best_model.pth'))['texts']
    if largs.method == 'text-only':
        from src.model.text_only import TextOnlyModel
        model = TextOnlyModel(args=largs,
                              backbone=largs.backbone,
                              texts=texts,
                              text_method=largs.text_method,
                              fdims=largs.fdims,
                              fc_arch='A',
                              init_with_glove=False,
                              loss_type=largs.loss_type)
    elif largs.method == 'tirg':
        from src.model.tirg import TIRG
        model = TIRG(
            args=largs,
            backbone=largs.backbone,
            texts=texts,
            text_method=largs.text_method,
            fdims=largs.fdims,
            fc_arch='B',
            init_with_glove=True,
            loss_type=largs.loss_type,
        )
    elif largs.method == 'match-tirg':
        from src.model.match import MatchTIRG
        model = MatchTIRG(
            args=largs,
            backbone=largs.backbone,
            texts=texts,
            text_method=largs.text_method,
            fdims=largs.fdims,
            fc_arch='B',
            init_with_glove=True,
            loss_type=largs.loss_type,
        )
    elif largs.method == 'match-text-only':
        from src.model.match import MatchTextOnly
        model = TextOnlyModel(backbone=largs.backbone,
                              texts=texts,
                              text_method=largs.text_method,
                              fdims=largs.fdims,
                              fc_arch='A',
                              init_with_glove=False,
                              loss_type=largs.loss_type)
            
    model.load(os.path.join(root_path, 'best_model.pth'))
    model = model.cuda()
    model.eval()
    print(model)

    SPLITS = ['val','test']
    targets = ['dress', 'toptee', 'shirt']
    for SPLIT in SPLITS:
        for target in targets:
            print(f">> SPLIT: {SPLIT} / TARGET: {target}")
            score[target] = dict()
            # train/val data loader
            from src.dataset import FashionIQTestDataset
            test_dataset = FashionIQTestDataset(
                data_root=args.data_root,
                image_size=image_size,
                split=SPLIT,
                target=target
            )
            test_loader = test_dataset.get_loader(batch_size=16)
           
            # get original image features of index.
            index_ids = []
            index_feats = []
            print('Extract Index Features...')
            test_loader.dataset.set_mode('index')
            for bidx, input in enumerate(tqdm(test_loader, desc='Index')):
                input[0] = Variable(input[0].cuda())      # input = (x, image_id)
                data = input[0]
                image_id = input[1]

                with torch.no_grad():
                    output = model.get_original_image_feature(data)

                for i in range(output.size(0)):
                    _iid = image_id[i]
                    _feat = output[i].squeeze().cpu().numpy()
                    index_feats.append(_feat)
                    index_ids.append(_iid)

            index_feats = np.asarray(index_feats)

            # get manipulated image features of query.
            query_ids = []
            query_feats = []
            print('Extract Query Features...')
            test_loader.dataset.set_mode('query')
            for bidx, input in enumerate(tqdm(test_loader, desc='Query')):
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
                    query_feats.append(_feat)
                    query_ids.append(_qid)
            
            query_feats = np.asarray(query_feats)
                    
            # calculate cosine similarity
            print('calculating cosine similarity score...')
            y_score = np.dot(query_feats, index_feats.T)
            y_indices = np.argsort(-1 * y_score, axis=1)

            _hyperopt = {
                'score': y_score,
                'query_ids': query_ids,
                'index_ids': index_ids
            }
            hyperopt[target] = _hyperopt

            _score = []
            for qidx, query_id in enumerate(tqdm(query_ids)):
                _r = []
                for j in range(min(TOP_K, y_score.shape[1])):
                    index = y_indices[qidx, j]
                    _r.append([
                        str(index_ids[index]),
                        float(y_score[qidx, index])
                    ])
                _score.append([query_id, _r])
            score[target] = _score

        # save score to file.
        print(f'Dump top-{TOP_K} rankings to .pkl file ...')
        os.makedirs(f'./output_score/{date_key}_{args.expr_name}', exist_ok=True)
        with open(os.path.join(f'./output_score/{date_key}_{args.expr_name}', f'hyperopt.{SPLIT}.pkl'), 'wb') as f:
            pickle.dump(hyperopt, f)
        with open(os.path.join(f'./output_score/{date_key}_{args.expr_name}', f'score.{SPLIT}.pkl'), 'wb') as f:
            pickle.dump(score, f)
        print('Done.')


if __name__ == "__main__":
    # args for region
    parser = argparse.ArgumentParser('Test')
    
    # Common options.
    parser.add_argument('--gpu_id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--manualSeed', type=int, default=int(time.time()), help='manual seed')
    parser.add_argument('--data_root', required=True, type=str, help='experiment name')
    parser.add_argument('--expr_name', default='tirg', type=str, help='experiment name')
    
    ## parse and save args.
    args, _ = parser.parse_known_args()

    # main.
    main(args)
