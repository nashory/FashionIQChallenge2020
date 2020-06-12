'''
Perform Bayes Optimization on the scores.
'''

import os
import sys
sys.path.append('../train')

import glob
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import uuid
from datetime import datetime
import urllib.request

from hyperopt import hp, tpe, fmin
from irbench.evals.eval_helper import EvalHelper


def objective_fn(W, args):
    W = np.array(W) / sum(W)
    
    # Weighted sum of score matrix
    final_matrix = {}
    for i, repo in enumerate(args.repos):
        try:
            pkl_path = os.path.join("output_score", repo, f"hyperopt.val.pkl")
            assert os.path.exists(pkl_path)
            saved_scores = pickle.load(open(pkl_path, 'rb'))
        except Exception as err:
            raise OSError(f"{pkl_path}, {err}")
                
        for target, node in saved_scores.items():
            tag = f'{target}_score'
            if tag in final_matrix:
                final_matrix[tag] += node['score'] * W[i]
            else:
                final_matrix[tag] = node['score'] * W[i]

    # Get metric result
    r10 = 0.
    r50 = 0.
    r10r50 = 0.

    for target in saved_scores.keys():

        # Rendering topk result
        y_score = final_matrix[f'{target}_score']  # (num of query, num of index)
        y_indices = np.argsort(-1 * y_score, axis=1)[:, :args.topk]  # (num of query, topk)

        res = {}
        for qidx, query_id in enumerate(saved_scores[target]['query_ids']):
            res[query_id] = []
            for j in range(args.topk):
                index = y_indices[qidx, j]
                jth_rank_id = saved_scores[target]['index_ids'][index]
                res[query_id].append(jth_rank_id)

        from src.dataset import FashionIQValIDDataset
        val_dataset = FashionIQValIDDataset(
            data_root=args.data_root,
            split='val',
            target=target
        )
        eval_helper = EvalHelper()

        # Feeding GT
        for bidx, input in enumerate(val_dataset):
            """
            input[0] = candidate_id
            input[1] = target_id
            input[2] = we_key
            """
            _w_key = input[2]
            _tid = input[1]

            eval_helper.feed_gt([_w_key, [_tid]])

        eval_helper.feed_rank_from_dict(res)
        score = eval_helper.evaluate(metric=['top_k_acc'], kappa=[10, 50])
        _r10 = score[0][str(10)]['top_k_acc']
        _r50 = score[0][str(50)]['top_k_acc']
        _r10r50 = 0.5 * (score[0][str(10)]['top_k_acc'] + score[0][str(50)]['top_k_acc'])
        r10 += _r10
        r50 += _r50
        r10r50 += _r10r50

    r10 /= 3
    r50 /= 3
    r10r50 /= 3
    print(f"[{W}] r10: {r10}, r50: {r50}, r10r50: {r10r50}")

    return -1 * r10r50  # flip sign for maximize


def main(args):
    args.repos = args.repos.strip().split(",")

    space = [hp.uniform(f'w{i}', 0, 1) for i in range(len(args.repos))]
    best = fmin(fn=lambda W: objective_fn(W, args),
                space=space,
                algo=tpe.suggest,
                max_evals=args.max_eval)
    print(f"best: {best}")


    date_key = str(datetime.now().strftime("%Y%m%d%H%M"))[2:]

    SPLITS = ['val', 'test']
    for SPLIT in SPLITS:
        print(f'Save final results for {SPLIT}...')
        os.system('mkdir -p output_optimize')
        final_score = dict()
        for idx, repo in enumerate(args.repos):
            try:
                pkl_path = os.path.join("output_score", repo, f"hyperopt.{SPLIT}.pkl")
                assert os.path.exists(pkl_path)
                saved_scores = pickle.load(open(pkl_path, 'rb'))
            except Exception as err:
                raise OSError(f"{pkl_path}, {err}")
           
            for target, node in saved_scores.items():
                if not target in final_score:
                    final_score[target] = {
                        'score': None,
                        'query_ids': [],
                        'index_ids': [],
                    }
                w = float(best['w{}'.format(idx)])
                if idx == 0:
                    final_score[target]['score'] = w * node['score']
                else:
                    final_score[target]['score'] += w * node['score']
                final_score[target]['query_ids'] = node['query_ids']
                final_score[target]['index_ids'] = node['index_ids']
        
        # save optimized hyperopt pkl file.
        os.makedirs(f'./output_optimize/{date_key}', exist_ok=True)
        with open(f'./output_optimize/{date_key}/hyperopt.{SPLIT}.pkl', 'wb') as f:
            pickle.dump(final_score, f)

        TOP_K=50
        result = dict()
        for target, v in final_score.items():
            _result = []
            query_ids = v['query_ids']
            index_ids = v['index_ids']
            y_score = v['score']
            y_indices = np.argsort(-1 * y_score, axis=1)
            for qidx, query_id in enumerate(tqdm(query_ids)):
                _r = []
                for j in range(min(TOP_K, y_score.shape[1])):
                    index = y_indices[qidx, j]
                    _r.append([
                        str(index_ids[index]),
                        float(y_score[qidx, index])
                    ])
                _result.append([query_id, _r])
            result[target] = _result

        with open(f'./output_optimize/{date_key}/score.{SPLIT}.pkl', 'wb') as f:
            pickle.dump(result, f)

if __name__ == '__main__':
    # args for region
    parser = argparse.ArgumentParser('Bayes Optimization on the scores')

    # Common options.
    parser.add_argument('--data_root', required=True, type=str)
    parser.add_argument('--repos', default='devel', type=str)
    parser.add_argument('--topk', default='50', type=int)
    parser.add_argument('--max_eval', default='100', type=int)

    ## parse and save args.
    args, _ = parser.parse_known_args()

    # main.
    main(args)
