# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from pathlib import Path
import pkg_resources
import pickle
from typing import Dict, Tuple, List

import numpy as np
import torch
from models import KBCModel


# DATA_PATH = Path(pkg_resources.resource_filename('kbc', 'data/'))
DATA_PATH = Path('/blue/daisyw/ma.haodi/ComplEx-Inject/kbc/data/')


class Dataset(object):
    def __init__(self, name: str):
        self.root = DATA_PATH / name

        self.data = {}
        for f in ['train', 'test', 'valid']:
            in_file = open(str(self.root / (f + '.pickle')), 'rb')
            self.data[f] = pickle.load(in_file)
        # print(type(self.data['train']))
        # print(self.data['train'])

        # select certain percentage of training data
        # total_train = len(self.data['train'])
        # train_percent = 0.05
        # train_idx = int(total_train * train_percent)
        # print("original training number: " + str(total_train))
        # self.data['train'] = self.data['train'][train_idx : train_idx * 2]
        # print("sample training number: " + str(len(self.data['train'])))
        
        # entity and relation numbers
        # maxis = np.max(self.data['train'], axis=0)
        # self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        # self.n_predicates = int(maxis[1] + 1)
        # self.n_predicates *= 2
        # read from id file
        self.n_entities = 0
        self.n_predicates = 0
        self.rel_id = {}
        self.ent_id = {}
        with open(str(self.root / 'ent_id')) as f:
            while True:
                line = f.readline()
                if line:
                    ent_n, ent_idx = line.split('\t')
                    self.ent_id[ent_n] = int(ent_idx)
                    self.n_entities += 1
                else:
                    break
        with open(str(self.root / 'rel_id')) as f:
            while True:
                line = f.readline()
                if line:
                    rel_n, rel_idx = line.split('\t')
                    self.rel_id[rel_n] = int(rel_idx)
                    self.n_predicates += 1
                else:
                    break
        # self.n_predicates *= 2
        # print ("\n======> Number of entities and 2*relations: " + str(self.n_entities) + ' ' + str(self.n_predicates))

        inp_f = open(str(self.root / f'to_skip.pickle'), 'rb')
        self.to_skip: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
        inp_f.close()

    def get_examples(self, split):
        return self.data[split]

    def get_train(self):
        copy = np.copy(self.data['train'])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        # copy[:, 1] += self.n_predicates // 2  # has been multiplied by two.
        # return np.vstack((self.data['train'], copy))
        return self.data['train']

    def eval(
            self, model: KBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both',
            at: Tuple[int] = (1, 3, 10)
    ):
        test = self.get_examples(split)
        examples = torch.from_numpy(test.astype('int64')).cuda()
        missing = [missing_eval]
        if missing_eval == 'both':
            missing = ['rhs', 'lhs']

        mean_reciprocal_rank = {}
        hits_at = {}

        for m in missing:
            q = examples.clone()
            if n_queries > 0:
                permutation = torch.randperm(len(examples))[:n_queries]
                q = examples[permutation]
            if m == 'lhs':
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                # q[:, 1] += self.n_predicates // 2
            ranks = model.get_ranking(q, self.to_skip[m], batch_size=500)
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                at
            ))))
        # print("# of triples in ranking: " + str(len(ranks)))

        return mean_reciprocal_rank, hits_at

    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities
    
    def rank_result(
            self, model: KBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both',
            at: Tuple[int] = (1, 3, 10)
    ):
        test = self.get_examples(split)
        examples = torch.from_numpy(test.astype('int64')).cuda()
        missing = [missing_eval]
        if missing_eval == 'both':
            missing = ['rhs', 'lhs']

        rank_res = []
        for m in missing:
            q = examples.clone()
            if n_queries > 0:
                permutation = torch.randperm(len(examples))[:n_queries]
                q = examples[permutation]
            if m == 'lhs':
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                # q[:, 1] += self.n_predicates // 2
            ranks = model.get_ranking(q, self.to_skip[m], batch_size=500)
            rank_res.append(ranks)

        return rank_res

    def generateHeadNegTriple(self, m_NumNeg, triple):
        iPosHead = triple[0]
        iPosTail = triple[2]
        iPosRelation = triple[1]
        iNumberOfEntities = self.get_shape()[0]		
		
        NegativeTripleSet = set()		
        while (len(NegativeTripleSet) < m_NumNeg):
            iNegHead = iPosHead
            NegativeTriple = (iNegHead, iPosRelation.data, iPosTail.data)
            while (iNegHead == iPosHead):
                iNegHead = np.random.randint(iNumberOfEntities)
                NegativeTriple = (iNegHead, int(iPosRelation.data), int(iPosTail.data))
                # print('new triple generated: ' + str(NegativeTriple))
            NegativeTripleSet.add(NegativeTriple)
        return NegativeTripleSet

    def generateTailNegTriple(self, m_NumNeg, triple):
        iPosHead = triple[0]
        iPosTail = triple[2]
        iPosRelation = triple[1]
        iNumberOfEntities = self.get_shape()[0]		
		
        NegativeTripleSet = set()		
        while (len(NegativeTripleSet) < m_NumNeg):
            iNegTail = iPosTail
            NegativeTriple = (iPosHead.data, iPosRelation.data, iNegTail)
            while (iNegTail == iPosTail):
                iNegTail = np.random.randint(iNumberOfEntities)
                NegativeTriple = (int(iPosHead.data), int(iPosRelation.data), iNegTail)
            NegativeTripleSet.add(NegativeTriple)

        return NegativeTripleSet