# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from typing import Dict

import torch
from torch import optim
import numpy as np

from datasets import Dataset
from models import CP, ComplEx, ComplEx_NNE, ComplEx_logicNN
from regularizers import F2, N3
from optimizers import KBCOptimizer

from datetime import datetime


torch.cuda.empty_cache()

big_datasets = ['FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10', 'family']
datasets = big_datasets

parser = argparse.ArgumentParser(
    description="Relational learning contraption"
)

parser.add_argument(
    '--dataset', choices=datasets,
    help="Dataset in {}".format(datasets)
)

models = ['CP', 'ComplEx', 'ComplEx_NNE', 'ComplEx_logicNN']
parser.add_argument(
    '--model', choices=models,
    help="Model in {}".format(models)
)

regularizers = ['N3', 'F2']
parser.add_argument(
    '--regularizer', choices=regularizers, default='N3',
    help="Regularizer in {}".format(regularizers)
)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument(
    '--optimizer', choices=optimizers, default='Adagrad',
    help="Optimizer in {}".format(optimizers)
)

parser.add_argument(
    '--max_epochs', default=50, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid', default=3, type=float,
    help="Number of epochs before valid."
)
parser.add_argument(
    '--rank', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--reg', default=0, type=float,
    help="Regularization weight"
)
parser.add_argument(
    '--init', default=1e-3, type=float,
    help="Initial scale"
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--decay1', default=0.9, type=float,
    help="decay rate for the first moment estimate in Adam"
)
parser.add_argument(
    '--decay2', default=0.999, type=float,
    help="decay rate for second moment estimate in Adam"
)
args = parser.parse_args()
print("\n======> Parameter settings: " + str(args))
dataset = Dataset(args.dataset)
examples = torch.from_numpy(dataset.get_train().astype('int64'))
# print("\n======> Number of training triples: " + str(examples.size()))

######## used for rule injection
rule_path = '/home/ComplEx-Inject/kbc/src_data/' + args.dataset + '/kbc_id_cons.txt'
print("\n======> Rule set path: " + str(rule_path))
#### read in & convert rule ids and confidence
# get number of predicates
r_num = dataset.get_shape()[1] // 2
print ("\n======> Number of relations: " + str(r_num))
print(dataset.get_shape())

model = {
    'CP': lambda: CP(dataset.get_shape(), args.rank, args.init),
    'ComplEx': lambda: ComplEx(dataset.get_shape(), args.rank, args.init),
    'ComplEx_NNE': lambda: ComplEx_NNE(dataset.get_shape(), args.rank, [], args.init, 10),
    'ComplEx_logicNN': lambda: ComplEx_logicNN(dataset.get_shape(), args.rank, [], args.init, 6, [], [0.95, 0]),
}[args.model]()

regularizer = {
    'F2': F2(args.reg),
    'N3': N3(args.reg),
}[args.regularizer]

device = 'cuda'
model.to(device)

optim_method = {
    'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
    'Adam': lambda: optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.decay1, args.decay2)),
    'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate)
}[args.optimizer]()

# only used for logicNN
n_train_batches = examples.shape[0] / args.batch_size
print("training batch number: " + str(n_train_batches))
optimizer = KBCOptimizer(model, regularizer, optim_method, args.batch_size, n_train_batches)

def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MRR': m, 'hits@[1,3,10]': h}

print("======> Checking model type")
if isinstance(optimizer.model, ComplEx_NNE) or isinstance(optimizer.model, ComplEx_logicNN):
    print('model is complex-nne or ComplEx_logicNN')
    # set mu value
    if isinstance(optimizer.model, ComplEx_NNE):
        init_mu = model.mu
    # extract rule info
    kbc_id_conf_f = rule_path
    r_p_list = []
    r_q_list = []
    conf_list = []
    neg_list = []
    with open(kbc_id_conf_f, 'r') as f:
        while True:
            line = f.readline()
            if line:
                # two relations split by ',', confidence split by tab
                line.replace('\n', '')
                r_p = int(line.split(',')[0])
                if r_p < 0:
                    # print("ignore negative rules")
                    neg_list.append(-1)
                    r_p = -r_p
    #                 if r_p >= r_num:
    #                     print("suspicious relation id: " + str(r_p))
                else:
                    neg_list.append(1)
                r_q = int(line.split(',')[1].split('\t')[0])
                conf = float(line.split('\t')[1])
                # print(rel0, rel1, conf)
                r_p_list.append(r_p)
                r_q_list.append(r_q)
                conf_list.append(conf)
                # print(r_p, r_q, conf)
            else:
                break
    rule_list = [r_p_list, r_q_list, conf_list, neg_list]
    optimizer.model.rule_list = rule_list

# Extract features for teach student network
# for each rule, create its own feature and feature idx
if isinstance(optimizer.model, ComplEx_logicNN):
    # create new triples(features) based on related training triples
    # store idx
    entailment_triples = []
    entailment_idx = []
    for (r_p, r_q) in zip(r_p_list, r_q_list):
        rule_entailment_triples = set()
        rule_entailment_idx = np.zeros(examples.shape[0])
        for i, (e1, r, e2) in enumerate(examples):
            if r == r_p:
                if ((e1, r, e2), (e1, r_q, e2)) not in rule_entailment_triples:
                    rule_entailment_triples.add(((e1, r, e2), (e1, r_q, e2)))
                    rule_entailment_idx[i] = 1
        entailment_triples.append(rule_entailment_triples)
        entailment_idx.append(rule_entailment_idx)
    optimizer.model.rule_feas = entailment_triples
    print(len(entailment_triples), len(entailment_idx))

cur_loss = 0
curve = {'train': [], 'valid': [], 'test': []}
for e in range(args.max_epochs):
    if isinstance(optimizer.model, ComplEx_logicNN):
        print("Epoch {}: ".format(e))
        print("   pi: {}".format(optimizer.pi))

    if (e == 30) or (e == 70):
        if isinstance(optimizer.model, ComplEx_NNE):
            model.mu = 2 * model.mu
    cur_loss = optimizer.epoch(examples)


    if (e + 1) % args.valid == 0:
        valid, test, train = [
            avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
            for split in ['valid', 'test', 'train']
        ]

        curve['valid'].append(valid)
        curve['test'].append(test)
        curve['train'].append(train)

        print("\t TRAIN: ", train)
        print("\t VALID : ", valid)
        print("\t TEST : ", test)


now = datetime.now()
time_stamp = str(now)[:19].replace(':','-').replace(' ', '_')
# write relation embeddings of each rule at the end
if isinstance(optimizer.model, ComplEx_NNE):
    all_rules_real_path = time_stamp + '_' + args.dataset + '_' + args.model + '_mu_' + str(init_mu) + '_real_compare' + '.txt'
    all_rules_img_path = time_stamp + '_' + args.dataset + '_' + args.model + '_mu_' + str(init_mu) + '_img_compare' + '.txt'
    model_path = '/home/ComplEx-Inject/saved_models/' + time_stamp + '_' + args.dataset + '_' + args.model + '_mu_' + str(init_mu) + '.pkl'
else:
    all_rules_real_path = time_stamp + '_' + args.dataset + '_' + args.model + '_real_compare' + '.txt'
    all_rules_img_path = time_stamp + '_' + args.dataset + '_' + args.model + '_img_compare' + '.txt'
    model_path = '/home/ComplEx-Inject/saved_models/' + time_stamp + '_' + args.dataset + '_' + args.model + '.pkl'

# model_path = '/home/ComplEx-Inject/saved_models/' + args.model + '_' + args.dataset + '_mu_' + str(model.mu) + '_' + time_stamp + '.pkl'
results = dataset.eval(model, 'test', -1)
# torch.save((model,dataset), model_path)
print("\n\nTEST : ", results)