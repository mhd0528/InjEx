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
import pickle
import os

from datasets import Dataset
from models import CP, ComplEx, InjEx
from regularizers import F2, N3
from optimizers import KBCOptimizer

from datetime import datetime


torch.cuda.empty_cache()

big_datasets = ['FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10', 'family', 'NELL', 'Wiki']
datasets = big_datasets

parser = argparse.ArgumentParser(
    description="Relational learning contraption"
)

parser.add_argument(
    '--dataset', choices=datasets,
    help="Dataset in {}".format(datasets)
)

models = ['CP', 'ComplEx', 'InjEx', 'ComplEx_logicNN', 'ComplEx_supportNN']
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
parser.add_argument(
    '--rule_type', default=0, type=int,
    help="Rule type for injection:\n\t0: combine entailment and composition \n\t1: entailment \n\t4: type 4(composition)"
)
parser.add_argument(
    '--mu_1', default=0.1, type=float,
    help="weight for the teacher (rules)"
)
parser.add_argument(
    '--mu_2', default=0.1, type=float,
    help="weight for the teacher (rules)"
)

args = parser.parse_args()
print("\n======> Parameter settings: " + str(args))
dataset = Dataset(args.dataset)
examples = torch.from_numpy(dataset.get_train().astype('int64'))
print("\n======> Number of training triples: " + str(examples.size()))
print("======> triples example: " + str(examples[0]))

######## used for rule injection
dataset_path = '/home/ComplEx-Inject/kbc/src_data/' + args.dataset
# rule_path = '/home/ComplEx-Inject/kbc/src_data/' + args.dataset + '/kbc_id_cons.txt'
#### read in & convert rule ids and confidence
# get number of predicates
ent_num = dataset.get_shape()[0]
r_num = dataset.get_shape()[1]
print ("\n======> Number of entities and relations: " + str(ent_num) + ' ' + str(r_num))
# print(dataset.get_shape())

#### now all the rules (after read in) are r_q <= (r_p1, r_p2, ...), conf, direction_flag
def rule_reader(dataset_path, rule_type, train_data, ent_num):
    if rule_type == 0:
        rule_list = []
        rule_list.append(rule_reader(dataset_path, 1, train_data, ent_num))
        rule_list.append(rule_reader(dataset_path, 4, train_data, ent_num))
    elif rule_type == 1:
        kbc_id_conf_f = dataset_path + '/cons.txt'
        rule_list = []
        # r_num = dataset.get_shape()[1] // 2
        r_num = dataset.get_shape()[1]
        with open(kbc_id_conf_f, 'r') as f:
            while True:
                line = f.readline()
                if line:
                    flag = 1
                    # two relations split by ',', confidence split by tab
                    line.replace('\n', '')
                    if line[0] == '-': # negative rules
                        flag = -1
                        line = line[1:]
                    tokens = line.split('\t')
                    r_p = int(tokens[0].split(',')[0])
                    r_q = int(tokens[0].split(',')[1])
                    conf = float(tokens[1])
                    # if 0.9 <= conf < 1.0:
                    if conf >= 0.8:
                        rule_list.append((r_q, r_p, conf, flag))
                else:
                    break
        # rule_list = rule_list[:152]
    elif rule_type == 4:
        kbc_id_conf_f = dataset_path + '/cons_4.txt'
        rule_list = []
        with open(kbc_id_conf_f, 'r') as f:
            while True:
                line = f.readline()
                if line:
                    # format: p, q, r, conf: triple_ids, split by tab
                    # turn into: p, q, r, conf, tuples (e1, q, e2)
                    line.replace('\n', '')
                    rels = line.split('\t')[0]
                    # print(rels)
                    rel_tup = list(map(int, rels.split(',')))
                    conf = float(line.split('\t')[1])
                    rule_list.append((rel_tup[0], rel_tup[1:], conf))
                else:
                    break
    # return rule_list[200:300]
    return rule_list

print("======> Checking model type: " + args.model)
if args.model == 'InjEx' or args.model == 'ComplEx_logicNN':
    print('model is complex-nne or InjEx')
    # extract rule info
    rule_list = rule_reader(dataset_path, args.rule_type, examples, dataset.get_shape()[0])
    if args.rule_type:
        print ("\n======> Number of rules: " + str(len(rule_list)))
        print((rule_list[0]))
    ## combination rule injection
    else:
        # rule_list[0] = rule_list[0][:len(rule_list[0]) // 2 + 1]
        print ("\n======> Number of entailment rules: " + str(len(rule_list[0])))
        print ("\n======> Number of composition rules: " + str(len(rule_list[1])))
        print((rule_list[0][0]))

model = {
    'CP': lambda: CP(dataset.get_shape(), args.rank, args.init),
    'ComplEx': lambda: ComplEx(dataset.get_shape(), args.rank, args.init),
    'InjEx': lambda: InjEx(dataset.get_shape(), args.rank, rule_list, args.init, args.mu_1, args.mu_2, args.rule_type),
    # 'ComplEx_logicNN': lambda: ComplEx_logicNN(dataset.get_shape(), args.rank, rule_list, args.init, 6, [], [0.95, 0]),
    # 'ComplEx_supportNN': lambda: ComplEx_supportNN(sizes=dataset.get_shape(), rank=args.rank, init_size=args.init, mu=0.01, feas={}, sup=[])
}[args.model]()

device = 'cuda'
torch.cuda.empty_cache()
model.to(device)
regularizer = {
    'F2': F2(args.reg),
    'N3': N3(args.reg),
}[args.regularizer]


optim_method = {
    'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
    'Adam': lambda: optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.decay1, args.decay2)),
    'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate)
}[args.optimizer]()

######## preprocess for different models

## load in a pretrained model (use the pre-trained embeddings)
# (old_model, old_data) = torch.load('saved_models/2021-09-01_21-13-27_FB237_ComplEx.pkl')
# model.embeddings = old_model.embeddings
# print("finish loading pre-trained embeddings")
# print("======> This model: 30 epochs: origianl data; 90 epochs: new data; extract top 0.05% triples (add reciprocal) at 30 and 90, switch training every 5 epochs after 30 epochs")
# print("======> This model use ComplEx for the first 30 epochs and use new extracted data for other 70 epochs with reciprocal triples added")
## only used for logicNN
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


cur_loss = 0
curve = {'train': [], 'valid': [], 'test': []}
#### InjEx
## set mu value
if args.model == 'InjEx':
    init_mu_1 = model.mu_1
    init_mu_2 = model.mu_2
for e in range(args.max_epochs):
    # if (e == 30) or (e == 70):
    #     if isinstance(optimizer.model, InjEx):
    #         model.mu_1 = 0.5 * model.mu_1
    #         model.mu_2 = 0.5 * model.mu_2
    if isinstance(optimizer.model, InjEx):
        if (e == 10):
            model.mu_1 = 0#.5 * model.mu_1
            model.mu_2 = 0#.5 * model.mu_2
        # elif (e == 80):
        #     model.mu_1 = init_mu_1
        #     model.mu_2 = init_mu_2

    cur_loss = optimizer.epoch(examples, args.rule_type)

    if (e) % args.valid == 0:
        valid, test, train = [
            avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
            for split in ['valid', 'test', 'train']
        ]

        curve['valid'].append(valid)
        curve['test'].append(test)
        curve['train'].append(train)

        print("epoch " + str(e) + ':')
        print("\t TRAIN: ", train)
        print("\t VALID : ", valid)
        print("\t TEST : ", test)


now = datetime.now()
time_stamp = str(now)[:19].replace(':','-').replace(' ', '_')
# write relation embeddings of each rule at the end
if isinstance(optimizer.model, InjEx):
    all_rules_real_path = time_stamp + '_' + args.dataset + '_' + args.model + '_rule_' + str(args.rule_type) + '_real_compare' + '.txt'
    all_rules_img_path = time_stamp + '_' + args.dataset + '_' + args.model + '_rule_' + str(args.rule_type) + '_img_compare' + '.txt'
    model_path = '/home/ComplEx-Inject/saved_models/' + time_stamp + '_' + args.dataset + '_' + args.model + '_rule_' + str(args.rule_type) + '.pkl'
else:
    all_rules_real_path = time_stamp + '_' + args.dataset + '_' + args.model + '_real_compare' + '.txt'
    all_rules_img_path = time_stamp + '_' + args.dataset + '_' + args.model + '_img_compare' + '.txt'
    model_path = '/home/ComplEx-Inject/saved_models/' + time_stamp + '_' + args.dataset + '_' + args.model + '.pkl'

results = dataset.eval(model, 'test', -1)
torch.save((model,dataset), model_path)
print("\n\nTEST : ", results)