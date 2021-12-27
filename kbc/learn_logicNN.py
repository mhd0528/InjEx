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
parser.add_argument(
    '--rule_type', default=0, type=int,
    help="Rule type for injection:\n\t0: entailment \n\t4: type 4"
)
args = parser.parse_args()
print("\n======> Parameter settings: " + str(args))
dataset = Dataset(args.dataset)
examples = torch.from_numpy(dataset.get_train().astype('int64'))
print("\n======> Number of training triples: " + str(examples.size()))

######## used for rule injection
dataset_path = '/home/ComplEx-Inject/kbc/src_data/' + args.dataset
# rule_path = '/home/ComplEx-Inject/kbc/src_data/' + args.dataset + '/kbc_id_cons.txt'
# print("\n======> Rule set path: " + str(rule_path))
#### read in & convert rule ids and confidence
# get number of predicates
ent_num = dataset.get_shape()[0]
r_num = dataset.get_shape()[1]
print ("\n======> Number of entities and relations: " + str(ent_num) + ' ' + str(r_num))
# print(dataset.get_shape())

def rule_reader(dataset_path, rule_type, train_data, ent_num):
    if args.rule_type == 0:
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
                    # print(r_p, r_q, conf)
                    rule_list.append((r_p, r_q, conf, flag))
                else:
                    break
    elif args.rule_type == 4:
        kbc_id_conf_f = dataset_path + '/all_cons_4.txt'
        rule_list = []
        with open(kbc_id_conf_f, 'r') as f:
            while True:
                line = f.readline()
                if line:
                    # format: p, q, r, conf: triple_ids, split by tab
                    # turn into: p, q, r, conf, tuples (e1, q, e2)
                    line.replace('\n', '')
                    rels = line.split('\t')[0]
                    rel_tup = list(map(int, rels.split(',')))
                    conf = float(line.split('\t')[1])
                    triple_ids = map(int, line.split('\t')[2].split(' '))
                    tuple_list = []
                    for id in triple_ids:
                        tuple = examples[id]
                        tuple_list.append(tuple)
                    rule_list.append((rel_tup, conf, tuple_list))
                else:
                    break
    return rule_list

print("======> Checking model type")
if args.model == 'ComplEx_NNE' or args.model == 'ComplEx_logicNN':
    print('model is complex-nne or ComplEx_logicNN')
    # extract rule info
    rule_list = rule_reader(dataset_path, args.rule_type, examples, dataset.get_shape()[0])
    print ("\n======> Number of rules: " + str(len(rule_list)) + str(rule_list[0]))
    print(rule_list)

model = {
    'CP': lambda: CP(dataset.get_shape(), args.rank, args.init),
    'ComplEx': lambda: ComplEx(dataset.get_shape(), args.rank, args.init),
    'ComplEx_NNE': lambda: ComplEx_NNE(dataset.get_shape(), args.rank, rule_list, args.init, 1),
    'ComplEx_logicNN': lambda: ComplEx_logicNN(dataset.get_shape(), args.rank, rule_list, args.init, 6, [], [0.95, 0]),
}[args.model]()
# set mu value
if args.model == 'ComplEx_NNE':
    init_mu = model.mu

# Extract features for teach student network with entailment rules
# for each rule, create its own feature and feature idx
if isinstance(model, ComplEx_logicNN):
    if args.rule_type == 0:
        model.fea_generator(args.rule_type, examples, ent_num)
# load in a pretrained model (use the pre-trained embeddings)
# (old_model, old_data) = torch.load('saved_models/2021-09-01_21-13-27_FB237_ComplEx.pkl')
# model.embeddings = old_model.embeddings
# print("finish loading pre-trained embeddings")
# print("======> This model: 30 epochs: origianl data; 90 epochs: new data; extract top 0.05% triples (add reciprocal) at 30 and 90, switch training every 5 epochs after 30 epochs")
# print("======> This model use ComplEx for the first 30 epochs and use new extracted data for other 70 epochs with reciprocal triples added")

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


cur_loss = 0
curve = {'train': [], 'valid': [], 'test': []}
for e in range(args.max_epochs):
    # if isinstance(optimizer.model, ComplEx_logicNN):
    #     # print("Epoch {}: ".format(e))
    #     # print("   pi: {}".format(optimizer.pi))
    #     # generate rule related training data with type 4 rules
    #     if args.rule_type:
    #         # if e % 20 == 0:
    #         if e == 30 or e == 90:
    #             optimizer.model.fea_generator(args.rule_type, examples, ent_num)
    #             new_examples = optimizer.model.rule_feas
    #             print("generating rule_related data: " + str(new_examples.size()))

    if (e == 30) or (e == 70):
        if isinstance(optimizer.model, ComplEx_NNE):
            model.mu = 2 * model.mu

    # if e > 30 and e < 90:
    #     if (e // 5) % 2:
    #         cur_loss = optimizer.epoch(examples, args.rule_type)
    #     else:
    #         cur_loss = optimizer.epoch(new_examples, args.rule_type)
    # elif e > 90:
    #     cur_loss = optimizer.epoch(new_examples, args.rule_type)
    # else:
    #     cur_loss = optimizer.epoch(examples, args.rule_type)
    cur_loss = optimizer.epoch(examples, args.rule_type)

    if (e + 1) % args.valid == 0:
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
        # print("\t TEST : ", test)


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

results = dataset.eval(model, 'test', -1)
# torch.save((model,dataset), model_path)
print("\n\nTEST : ", results)