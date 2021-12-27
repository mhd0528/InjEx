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
from models import CP, ComplEx, ComplEx_NNE
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

models = ['CP', 'ComplEx', 'ComplEx_NNE']
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
print("\n======> Number of training triples: " + str(examples.size()))

######## used for rule injection
rule_path = '/blue/daisyw/ma.haodi/ComplEx-Inject/kbc/src_data/' + args.dataset + '/cons.txt'
#### read in & convert rule ids and confidence
# get number of predicates
print ("\n======> Number of entities and relations: " + str(dataset.get_shape()))

if args.model == 'ComplEx_NNE':
    kbc_id_conf_f = rule_path
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
    print ("\n======> Number of rules: " + str(len(rule_list)) + str(rule_list[0]))
    print(rule_list)

model = {
    'CP': lambda: CP(dataset.get_shape(), args.rank, args.init),
    'ComplEx': lambda: ComplEx(dataset.get_shape(), args.rank, args.init),
    'ComplEx_NNE': lambda: ComplEx_NNE(dataset.get_shape(), args.rank, rule_list, args.init, 1),
}[args.model]()
print("======> This model: ComplEx_NNE_AER; train: synthesized experiments, validating with no r_q in training set, 12.07 data; less embedding rank and larger batch")
# print("======> This model: original ComplEx; train: synthesized experiments, 12.07 data; less embedding rank and larger batch, no reciprocal setting, F2 regularizer")

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

optimizer = KBCOptimizer(model, regularizer, optim_method, args.batch_size)

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
if isinstance(optimizer.model, ComplEx_NNE):
    init_mu = model.mu
    print('model is complex-nne')

cur_loss = 0
curve = {'train': [], 'valid': [], 'test': []}
for e in range(args.max_epochs):
    if (e == 30) or (e == 70):
        if isinstance(optimizer.model, ComplEx_NNE):
            model.mu = 2 * model.mu
    cur_loss = optimizer.epoch(examples)
    # cur_loss = optimizer.epoch_2(examples, dataset)
    
    if (e + 1) % args.valid == 0:
        valid, test, train = [
            avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
            # for split in ['valid', 'test', 'train']
            for split in ['valid', 'test', 'train']
        ]

        curve['valid'].append(valid)
        curve['test'].append(test)
        curve['train'].append(train)

        print("\t TRAIN: ", train)
        print("\t VALID : ", valid)


now = datetime.now()
time_stamp = str(now)[:19].replace(':','-').replace(' ', '_')
# write relation embeddings of each rule at the end
if isinstance(optimizer.model, ComplEx_NNE):
    all_rules_real_path = time_stamp + '_' + args.dataset + '_' + args.model + '_mu_' + str(init_mu) + '_real_compare' + '.txt'
    all_rules_img_path = time_stamp + '_' + args.dataset + '_' + args.model + '_mu_' + str(init_mu) + '_img_compare' + '.txt'
    model_path = '/blue/daisyw/ma.haodi/ComplEx-Inject/saved_models/' + time_stamp + '_' + args.dataset + '_' + args.model + '_mu_' + str(init_mu) + '.pkl'
else:
    all_rules_real_path = time_stamp + '_' + args.dataset + '_' + args.model + '_real_compare' + '.txt'
    all_rules_img_path = time_stamp + '_' + args.dataset + '_' + args.model + '_img_compare' + '.txt'
    model_path = '/blue/daisyw/ma.haodi/ComplEx-Inject/saved_models/' + time_stamp + '_' + args.dataset + '_' + args.model + '.pkl'


# real_file = open(all_rules_real_path, "a")
# img_file = open(all_rules_img_path, "a")

# rel = optimizer.model.embeddings[1]
# idx_p = torch.LongTensor(rule_list[0]).cuda()
# idx_q = torch.LongTensor(rule_list[1]).cuda()
# #print(torch.max(idx_p))

# r_p_ebds = rel(idx_p)
# r_p_ebds = r_p_ebds[:, :optimizer.model.rank], r_p_ebds[:, optimizer.model.rank:]
# r_q_ebds = rel(idx_q)
# r_q_ebds = r_q_ebds[:, :optimizer.model.rank], r_q_ebds[:, optimizer.model.rank:]
# img_difference = []
# rel_difference = []
# for i in range(len(rule_list[0])):
#     # for all rules, real(r_p) < real(r_q)
#     p_lt_q = torch.lt(r_p_ebds[0][i], r_q_ebds[0][i]).tolist()
#     real_file.write(', '.join(map(str, p_lt_q)) + '\n')
#     rel_difference.append(torch.sum(torch.abs(r_p_ebds[0][i] - r_q_ebds[0][i])).item())
#     # for negative entailment rules, img(r_p) == -img(r_q), calculate the difference
#     if rule_list[3][i] < 0:
#         img_tensor = torch.square(r_p_ebds[1][i] + r_q_ebds[1][i])
#         img_file.write(', '.join(map(str, img_tensor.tolist())) + '\n')
#         img_difference.append(torch.sum(img_tensor).item())
#     # for entailment rules, img(r_p) == img(r_q), calculate the difference
#     else:
#         img_tensor = torch.square(r_p_ebds[1][i] - r_q_ebds[1][i])
#         img_file.write(','.join(map(str, img_tensor.tolist())) + '\n')
#         img_difference.append(torch.sum(img_tensor).item())

# img_file.write('======> imaginary part difference on each rule: ' + ', '.join(map(str, img_difference)) + '\n')
# img_file.write('======> imaginary part difference on all rules: ' + str(np.sum(img_difference)) + '\n')
# real_file.write('======> real part difference on each rule: ' + ', '.join(map(str, rel_difference)) + '\n')
# real_file.write('======> real part difference on all rules: ' + str(np.sum(rel_difference)) + '\n')
# real_file.close()
# img_file.close()

# model_path = '/home/ComplEx-Inject/saved_models/' + args.model + '_' + args.dataset + '_mu_' + str(model.mu) + '_' + time_stamp + '.pkl'
results = dataset.eval(model, 'test', -1)
# torch.save((model,dataset), model_path)
print("\n\nTEST : ", results)