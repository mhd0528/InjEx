# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import tqdm
import torch
from torch import nn
from torch import optim
import numpy as np

import models
from models import KBCModel
from regularizers import Regularizer


class KBCOptimizer(object):
    def __init__(
            self, model: KBCModel, regularizer: Regularizer, optimizer: optim.Optimizer, batch_size: int = 256, n_train_batches = 10, 
            verbose: bool = True
    ):
        self.model = model
        # if isinstance(model, models.ComplEx_NNE):
        #     print ("type matched!!!")

        self.regularizer = regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        # pi is only used for logicNN model
        self.pi = 0.5
        self.n_train_batches = n_train_batches
        self.m_NumNeg = 10

    #### get new pi for each epoch base on the epoch number
    def get_pi(self, cur_iter, params=None):
        """ exponential decay: pi_t = max{1 - k^t, lb} """
        k, lb = params[0], params[1]
        # pi = 1. - max([k ** cur_iter, lb])
        pi = max([k ** cur_iter, lb])
        # print("pi for eopch:", (cur_iter, pi))
        return pi

    def epoch(self, examples: torch.LongTensor, rule_type=0):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean')
        loss2 = nn.BCEWithLogitsLoss()
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            b_cnt = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin:b_begin + self.batch_size
                ].cuda()

                predictions, factors = self.model.forward(input_batch)
                truth = input_batch[:, 2]

                l_fit = loss(predictions, truth)
                l_reg = self.regularizer.forward(factors)
                l = l_fit + l_reg

                if isinstance(self.model, models.ComplEx_NNE):
                    # print ("add rule injection term to loss function")
                    # l_rule_constraint = self.model.get_rules_score()
                    l_rule_constraint = self.model.get_rules_loss()
                    # print(l)
                    # print(l_rule_constraint)
                    # l += l_rule_constraint.squeeze()
                    l += l_rule_constraint
                
                # loss term for ComplEx_logicNN with entailment rules
                if isinstance(self.model, models.ComplEx_logicNN):
                    # self.pi = self.get_pi(cur_iter=(b_cnt * 1. / self.n_train_batches), params=self.model.pi_params)
                    # print("=======> optimizor pi: " + str(self.pi) + " at batch " + str(b_cnt))
                    # for each rule, compute the penalty
                    if rule_type == 0:
                        ## directly use rules as feature, compute gradient on relations
                        ## print("=======> adding rule feature loss to total loss")
                        # l_rule_constraint = self.model.get_rules_loss()
                        # l = (1-self.pi) * l + self.pi * l_rule_constraint

                        # Grounding injection: use r_q triples as rule feature
                        # print("=======> computing rule grounding loss")
                        rule_predictions, rule_factors = self.model.forward(self.model.rule_feas)
                        rule_truth = self.model.rule_feas[:, 2]

                        l_rule_fit = loss(rule_predictions, rule_truth)
                        l_rule_reg = self.regularizer.forward(rule_factors)
                        l = (1-self.pi) * l + self.pi * (l_rule_fit + l_rule_reg)
                    elif rule_type == 4:
                        ## Grounding injection: use created r_p triples as rule feature
                        rule_predictions, rule_factors = self.model.forward(self.model.rule_feas)
                        rule_truth = self.model.rule_feas[:, 2]

                        l_rule_fit = loss(rule_predictions, rule_truth)
                        l_rule_reg = self.regularizer.forward(rule_factors)
                        # l = (1-self.pi) * l + self.pi * (l_rule_fit + l_rule_reg)

                if isinstance(self.model, models.ComplEx_supportNN):
                    # print ("add rule injection term for supportNN")
                    l_rule_constraint = self.model.get_rules_loss()
                    print(l)
                    print(l_rule_constraint)
                    l += l_rule_constraint

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                # constraint on entity embeddings, should be used for ComplEx_NNE and ComplEx_logicNN with entailment rules
                # round only entity embeddings, only < 0 or > 1, others stay the same
                if rule_type == 0:
                    if isinstance(self.model, models.ComplEx_NNE) or isinstance(self.model, models.ComplEx_logicNN):
                        # print("clamping entity embeddeing constraint")
                        self.model.embeddings[0].weight.data = self.model.embeddings[0].weight.data.clamp(1e-3, 1)
                        # with torch.no_grad():
                        #     for param in self.model.parameters():
                        #         if ((param.shape[0] == self.model.embeddings[0].num_embeddings)):
                        #             param.clamp_(1e-3, 1)
                
                b_begin += self.batch_size
                b_cnt += 1
                bar.update(input_batch.shape[0])
                if isinstance(self.model, models.ComplEx_NNE):
                    bar.set_postfix(loss=f'{l.item():.3f}', mu=self.model.mu)
                else:
                    bar.set_postfix(loss=f'{l.item():.3f}')

    # used for ComplEx_origin & injection, java version
    def epoch_2(self, examples: torch.LongTensor, dataset):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean')
        # print("epoch_2 function")
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            b_cnt = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin:b_begin + self.batch_size
                ].cuda()

                # if isinstance(self.model, models.ComplEx_logicNN):
                #     self.pi = self.get_pi(cur_iter=(b_cnt * 1. / self.n_train_batches), params=self.model.pi_params)
                #     # print("   optimizor pi: " + str(self.pi) + " at batch " + str(b_cnt))
                #     b_cnt += 1

                predictions, factors = self.model.forward(input_batch)
                truth = input_batch[:, 2]

                l_fit = loss(predictions, truth)
                # print(predictions.size())

                l_reg = self.regularizer.forward(factors)
                l = l_fit + l_reg 
                if isinstance(self.model, models.ComplEx_NNE):
                    # create negative samples
                    neg_head_set = set()
                    neg_tail_set = set()
                    for triple in input_batch:
                        tmp_head_set = dataset.generateHeadNegTriple(self.m_NumNeg // 2, triple)
                        neg_head_set = tmp_head_set.union(neg_head_set)
                        neg_tail_set = neg_tail_set.union(dataset.generateTailNegTriple(self.m_NumNeg // 2, triple))
                    # print(neg_tail_set)

                    neg_head_batch = torch.tensor(list(neg_head_set)).cuda()
                    neg_tail_batch = torch.tensor(list(neg_tail_set)).cuda()
                    # ent_list = list(set(input_batch[:, 0].tolist.append(input_batch[:, 2].tolist).append(neg_head_batch[:, 0].tolist).append(neg_head_batch[:, 2].tolist).append(neg_tail_batch[:, 0].tolist).append(neg_tail_batch[:, 2].tolist)))
                    ent_batch = torch.cat((input_batch[:, 0], input_batch[:, 2], neg_head_batch[:, 0], neg_head_batch[:, 2], neg_tail_batch[:, 0], neg_tail_batch[:, 2]), -1)
                    ent_list = list(set(ent_batch.tolist()))
                    # loss on negative samples
                    neg_weight = torch.neg(torch.ones(predictions.size()[1])).cuda()
                    loss_n = nn.CrossEntropyLoss(weight=neg_weight, reduction='mean')
                    neg_head_predictions, neg_head_factors = self.model.forward(neg_head_batch)
                    # print(neg_head_predictions.size())
                    l_head_fit = loss_n(neg_head_predictions, neg_head_batch[:, 2])
                    neg_tail_predictions, neg_tail_factors = self.model.forward(neg_tail_batch)
                    l_tail_fit = loss_n(neg_tail_predictions, neg_tail_batch[:, 2])
                    # print("adding head and tail loss")
                    mu_factor = (self.model.mu / torch.ceil( torch.Tensor([examples.shape[0]/ 1000]) )).cuda()
                    l = l + (l_head_fit + l_tail_fit) * mu_factor
                #print("type of l: " + str(type(l)))
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                # constraint on entity embeddings, should be used for ComplEx_NNE and ComplEx_logicNN
                # round only entity embeddings, only < 0 or > 1, others stay the same
                # if isinstance(self.model, models.ComplEx):
                #     self.model.embeddings[0].weight.data = self.model.embeddings[0].weight.data.clamp_(1e-3, 1)
                # if isinstance(self.model, models.ComplEx_NNE) or isinstance(self.model, models.ComplEx_logicNN):
                #     self.model.embeddings[0].weight.data[ent_list] = self.model.embeddings[0].weight.data[ent_list].clamp_(1e-3, 1)
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.0f}')
