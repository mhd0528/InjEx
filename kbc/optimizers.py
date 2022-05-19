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

                if isinstance(self.model, models.InjEx):
                    # print ("add rule injection term to loss function")
                    # l_rule_constraint = self.model.get_rules_score()
                    if self.model.rule_type:
                        # print("injecting one type of rules")
                        l_rule_constraint = self.model.get_rules_loss(self.model.rule_type, self.model.rule_list)
                        # print(l)
                        # print(l_rule_constraint)
                        # l += l_rule_constraint.squeeze()
                    else:
                        ## compute loss for both types of rules
                        # print("injecting multiple types of rules")
                        l_rule_constraint = self.model.get_rules_loss(1, self.model.rule_list[0])
                        l_rule_constraint += self.model.get_rules_loss(4, self.model.rule_list[1])
                    l += l_rule_constraint

                
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                # constraint on entity embeddings, should be used for InjEx and ComplEx_logicNN with entailment rules
                # round only entity embeddings, only < 0 or > 1, others stay the same
                # if rule_type == 0:
                if isinstance(self.model, models.InjEx) or isinstance(self.model, models.ComplEx_logicNN):
                    # print("clamping entity embeddeing constraint")
                    self.model.embeddings[0].weight.data = self.model.embeddings[0].weight.data.clamp(1e-3, 1)
                    self.model.embeddings[1].weight.data = self.model.embeddings[1].weight.data.clamp(1e-3, 1)
                        # with torch.no_grad():
                        #     for param in self.model.parameters():
                        #         if ((param.shape[0] == self.model.embeddings[0].num_embeddings)):
                        #             param.clamp_(1e-3, 1)
                
                b_begin += self.batch_size
                b_cnt += 1
                bar.update(input_batch.shape[0])
                if isinstance(self.model, models.InjEx):
                    bar.set_postfix(loss=f'{l.item():.3f}', mu=self.model.mu)
                else:
                    bar.set_postfix(loss=f'{l.item():.3f}')