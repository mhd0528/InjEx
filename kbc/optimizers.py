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
        self.pi = 0
        self.n_train_batches = n_train_batches
        self.m_NumNeg = 6

    def get_pi(self, cur_iter, params=None, pi=None):
        """ exponential decay: pi_t = max{1 - k^t, lb} """
        k, lb = params[0], params[1]
        pi = 1. - max([k ** cur_iter, lb])
        return pi

    def epoch(self, examples: torch.LongTensor):
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
                
                if isinstance(self.model, models.ComplEx_logicNN):
                    self.pi = self.get_pi(cur_iter=(b_cnt * 1. / self.n_train_batches), params=self.model.pi_params)
                    # print("   optimizor pi: " + str(self.pi) + " at batch " + str(b_cnt))
                    b_cnt += 1

                predictions, factors = self.model.forward(input_batch)
                truth = input_batch[:, 2]

                l_fit = loss(predictions, truth)
                
                #print(torch.sum(torch.isnan(predictions)))

                #print("type of l_fit: " + str(type(l_fit)))

                l_reg = self.regularizer.forward(factors)
                l = l_fit + l_reg
                #print("type of l: " + str(type(l)))
                

                if isinstance(self.model, models.ComplEx_NNE):
                    mu_factor = (self.model.mu / torch.ceil( torch.Tensor([examples.shape[0]/ 1000]) )).cuda()
                    #print ("add rule injection term to loss function")
                    l_rule_constraint = self.model.get_rules_score()
                    # print(type(l_rule_constraint))
                    # print(type(mu_factor))
                    # raise
                    l_rule_constraint = mu_factor * l_rule_constraint
                    # print("======> l_new: " + str(l_rule_constraint))
                    # with open("237_conjugate_loss_values_mu_10.txt", "a") as myfile:
                    #     myfile.write(f"({l},{l_rule_constraint.squeeze()}) \n")
                    l += l_rule_constraint.squeeze()
                
                if isinstance(self.model, models.ComplEx_logicNN):
                    # for each rule, compute the penalty
                    for i, fea in enumerate(self.model.rule_feas):
                        # print("======> applying rule " + str(i) + "======>")
                        # select related triples in current batch
                        batch_fea = []
                        for ((e1, head, e2), (e1, tail, e2)) in fea:
                            if torch.tensor([e1, head, e2]).cuda() in input_batch:
                                batch_fea.append(((e1, head, e2), (e1, tail, e2)))
                        # inverse rules treat differently in model function
                        rule_conf = self.model.rule_list[2][i]
                        rule_inv = self.model.rule_list[3][i]
                        rule_pred = torch.tensor(self.model.get_rules_pred(batch_fea, rule_conf, rule_inv)).cuda()
                        rule_truth = torch.ones(len(batch_fea)).cuda() * rule_conf
                        rule_pen = loss2(rule_pred, rule_truth) * self.model.C * rule_conf
                        if torch.isnan(rule_pen):
                            # print("penalty term is nan, skip......")
                            self.pi = 1
                        else:
                            # print("======> rule_penalty: " + str(rule_pen))
                            l = (1-self.pi) * l + self.pi * rule_pen

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                # constraint on entity embeddings, should be used for ComplEx_NNE and ComplEx_logicNN
                # round only entity embeddings, only < 0 or > 1, others stay the same
                if isinstance(self.model, models.ComplEx_NNE) or isinstance(self.model, models.ComplEx_logicNN):
                    with torch.no_grad():
                        for param in self.model.parameters():
                            if ((param.shape[0] == self.model.embeddings[0].num_embeddings)):
                                # print(param.shape)
                                # raise
                                # temp = param.detach().clone()
                                # embedding.weight.data are embeddings???????
                                param.clamp_(1e-3, 1)
                        
                        # all_idx = [i for i in range(self.model.embeddings[0].num_embeddings)]
                        # all_idx = torch.LongTensor(self.all_idx).cuda()
                        # tmp = self.model.embeddings[0](all_idx)
                
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.0f}')

    def epoch_2(self, examples: torch.LongTensor, dataset):
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
                # print(len(ent_list))

                
                # if isinstance(self.model, models.ComplEx_logicNN):
                #     self.pi = self.get_pi(cur_iter=(b_cnt * 1. / self.n_train_batches), params=self.model.pi_params)
                #     # print("   optimizor pi: " + str(self.pi) + " at batch " + str(b_cnt))
                #     b_cnt += 1

                predictions, factors = self.model.forward(input_batch)
                truth = input_batch[:, 2]

                l_fit = loss(predictions, truth)
                # print(predictions.size())

                # loss on negative samples
                neg_weight = torch.neg(torch.ones(predictions.size()[1])).cuda()
                loss_n = nn.CrossEntropyLoss(weight=neg_weight, reduction='mean')
                neg_head_predictions, neg_head_factors = self.model.forward(neg_head_batch)
                # print(neg_head_predictions.size())
                l_head_fit = loss_n(neg_head_predictions, neg_head_batch[:, 2])
                neg_tail_predictions, neg_tail_factors = self.model.forward(neg_tail_batch)
                l_tail_fit = loss_n(neg_tail_predictions, neg_tail_batch[:, 2])

                l_reg = self.regularizer.forward(factors)
                l = l_fit + l_reg + l_head_fit + l_tail_fit
                #print("type of l: " + str(type(l)))
                

                # if isinstance(self.model, models.ComplEx_NNE):
                #     mu_factor = (self.model.mu / torch.ceil( torch.Tensor([examples.shape[0]/ 1000]) )).cuda()
                #     #print ("add rule injection term to loss function")
                #     l_rule_constraint = self.model.get_rules_score()
                #     # print(type(l_rule_constraint))
                #     # print(type(mu_factor))
                #     # raise
                #     l_rule_constraint = mu_factor * l_rule_constraint
                #     l += l_rule_constraint.squeeze()
                
                # if isinstance(self.model, models.ComplEx_logicNN):
                #     # for each rule, compute the penalty
                #     for i, fea in enumerate(self.model.rule_feas):
                #         # print("======> applying rule " + str(i) + "======>")
                #         # select related triples in current batch
                #         batch_fea = []
                #         for ((e1, head, e2), (e1, tail, e2)) in fea:
                #             if torch.tensor([e1, head, e2]).cuda() in input_batch:
                #                 batch_fea.append(((e1, head, e2), (e1, tail, e2)))
                #         # inverse rules treat differently in model function
                #         rule_conf = self.model.rule_list[2][i]
                #         rule_inv = self.model.rule_list[3][i]
                #         rule_pred = torch.tensor(self.model.get_rules_pred(batch_fea, rule_conf, rule_inv)).cuda()
                #         rule_truth = torch.ones(len(batch_fea)).cuda() * rule_conf
                #         rule_pen = loss2(rule_pred, rule_truth) * self.model.C * rule_conf
                #         if torch.isnan(rule_pen):
                #             # print("penalty term is nan, skip......")
                #             self.pi = 1
                #         else:
                #             # print("======> rule_penalty: " + str(rule_pen))
                #             l = (1-self.pi) * l + self.pi * rule_pen


                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                # constraint on entity embeddings, should be used for ComplEx_NNE and ComplEx_logicNN
                # round only entity embeddings, only < 0 or > 1, others stay the same
                if isinstance(self.model, models.ComplEx_NNE) or isinstance(self.model, models.ComplEx_logicNN):
                    self.model.embeddings[0].weight.data[ent_list] = self.model.embeddings[0].weight.data[ent_list].clamp_(1e-3, 1)
                    # with torch.no_grad():
                    #     for param in self.model.parameters():
                    #         if ((param.shape[0] == self.model.embeddings[0].num_embeddings)):
                    #             # print(param.shape)
                    #             # raise
                    #             # temp = param.detach().clone()
                    #             # embedding.weight.data are embeddings???????
                    #             # param.clamp_(1e-3, 1)
                    #             param[ent_list].clamp_(0, 1)
                        
                    #     # all_idx = [i for i in range(self.model.embeddings[0].num_embeddings)]
                    #     # all_idx = torch.LongTensor(self.all_idx).cuda()
                    #     # tmp = self.model.embeddings[0](all_idx)
                
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.0f}')
