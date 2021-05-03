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
            self, model: KBCModel, regularizer: Regularizer, optimizer: optim.Optimizer, batch_size: int = 256,
            verbose: bool = True
    ):
        self.model = model
        # if isinstance(model, models.ComplEx_NNE):
        #     print ("type matched!!!")

        self.regularizer = regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose

    def epoch(self, examples: torch.LongTensor):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean')
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin:b_begin + self.batch_size
                ].cuda()
                
                #raise

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
                    with open("loss_values.txt", "a") as myfile:
                        myfile.write(f"({l},{l_rule_constraint.squeeze()}) \n")
                    l += l_rule_constraint.squeeze()

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()

                # constraint on entity embeddings, should be only used for ComplEx_NNE
                if isinstance(self.model, models.ComplEx_NNE):
                    with torch.no_grad():
                        for param in self.model.parameters():
                            if ((param.shape[0] == 14951)):
                                #print(param.shape)
                                #raise
                                #temp = param.detach().clone()
                                param.clamp_(1e-3, 1)
                                #raise
                
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.0f}')
            
            
            ########
            
            # 1) run with 1e-3 for 100 epochs
            # 2*) check if there's still some nan values in the embeddings, when *only* putting the above constraints
            # 3) run with rule injection and constraints