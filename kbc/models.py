# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import torch
from torch import nn
from collections import defaultdict
import numpy as np
import math


class KBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    q = self.get_queries(these_queries)

                    scores = q @ rhs
                    targets = self.score(these_queries)

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                            # print(query)
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks


class CP(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(CP, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.lhs = nn.Embedding(sizes[0], rank, sparse=True)
        self.rel = nn.Embedding(sizes[1], rank, sparse=True)
        self.rhs = nn.Embedding(sizes[2], rank, sparse=True)

        self.lhs.weight.data *= init_size
        self.rel.weight.data *= init_size
        self.rhs.weight.data *= init_size

    def score(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        return torch.sum(lhs * rel * rhs, 1, keepdim=True)

    def forward(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])
        return (lhs * rel) @ self.rhs.weight.t(), (lhs, rel, rhs)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        return self.lhs(queries[:, 0]).data * self.rel(queries[:, 1]).data


class ComplEx(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        #original loss function
        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        # ebd of left hand side, relation and right hand side of triples
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        return (
            (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
        ), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)

class ComplEx_NNE(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            rule_list: list, 
            init_size: float = 1e-3, 
            mu: float = 0.1,
            rule_type: int = 0
    ):
        super(ComplEx_NNE, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.rule_list = rule_list

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        # check_nan = torch.sum(torch.isnan(self.embeddings[0].weight.data))
        # print ("have nan value in weight: " + str(check_nan))
        self.embeddings[0].weight.data *= init_size
        #self.embeddings[0].weight.data += torch.abs(torch.min(self.embeddings[0].weight.data)) + 1e-3
        self.embeddings[1].weight.data *= init_size
        #self.embeddings[1].weight.data += torch.abs(torch.min(self.embeddings[1].weight.data))

        #### limit relation embedding range
        self.rel_embedding_range = 1
        # nn.init.uniform_(
        #     tensor = self.embeddings[1].weight.data, 
        #     a=0, 
        #     b=self.rel_embedding_range
        # )
        self.mu = mu
        self.rule_type = rule_type
        print("======> mu value: " + str(self.mu))

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        #original loss function
        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        # ebd of left hand side, relation and right hand side of triples
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        # print(x[:, 1])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        
        check_nan = torch.sum(torch.isnan(rel[0])) + torch.sum(torch.isnan(rel[1]))
        if check_nan > 0 :
            #print(torch.isnan(rel[0]))
            print("number of triples: " + str(len(rel[0])))
            print ("have nan value in forward embedding: " + str(check_nan))

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        return (
            (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
        ), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)

    def get_rules_score(self):
        # get embeddings for all r_p and r_q
        rel = self.embeddings[1]
        idx_p = torch.LongTensor(self.rule_list[0]).cuda()
        idx_q = torch.LongTensor(self.rule_list[1]).cuda()
        #print(torch.max(idx_p))

        r_p_ebds = rel(idx_p)
        r_p_ebds = r_p_ebds[:, :self.rank], r_p_ebds[:, self.rank:]
        r_q_ebds = rel(idx_q)
        r_q_ebds = r_q_ebds[:, :self.rank], r_q_ebds[:, self.rank:]

        check_nan = torch.sum(torch.isnan(r_p_ebds[0]))
        if check_nan > 0 :
            # print ("number of relations: " + str(len(r_p_ebds[0])))
            print ("have nan value in rule score embedding: " + str(check_nan))
            print(torch.isnan(r_p_ebds[0]))

        # compute score of rules
        score = 0
        # print("======> number of rules: " + str(len(r_p_ebds[0])))
        for i in range(len(r_p_ebds[0])):
            score += torch.sum(torch.max(torch.zeros(self.rank).cuda(), r_p_ebds[0][i] - r_q_ebds[0][i])) * self.rule_list[2][i]
            if self.rule_list[3][i] < 0:
                r_q_ebds[1][i] = -r_q_ebds[1][i]
                score += torch.sum(torch.square(r_p_ebds[1][i] + r_q_ebds[1][i])) * self.rule_list[2][i]
            else:
                score += torch.sum(torch.square(r_p_ebds[1][i] - r_q_ebds[1][i])) * self.rule_list[2][i]

        #score *= self.mu
        # score = factor[0] * score
        # print (score)
        return score * self.mu
    
    def get_rules_loss(self, rule_type, rule_list):
        if rule_type == 1:
            # get embeddings for all r_p and r_q
            rel = self.embeddings[1]
            rule_score = 0
            for i, rule in enumerate(rule_list):
                r_p, r_q, conf, r_dir = rule
                r_p = torch.LongTensor([r_p]).cuda()
                r_q = torch.LongTensor([r_q]).cuda()
                r_p_ebds = torch.transpose(rel(r_p), 0, 1)
                r_q_ebds = torch.transpose(rel(r_q), 0, 1)
                # r_p_ebds = rel(r_p)[0]
                # r_q_ebds = rel(r_q)[0]
                r_p_re, r_p_im = r_p_ebds[:self.rank], r_p_ebds[self.rank:]
                r_q_re, r_q_im = r_q_ebds[:self.rank], r_q_ebds[self.rank:]
                # print(r_p_ebds.size(), r_p_re.size(), self.rank)
                
                r_p_re *= conf
                r_p_im *= r_dir
                r_q_re *= conf
                # print("rule grad exists?: " + str(r_q_im.requires_grad))
                # real penalty
                # rule_score += self.mu * torch.sum(torch.max(torch.zeros(self.rank).cuda(), (r_p_re - r_q_re)))
                rule_score += self.mu * torch.sum(torch.max(torch.zeros(self.rank).cuda(), (r_q_re - r_p_re)))
                # imaginary penalty
                rule_score += self.mu * torch.sum(torch.square(r_q_im - r_p_im) * conf).cuda() 

            rule_score /= len(rule_list)
            # rule_score *= self.mu
            # print(rule_score.requires_grad)
            return rule_score

        if rule_type == 4:
            ## Re(tail) > 1/R * Re(head_i)
            ## Im(tail) == 1/R * Im(head_i)
            head_para = math.sqrt(self.rel_embedding_range * self.rank)
            ## get embeddings for all head and tail relations
            ## head relations ebd are combined using element-wise multiply
            rel = self.embeddings[1]
            rule_score = 0
            for i, rule in enumerate(rule_list):
                tail, head, conf = rule
                tail = torch.LongTensor([tail]).cuda()
                tail_ebd = torch.transpose(rel(tail), 0, 1)
                head_ebd = torch.ones(self.rank * 2).cuda()
                for r in head:
                    r = torch.LongTensor([r]).cuda()
                    head_ebd = head_ebd * rel(r)
                head_ebd = torch.transpose(head_ebd, 0, 1)# / head_para
                # r_p_ebds = rel(r_p)[0]
                # r_q_ebds = rel(r_q)[0]
                tail_re, tail_im = tail_ebd[:self.rank], tail_ebd[self.rank:]
                head_re, head_im = head_ebd[:self.rank], head_ebd[self.rank:]
                # print(r_p_ebds.size(), r_p_re.size(), self.rank)
                
                # print("rule grad exists?: " + str(r_q_im.requires_grad))
                # real penalty
                # rule_score += self.mu * conf * torch.sum(torch.square(head_re - tail_re)).cuda()
                rule_score += self.mu * conf * torch.sum(torch.max(torch.zeros(self.rank).cuda(), head_re - tail_re))
                # imaginary penalty
                # rule_score += self.mu * conf * torch.sum(torch.max(torch.zeros(self.rank).cuda(), tail_im - head_im))
                rule_score += self.mu * conf * torch.sum(torch.square(tail_im - head_im)).cuda()

            rule_score /= len(rule_list)
            # rule_score *= self.mu
            # print(rule_score.requires_grad)
            return rule_score