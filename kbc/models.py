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
import numpy as np


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
            mu: float = 0.1
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
        self.mu = mu
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
    
    def get_rules_loss(self):
        # get embeddings for all r_p and r_q
        rel = self.embeddings[1]
        for i, rule in enumerate(self.rule_list):
            r_p, r_q, conf, r_dir = rule
            r_p = torch.LongTensor([r_p]).cuda()
            r_q = torch.LongTensor([r_q]).cuda()
            r_p_ebds = torch.transpose(rel(r_p), 0, 1)
            r_q_ebds = torch.transpose(rel(r_q), 0, 1)
            r_p_re, r_p_im = r_p_ebds[:self.rank], r_p_ebds[self.rank:]
            r_q_re, r_q_im = r_q_ebds[:self.rank], r_q_ebds[self.rank:]
            # print(r_p_ebds.size(), r_p_re.size(), self.rank)
            
            r_p_re *= conf
            r_p_im *= r_dir
            r_q_re *= conf
            # print("rule grad exists?: " + str(r_q_im.requires_grad))
            # real penalty
            if not i:
                rule_score = torch.sum(torch.max(torch.zeros(self.rank).cuda(), (r_p_re - r_q_re))) 
            else:
                rule_score += torch.sum(torch.max(torch.zeros(self.rank).cuda(), (r_p_re - r_q_re))) 
            # imaginary penalty
            rule_score += torch.sum(torch.square(r_p_im - r_q_im) * conf).cuda() 

        rule_score /= len(self.rule_list)
        rule_score *= self.mu
        # print(rule_score.requires_grad)
        return rule_score
    

class ComplEx_logicNN(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            rule_list: list, 
            init_size: float = 1e-3,
            C: float = 6,
            rule_feas: list = [],
            pi_params: list = [0.95, 0]
    ):
        super(ComplEx_logicNN, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.rule_list = rule_list
        self.C = C
        self.rule_feas = rule_feas
        self.pi_params = pi_params

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        # check_nan = torch.sum(torch.isnan(self.embeddings[0].weight.data))
        # print ("have nan value in weight: " + str(check_nan))
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
        if len(x[0]) == 4:
            r_dirs = torch.zeros(len(x), len(x)).cuda()
            for i, r_dir in enumerate(x[:, 3]):
                r_dirs[i][i] = r_dir
            # tmp = torch.matmul(r_dirs, rel[:, self.rank:])
            tmp = r_dirs @ rel[:, self.rank:]
            # print(rel[:, self.rank:].size(), r_dirs.size())
        # else:
        #     r_dirs = torch.eye(len(x)).cuda()
        # print(r_dirs)

        # print(rel[:, self.rank:])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        if len(x[0]) == 4:
            rel = rel[:, :self.rank], tmp
        else:
            rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        # print(rel[1])
        
        check_nan = torch.sum(torch.isnan(rel[0])) + torch.sum(torch.isnan(rel[1]))
        if check_nan > 0 :
            #print(torch.isnan(rel[0]))
            print("number of relations: " + str(len(rel[0])))
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

    def get_rules_loss(self):
        # get embeddings for all r_p and r_q
        rel = self.embeddings[1]
        for i, rule in enumerate(self.rule_list):
            r_p, r_q, conf, r_dir = rule
            r_p = torch.LongTensor([r_p]).cuda()
            r_q = torch.LongTensor([r_q]).cuda()
            r_p_ebds = torch.transpose(rel(r_p), 0, 1)
            r_q_ebds = torch.transpose(rel(r_q), 0, 1)
            r_p_re, r_p_im = r_p_ebds[:self.rank], r_p_ebds[self.rank:]
            r_q_re, r_q_im = r_q_ebds[:self.rank], r_q_ebds[self.rank:]
            # print(r_p_ebds.size(), r_p_re.size(), self.rank)
            
            r_p_re *= conf
            r_p_im *= r_dir
            r_q_re *= conf
            # print("rule grad exists?: " + str(r_q_im.requires_grad))
            # real penalty
            if not i:
                rule_score = torch.sum(torch.max(torch.zeros(self.rank).cuda(), (r_p_re - r_q_re))) 
            else:
                rule_score += torch.sum(torch.max(torch.zeros(self.rank).cuda(), (r_p_re - r_q_re))) 
            # imaginary penalty
            rule_score += torch.sum(torch.square(r_p_im - r_q_im) * conf).cuda() 

        rule_score /= len(self.rule_list)
        rule_score
        # print(rule_score.requires_grad)
        return rule_score

    # apply teacher "network" to generate penalty
    # currently, a grounding of a rule is just ebd of ra and rb
    # constraints follow Ding's paper
    def get_rules_pred(self, batch_fea, rule_conf, rule_inv):
        ent = self.embeddings[0]
        rel = self.embeddings[1]

        rule_pred = torch.zeros(len(batch_fea)).cuda()
        # for each triple pair, compute and compare their score
        for i, ((e1, head, e2), (e1, tail, e2)) in enumerate(batch_fea):
            e1 = torch.LongTensor([e1]).cuda()
            head = torch.LongTensor([head]).cuda()
            tail = torch.LongTensor([tail]).cuda()
            e2 = torch.LongTensor([e2]).cuda()
            
            triple_lhs = ent(e1)
            rule_head = rel(head)
            rule_tail = rel(tail)
            triple_rhs = ent(e2)

            triple_lhs = triple_lhs[:, :self.rank], triple_lhs[:, self.rank:]
            rule_head = rule_head[:, :self.rank], rule_head[:, self.rank:]
            rule_tail = rule_tail[:, :self.rank], rule_tail[:, self.rank:]
            triple_rhs = triple_rhs[:, :self.rank], triple_rhs[:, self.rank:]

            score_h = torch.sum(
                    (triple_lhs[0] * rule_head[0] - triple_lhs[1] * rule_head[1]) * triple_rhs[0] +
                    (triple_lhs[0] * rule_head[1] + triple_lhs[1] * rule_head[0]) * triple_rhs[1],
                    1, keepdim=True
                )
            if not rule_inv:
                score_t =torch.sum(
                        (triple_lhs[0] * rule_tail[0] - triple_lhs[1] * rule_tail[1]) * triple_rhs[0] +
                        (triple_lhs[0] * rule_tail[1] + triple_lhs[1] * rule_tail[0]) * triple_rhs[1],
                        1, keepdim=True
                    )
            else:
                score_t = torch.sum(
                        (triple_rhs[0] * rule_tail[0] - triple_rhs[1] * rule_tail[1]) * triple_lhs[0] +
                        (triple_rhs[0] * rule_tail[1] + triple_rhs[1] * rule_tail[0]) * triple_lhs[1],
                        1, keepdim=True
                    )
            # if the rule holds, score_t > score_h should hold
            # if score_t > score_h:
            rule_pred[i] = score_t - score_h
        # rule_pred = torch.tensor(rule_pred).cuda()
        # rule_pred = rule_pred / torch.sum(rule_pred)
        return rule_pred

    ######## extract/create groundings for different rules
    def fea_generator(self, rule_type, train_data, ent_num):
        if rule_type == 0:
            # create new triples(features) based on related training triples
            # store idx
            entailment_triples = []
            for r_p, r_q, conf, r_dir in self.rule_list:
                for i, (e1, r, e2) in enumerate(train_data):
                    if r == r_p:
                        entailment_triples.append([e1, r_q, e2, r_dir])
            # print(len(entailment_triples))
            # entailment_triples = set(map(tuple, entailment_triples))
            # print(len(entailment_triples))
            # entailment_triples = list(map(list, entailment_triples))
            entailment_triples = np.array(list(entailment_triples))
            self.rule_feas = torch.from_numpy(entailment_triples.astype('int64')).cuda()
            # exit()
        elif rule_type == 4:
            # rule: p(x, y) <- q(x, z), r(z, y)
            # format: p, q, r, conf, tuples (e1, q, e2)
            # for all entities, score (e2, r, e3) and filter
            # then maximize score of (e1, p, e3)
            # select top 0.5% score tups as valid tuples
            valid_num = ent_num // 2000
            total_tup_num = valid_num * len(self.rule_list)
            all_valid_tups = []
            for i, fea in enumerate(self.rule_list):
                r_p = fea[0][0]
                r_q = fea[0][1]
                r_r = fea[0][2]
                conf = fea[1]
                tuples = fea[2]
                for tup in tuples:
                    valid_tups = []
                    # for all entities, score (e2, r, e3) and filter
                    # for e in range(ent_num):
                    #     tmp_tup = [tup[2], r_r, e]
                    #     tmp_tup_torch = torch.from_numpy(np.array([tmp_tup])).cuda()
                    #     # if self.score(tmp_tup) > 0.95:
                    #     valid_tups.append((self.score(tmp_tup_torch), [tup[0].item(), r_p, e]))
                    for e in range(ent_num):
                        valid_tups.append([tup[2], r_r, e])
                    valid_tups = torch.from_numpy(np.array(valid_tups).astype('int64')).cuda()
                    score_tups = torch.transpose(self.score(valid_tups), 0, 1)
                    # select top 1% score tups as valid tuples
                    valid_idxs = torch.topk(score_tups, k=valid_num)[1].long().flatten()
                    valid_tups = torch.index_select(valid_tups, 0, valid_idxs)
                    all_valid_tups.append(valid_tups)
            all_valid_tups = torch.cat(all_valid_tups, dim=0)
            # print("======> checking format of created tuples: " + str(all_valid_tups[0]))
            # if len(all_valid_tups):
            #     all_valid_tups = np.array(all_valid_tups)
            #     # copy = np.copy(all_valid_tups)
            #     # # create reciprocal triples
            #     # tmp = np.copy(copy[:, 0])
            #     # copy[:, 0] = copy[:, 2]
            #     # copy[:, 2] = tmp
            #     # copy[:, 1] += train_data.shape[1] // 2  # has been multiplied by two.
            #     # all_valid_tups = np.vstack((all_valid_tups, copy))
            #     # print("Add reciprocal triples for new extracted data: " )
            #     # print(all_valid_tups)
            # else:
            #     print("======> empty valid tuples created")
            #     all_valid_tups = np.array(all_valid_tups)

            # all_valid_tups = torch.from_numpy(all_valid_tups.astype('int64')).cuda()
            self.rule_feas = all_valid_tups