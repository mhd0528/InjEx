import torch
import os

from typing import List

from config import args
from triplet import EntityDict
from collections import defaultdict
from doc import Example

def get_entity_dict(samples):
    entity_dict = defaultdict(lambda : defaultdict(list))
    for h, r, t in samples:
        entity_dict[h][r].append(t)
    return entity_dict


def rerank_by_rule(batch_score: torch.tensor,
                    examples, entity_dict, rules):

    bonus_weight = 1e-6

    for idx in range(batch_score.size(0)):
        cur_ex = examples[idx]
        n_hop_indices = []

        #### for each head entity in eval data
        #### for each tail entity, check if there is any rules followed
        
        get_link_graph().get_n_hop_entity_indices(cur_ex.head_id,
                                                                  entity_dict=entity_dict,
                                                                  n_hop=args.rerank_n_hop)
        delta = torch.tensor([bonus_weight for _ in n_hop_indices]).to(batch_score.device)
        n_hop_indices = torch.LongTensor(list(n_hop_indices)).to(batch_score.device)

        ## add bonus to tails that follow the rules
        batch_score[idx].index_add_(0, n_hop_indices, delta)