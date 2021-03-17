import os
import errno
from pathlib import Path
import numpy as np
import pickle
import torch

kbc_id_conf_f = './src_data/FB15K/kbc_id_cons.txt'
r_p_list = []
r_q_list = []
conf_list = []
with open(kbc_id_conf_f, 'r') as f:
    while True:
        line = f.readline()
        if line:
                # two relations split by ',', confidence split by tab
                line.replace('\n', '')
                r_p = line.split(',')[0]
                r_q = line.split(',')[1].split('\t')[0]
                conf = line.split('\t')[1]
                # print(rel0, rel1, conf)
                r_p_list.append(int(r_p))
                r_q_list.append(int(r_q))
                conf_list.append(float(conf))
        else:
            break

# rule_set = np.vstack((r_p_list, r_q_list, conf_list))
rule_list = [r_p_list, r_q_list, conf_list]
print (len(rule_list[0]), len(rule_list[1]), len(rule_list[2]))

# get embeddings for all r_p and r_q
with open('../model_data.pkl', 'rb') as handle:
    m = pickle.load(handle)

rel = m.embeddings[1]#(x[:, 1])
idx_p = torch.LongTensor(rule_list[0]).cuda()
idx_q = torch.LongTensor(rule_list[1]).cuda()
print (rel(idx_p).type)
print (len(rel(idx_p[0])))
r_p_ebds = rel(idx_p)
r_p_ebds = r_p_ebds[:, :500], r_p_ebds[:, 500:]
r_q_ebds = rel(idx_q)
r_q_ebds = r_q_ebds[:, :500], r_q_ebds[:, 500:]

# # compute score of rules
score = 0
for i in range(len(r_p_ebds[0])):
    score += torch.sum(torch.max(torch.zeros(500).cuda(), r_p_ebds[0][i] - r_q_ebds[0][i])) * rule_list[2][i]
    score += torch.sum(torch.square(r_p_ebds[1][i] - r_q_ebds[1][i])) * rule_list[2][i]

# score = factor[0] * score
print (score)