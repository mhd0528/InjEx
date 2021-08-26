import os
import errno
from pathlib import Path
import numpy as np

kbc_id_conf_f = './src_data/WN/kbc_id_cons.txt'
r_p_list = []
r_q_list = []
conf_list = []
with open(kbc_id_conf_f, 'r') as f:
    while True:
        line = f.readline()
        if line:
                # two relations split by ',', confidence split by tab
                r_p = line.split(',')[0]
                r_q = line.split(',')[1].split('\t')[0]
                conf = line.split('\t')[1]
                # print(rel0, rel1, conf)
                r_p_list.append(r_p)
                r_q_list.append(r_q)
                conf_list.append(conf)
        else:
            break

rule_set = np.vstack((r_p_list, r_q_list, conf_list))
print (len(rule_set))