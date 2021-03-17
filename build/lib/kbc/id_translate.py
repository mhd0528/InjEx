####### translate between original relation name (id/string) to new (kbc) ids

import pkg_resources
import os
import errno
from pathlib import Path
import pickle

path = './src_data/'
# generate string (if exists) to original id map
rel_name_path = path + 'WN/original/'
rel_origin_id_path = path + 'WN/'

files = ['train', 'valid', 'test']
relations_names = list()
for f in files:
    file_path = os.path.join(rel_name_path, f)
    to_read = open(file_path, 'r')
    for line in to_read.readlines():
        lhs, rel, rhs = line.strip().split('\t')
        relations_names.append(rel)
        # print(rel)
    to_read.close()

print (len(set(relations_names)))
print(relations_names[0])

relations_original_ids = list()
for f in files:
    file_path = os.path.join(rel_origin_id_path, f)
    to_read = open(file_path, 'r')
    for line in to_read.readlines():
        lhs, rel, rhs = line.strip().split('\t')
        relations_original_ids.append(rel)
        # print(rel)
    to_read.close()

print (len(set(relations_original_ids)))
print (relations_original_ids[0])

# build name to original id map
name2original_ids = dict()
name2original_ids_f = open(rel_origin_id_path+'rel_name_2_original_id', mode='w+', encoding='utf-8')
for (name, original_id) in zip(relations_names, relations_original_ids):
    # print (name, original_id)
    if name not in name2original_ids:
        name2original_ids[name] = original_id
        name2original_ids_f.write(name + '\t' + original_id + '\n')
    else:
        if original_id != name2original_ids[name]:
            print("not the same id for relation:" + name + ' ' + name2original_ids[name] + ' ' + original_id)
print(len(name2original_ids))

# read in original_id to kbc_id dict
origin_2_kbc_path = rel_origin_id_path + 'relation_origin_2_kbc_id.txt'
origin_2_kbc_dict = dict()
with open(origin_2_kbc_path, 'r') as f:
    while True:
        line = f.readline()
        if line:
            # split by \t, locate original id, kbc_id
            origin_id = line.split('\t')[0]
            kbc_id = line.split('\t')[1].replace('\n', '')
            if origin_id not in origin_2_kbc_dict:
                origin_2_kbc_dict[origin_id] = kbc_id
            else:
                print('duplicate id!!!!')
        else:
            break
print(len(origin_2_kbc_dict))

# build name to kbc id map and write out to file
name2kbc_ids = dict()
name2kbc_ids_f = open(rel_origin_id_path+'rel_name_2_kbc_id', mode='w+', encoding='utf-8')
for name in name2original_ids:
    if name not in name2kbc_ids:
        name2kbc_ids_f.write(name + '\t' + origin_2_kbc_dict[name2original_ids[name]] + '\n')
        name2kbc_ids[name] = origin_2_kbc_dict[name2original_ids[name]]
    else:
        print('duplicate relations...')
print(len(name2kbc_ids))
name2kbc_ids_f.close()

# translate entailment rule with confidence to kbc id
origin_conf_f = rel_origin_id_path + '_cons.txt'
kbc_id_conf_f = open(rel_origin_id_path + 'kbc_id_cons.txt', mode='w+', encoding='utf-8')
with open(origin_conf_f, 'r') as f:
    while True:
        line = f.readline()
        if line:
                # two relations split by ',', confidence split by tab
                # losing information about '-' now
                rel0 = line.split(',')[0].replace('-', '')
                rel1 = line.split(',')[1].split('\t')[0].replace('-', '')
                conf = line.split('\t')[1]
                # print(rel0, rel1, conf)
                rel_id0 = name2kbc_ids[rel0]
                rel_id1 = name2kbc_ids[rel1]
                kbc_id_conf_f.write(rel_id0 + ',' + rel_id1 + '\t' + conf)
        else:
            break
kbc_id_conf_f.close()