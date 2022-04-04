# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pkg_resources
import os
import errno
from pathlib import Path
import pickle
import pandas as pd

import numpy as np

from collections import defaultdict

# DATA_PATH = pkg_resources.resource_filename('kbc', 'data/')
# DATA_PATH = './data/'
DATA_PATH = Path('/blue/daisyw/ma.haodi/ComplEx-Inject/kbc/data/')lalala

def translate_cons(dataset, path, train_data, rule_type = 0):
    if rule_type == 0:
        rel2id = {}
        with open(str(DATA_PATH) + '/' + dataset+'/rel_id') as f:
            for i,line in enumerate(f):
                rel = line.split('\t')[0]
                rel_id = int(line.split('\t')[1])
                rel2id[rel] = rel_id
        # with open(path+'/_cons.txt') as f,open(path+'/cons.txt','w') as out, open(path+'/_237cons.txt', 'w') as out2:
        print(path)
        # with open(path+'/_cons_freq_train.txt') as f,open(path+'/cons.txt','w') as out:
        with open(path+'/_cons.txt') as f,open(path+'/cons.txt','w') as out:
            for line in f:
                rule_str, conf = line.strip().split()
                body,head = rule_str.split(',')
                prefix = ''
                if '-' in body:
                    prefix = '-'
                    body = body[1:]
                try:
                    rule = prefix + str(rel2id[body])+','+str(rel2id[head])
                    out.write('%s\t%s\n' % (rule,conf))
                    # out2.write(line)
                except KeyError:
                    print("rule not found: " + line)
    elif rule_type == 3:
        # read in each rule, translate, extract triples from all training triples
        # format: p, q, r, conf, triple_ids
        rel2id = {}
        with open(str(DATA_PATH) + '/' + dataset+'/rel_id') as f:
            for i,line in enumerate(f):
                rel = line.split('\t')[0]
                rel_id = int(line.split('\t')[1])
                rel2id[rel] = rel_id
        # read in rule set
        rule_df = pd.read_excel(path+'/Freebase_Rules.xlsx', sheet_name=None)['Type 3']
        # print(rule_df.head())
        with open(path+'/all_cons_3.txt','w') as out:
            for id, row in rule_df.iterrows():
                rel_p = '/' + row['p(x,y) <-'].replace('.', '/')
                rel_q = '/' + row['q(z,x)'].replace('.', '/')
                re_r = '/' + row['r(z,y)'].replace('.', '/')
                conf = row['Confidence']
                if conf >= 0.5:
                    try:
                        rule = str(rel2id[rel_p])+','+str(rel2id[rel_q])+','+str(rel2id[re_r])
                        # extract triples from training set
                        triple_ids = []
                        for i, triple in enumerate(train_data):
                            if triple[1] == rel2id[rel_q]:
                                triple_ids.append(str(i))
                        triple_ids_str = ' '.join(triple_ids)
                        out.write('%s\t%s\t%s\n' % (rule, conf, triple_ids_str))
                        print("rule found: " + str(rel_q))
                        # out2.write(line)
                    except KeyError:
                        continue
    # elif rule_type == 4:
    #     # read in each rule, translate, extract triples from all training triples
    #     # format: p, q, r, conf: triple_ids
    #     rel2id = {}
    #     with open(str(DATA_PATH) + '/' + dataset+'/rel_id') as f:
    #         for i,line in enumerate(f):
    #             rel = line.split('\t')[0]
    #             rel_id = int(line.split('\t')[1])
    #             rel2id[rel] = rel_id
    #     # read in rule set
    #     rule_df = pd.read_excel(path+'/Freebase_Rules.xlsx', sheet_name=None)['Type 4']
    #     # print(rule_df.head())
    #     with open(path+'/all_cons_4.txt','w') as out:
    #         for id, row in rule_df.iterrows():
    #             rel_p = '/' + row['p(x,y) <-'].replace('.', '/')
    #             rel_q = '/' + row['q(x,z)'].replace('.', '/')
    #             re_r = '/' + row['r(z,y)'].replace('.', '/')
    #             conf = row['Confidence']
    #             if conf >= 0.5:
    #                 try:
    #                     rule = str(rel2id[rel_p])+','+str(rel2id[rel_q])+','+str(rel2id[re_r])
    #                     # extract triples from training set
    #                     triple_ids = []
    #                     for i, triple in enumerate(train_data):
    #                         if triple[1] == rel2id[rel_q]:
    #                             triple_ids.append(str(i))
    #                     triple_ids_str = ' '.join(triple_ids)
    #                     out.write('%s\t%s\t%s\n' % (rule, conf, triple_ids_str))
    #                     print("rule found: " + str(rel_q))
    #                     # out2.write(line)
    #                 except KeyError:
    #                     continue
    #### type 4 rules from AnyBurl
    elif rule_type == 4:
        # read in each rule, translate, extract triples from all training triples
        # format: p, q, r, conf: triple_ids
        rel2id = {}
        with open(str(DATA_PATH) + '/' + dataset+'/rel_id') as f:
            for i,line in enumerate(f):
                rel = line.split('\t')[0]
                rel_id = int(line.split('\t')[1])
                rel2id[rel] = rel_id
        # read in rule set
        with open(os.path.join(path, 'AnyBurl_cons-type_4.txt')) as f, open(path+'/all_cons_4.txt','w') as out:
            for line in f:
                rule, conf = line[:-1].split('\t')
                rel_p, head = rule.split(' <= ')
                rel_q, re_r = head.split(',')
                if conf >= 0.5:
                    try:
                        rule = str(rel2id[rel_p])+','+str(rel2id[rel_q])+','+str(rel2id[re_r])
                        # extract triples from training set
                        triple_ids = []
                        for i, triple in enumerate(train_data):
                            if triple[1] == rel2id[rel_q]:
                                triple_ids.append(str(i))
                        triple_ids_str = ' '.join(triple_ids)
                        out.write('%s\t%s\t%s\n' % (rule, conf, triple_ids_str))
                        print("rule found: " + str(rel_q))
                        # out2.write(line)
                    except KeyError:
                        continue

def prepare_dataset(path, name):
    """
    Given a path to a folder containing tab separated files :
     train, test, valid
    In the format :
    (lhs)\t(rel)\t(rhs)\n
    Maps each entity and relation to a unique id, create corresponding folder
    name in pkg/data, with mapped train/test/valid files.
    Also create to_skip_lhs / to_skip_rhs for filtered metrics and
    rel_id / ent_id for analysis.
    """
    files = ['train', 'valid', 'test']
    entities, relations = set(), set()
    for f in files:
        # file_path = os.path.join(path+'/original/', f)
        file_path = os.path.join(path, f)
        print(file_path)
        to_read = open(file_path, 'r')
        for line in to_read.readlines():
            lhs, rel, rhs = line.strip().split('\t')
            entities.add(lhs)
            entities.add(rhs)
            relations.add(rel)
        to_read.close()
    # print (entities)
    entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
    relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}
    print("{} entities and {} relations".format(len(entities), len(relations)))
    n_relations = len(relations)
    n_entities = len(entities)
    os.makedirs(os.path.join(DATA_PATH, name))
    # write ent to id / rel to id
    for (dic, f) in zip([entities_to_id, relations_to_id], ['ent_id', 'rel_id']):
        ff = open(os.path.join(DATA_PATH, name, f), 'w+')
        for (x, i) in dic.items():
            ff.write("{}\t{}\n".format(x, i))
        ff.close()

    # map train/test/valid with the ids
    for f in files:
        # file_path = os.path.join(path+'/original/', f)
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        examples = []
        for line in to_read.readlines():
            lhs, rel, rhs = line.strip().split('\t')
            try:
                examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs]])
            except ValueError:
                continue
        out = open(Path(DATA_PATH) / name / (f + '.pickle'), 'wb')
        pickle.dump(np.array(examples).astype('uint64'), out)
        out.close()

    print("creating filtering lists")

    # create filtering files
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for f in files:
        examples = pickle.load(open(Path(DATA_PATH) / name / (f + '.pickle'), 'rb'))
        for lhs, rel, rhs in examples:
            # to_skip['lhs'][(rhs, rel + n_relations)].add(lhs)  # reciprocals
            to_skip['lhs'][(rhs, rel)].add(lhs) # no reciprocals
            to_skip['rhs'][(lhs, rel)].add(rhs)

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    out = open(Path(DATA_PATH) / name / 'to_skip.pickle', 'wb')
    pickle.dump(to_skip_final, out)
    out.close()

    examples = pickle.load(open(Path(DATA_PATH) / name / 'train.pickle', 'rb'))
    counters = {
        'lhs': np.zeros(n_entities),
        'rhs': np.zeros(n_entities),
        'both': np.zeros(n_entities)
    }

    for lhs, rel, rhs in examples:
        counters['lhs'][lhs] += 1
        counters['rhs'][rhs] += 1
        counters['both'][lhs] += 1
        counters['both'][rhs] += 1
    for k, v in counters.items():
        counters[k] = v / np.sum(v)
    out = open(Path(DATA_PATH) / name / 'probas.pickle', 'wb')
    pickle.dump(counters, out)
    out.close()
    
    # translate rules
    translate_cons(name, path, examples, 0)


if __name__ == "__main__":
    # datasets = ['FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10']
    datasets = ['FB15K', 'FB237', 'NELL']
    for d in datasets:
        print("Preparing dataset {}".format(d))
        try:
            prepare_dataset(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), 'src_data', d
                ),
                d
            )
        except OSError as e:
            if e.errno == errno.EEXIST:
                print(e)
                print("File exists. skipping...")
            else:
                raise