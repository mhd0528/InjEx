########
#### this file select rules with certain confidence
########
import os

# rule_path = './rules/FB237'
# rule_path = './rules/NELL-one'
# rule_path = './rules/Wiki-one'
# rule_path = './rules/FB237-target'
rule_path = './rules/Yago'
# out_path = './'
rule_type = 4

time_list = ['rules-10', 'rules-50', 'rules-100']
out_f = open(os.path.join(rule_path, 'AnyBurl_cons-type_4.txt'), 'w')
r_set = set()
for t in time_list:
    with open(os.path.join(rule_path, t)) as f:
        for line in f:
            _, _, conf, rule = line[:-1].split('\t')
            conf = float(conf)
            prefix = ''
            #### only extract rules with conf > 0.5
            if conf > 0.0:
                tail, head = rule.split(' <= ')
                tail = tail[:-1]
                head = head[:-1]
                # if len(head.split(', ')) > 1:
                #     print(len(head.split(', ')))
                #### only extract entailment rules
                ## check entailment rule
                if rule_type == 0:
                    if head == '' or len(head.split(', ')) != 1:
                        continue
                    ## check if is inverse rule
                    r_p, e_p = tail.split('(')
                    r_q, e_q = head.split('(')
                    if r_p == r_q or (r_p, r_q) in r_set:
                        continue
                    r_set.add((r_p, r_q))
                    e_p1, e_p2 = e_p.split(',')
                    e_q1, e_q2 = e_q.split(',')
                    if e_p1 == e_q2 or e_q2 == e_p1:
                        prefix = '-'                    
                    # target_rels = ['/award/award_category/category_of', '/film/film/country', '/award/award_nominee/award_nominations./award/award_nomination/award_nominee', '/people/person/profession', '/base/aareas/schema/administrative_area/administrative_parent', '/film/film/estimated_budget./measurement_unit/dated_money_value/currency', '/location/statistical_region/gdp_nominal_per_capita./measurement_unit/dated_money_value/currency', '/people/person/nationality']
                    # if r_p in target_rels:
                    out_f.write(prefix + r_p + ',' + r_q + '\t' + str(conf) + '\n')
                #### extract type4 rules
                elif rule_type == 4:
                    if len(head.split(', ')) != 2:
                        continue
                    ## check if is inverse rule
                    r_p, e_p = tail.split('(')
                    r_q , r_r = head.split(', ')
                    r_q, e_q = r_q.split('(')
                    r_r, e_r = r_r.split('(')
                    e_q = e_q[:-1]
                    if r_p == r_q or r_p == r_r or (r_p, r_q, r_r) in r_set:
                        continue
                    r_set.add((r_p, r_q, r_r))
                    e_q1, e_q2 = e_q.split(',')
                    e_r1, e_r2 = e_r.split(',')
                    # print(r_p, r_q, r_r)
                    # print(e_q1, e_q2, e_r1, e_r2)
                    if e_q2 == e_r1:
                        out_f.write(prefix + r_p + ' <= ' + r_q + ',' + r_r + '\t' + str(conf) + '\n')