import os

no_idx_path = './entities.txt'
idx_path = './entities.dict'

cnt = 0
out_f = open(idx_path, mode='w+', encoding='utf-8')
with open('entities.txt', encoding='utf-8', mode='r') as f:
    while True:
        ent = f.readline().strip()
        if ent:
            out_f.write(str(cnt) + '\t' + ent + '\n')
            cnt += 1
        else:
            break

f.close()
out_f.close()