import json
dismatch_files = ["/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/dismatch_data_prediction_t5_NILE.json",
         "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/dismatch_data_prediction_t5_wo_token.json",
         "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/dismatch_data_prediction_t5_xLIRE.json"]

midmatch_files = ["/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/midmatch_data_prediction_t5_NILE.json",
         "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/midmatch_data_prediction_t5_wo_token.json",
         "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/midmatch_data_prediction_t5_xLIRE.json"]

match_files = ["/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/match_data_prediction_t5_NILE.json",
         "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/match_data_prediction_t5_wo_token.json",
         "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/match_data_prediction_t5_xLIRE.json"]

wo_token_file = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/data_prediction_t5_wo_token.json"
NILE_file = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/data_prediction_t5_NILE.json"
xLIRE_file = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/data_prediction_t5_xLIRE.json"

NILE_data = []
with open(match_files[0], 'r') as f:
    for line in f:
        NILE_data.append(json.loads(line))

with open(midmatch_files[0], 'r') as f:
    for i, line in enumerate(f):
        item = json.loads(line)
        NILE_data[i]['exp'] += item['exp']

with open(dismatch_files[0], 'r') as f:
    for i, line in enumerate(f):
        item = json.loads(line)
        NILE_data[i]['exp'] += item['exp']

wo_token_data = []
with open(match_files[1], 'r') as f:
    for line in f:
        wo_token_data.append(json.loads(line))

with open(midmatch_files[1], 'r') as f:
    for i, line in enumerate(f):
        item = json.loads(line)
        wo_token_data[i]['exp'] += item['exp']

with open(dismatch_files[1], 'r') as f:
    for i, line in enumerate(f):
        item = json.loads(line)
        wo_token_data[i]['exp'] += item['exp']

xLIRE_data = []
with open(match_files[2], 'r') as f:
    for line in f:
        xLIRE_data.append(json.loads(line))

with open(midmatch_files[2], 'r') as f:
    for i, line in enumerate(f):
        item = json.loads(line)
        xLIRE_data[i]['exp'] += item['exp']

with open(dismatch_files[2], 'r') as f:
    for i, line in enumerate(f):
        item = json.loads(line)
        xLIRE_data[i]['exp'] += item['exp']

with open(wo_token_file, 'w') as f:
    for item in wo_token_data:
        f.writelines(json.dumps(item, ensure_ascii=False))
        f.write('\n')

with open(NILE_file, 'w') as f:
    for item in NILE_data:
        f.writelines(json.dumps(item, ensure_ascii=False))
        f.write('\n')

with open(xLIRE_file, 'w') as f:
    for item in xLIRE_data:
        f.writelines(json.dumps(item, ensure_ascii=False))
        f.write('\n')

