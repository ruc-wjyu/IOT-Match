import json
our_data_file = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/heat_map/temp.json"
xLIRE_file = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/data_prediction_t5_xLIRE.json"
wo_token_file = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/data_prediction_t5_wo_token.json"
NILE_file = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/data_prediction_t5_NILE.json"

with open(our_data_file, 'r') as f:
    for line in f:
        our_data = json.loads(line)

data_NILE = []
with open(NILE_file, 'r') as f:
    for line in f:
        data_NILE.append(json.loads(line))

data_xLIRE = []
with open(xLIRE_file, 'r') as f:
    for line in f:
        data_xLIRE.append(json.loads(line))

data_wo_token = []
with open(wo_token_file, 'r') as f:
    for line in f:
        data_wo_token.append(json.loads(line))


print("NILE  ", data_NILE[2985]['exp'][1])

print("xLIRE  ", data_xLIRE[2985]['exp'][1])

print("wo_token  ", data_wo_token[2985]['exp'][1])

print("golden exp", data_wo_token[2985]["explanation"])
# for i, data in enumerate(data_NILE):
#     if data["source_2_dis"][0] == "".join(our_data["case_A"][0]) and data["source_2_dis"][1] == "".join(our_data["case_B"][0]):
#         print(i)




