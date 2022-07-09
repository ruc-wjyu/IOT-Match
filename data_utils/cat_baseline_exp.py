import json
exp_file = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/data_prediction_t5_wo_token.json"
wo_rationale = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/baselines_datasets/ELAM_bert_legal_wo_rationale.json"
rationale = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/baselines_datasets/ELAM_bert_legal_rationale.json"
all_sents = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/baselines_datasets/ELAM_bert_legal_all_sents.json"

save_wo_rationale = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/stage3/data_prediction_wo_rationale.json"
save_rationale = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/stage3/data_prediction_rationale.json"
save_all_sents = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/stage3/data_prediction_all_sents.json"


def cat_file(exp_files, second_file, save_file):
    match_data = []
    midmatch_data = []
    dismatch_data = []
    exp_data = []
    with open(second_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            if item['label'] == 2:
                match_data.append(item)
            elif item['label'] == 1:
                midmatch_data.append(item)
            elif item['label'] == 0:
                dismatch_data.append(item)
            else:
                exit()
    datas = match_data + midmatch_data + dismatch_data
    with open(exp_files, 'r') as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            item['case_a'] = datas[i]['case_a']
            item['case_b'] = datas[i]['case_b']
            exp_data.append(item)

    with open(save_file, 'w') as f:
        for item in exp_data:
            f.writelines(json.dumps(item, ensure_ascii=False))
            f.write('\n')

cat_file(exp_file, wo_rationale, save_wo_rationale)
cat_file(exp_file, rationale, save_rationale)
cat_file(exp_file, all_sents, save_all_sents)








