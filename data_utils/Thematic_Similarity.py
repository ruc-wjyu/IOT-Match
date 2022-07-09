import json
def process_cail_data(file_path, save_path):
    datas = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            for k in item["case_A_dic_small"].keys():
                sents = ""
                for sent in item["case_A_dic_small"][k]:
                    sents += sent[0]
                item["case_A_dic_small"][k] = sents
            for k in item["case_B_dic_small"].keys():
                sents = ""
                for sent in item["case_B_dic_small"][k]:
                    sents += sent[0]
                item["case_B_dic_small"][k] = sents
            datas.append({"case_A":item["case_A_dic_small"], "case_B":item["case_B_dic_small"], "label":item["label"]})
    with open(save_path, 'w') as f:
        for l in datas:
            f.writelines(json.dumps(l, ensure_ascii=False))
            f.write('\n')


def process_ELAM_data(file_path, save_path):
    datas = []
    final_data = []
    for file in file_path:
        with open(file, 'r') as f:
            datas += json.load(f)

    for item in datas:
        final_data.append({"case_A": {item["case_A"][0]["tag"]:item["case_A"][0]["content"], item["case_A"][1]["tag"]:item["case_A"][1]["content"]},
                           "case_B": {item["case_B"][0]["tag"]:item["case_B"][0]["content"], item["case_B"][1]["tag"]:item["case_B"][1]["content"]},
                           "label":item["gold_label"]})
    with open(save_path, 'w') as f:
        for l in final_data:
            f.writelines(json.dumps(l, ensure_ascii=False))
            f.write('\n')

if __name__ == '__main__':
    cail_file_path = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/baselines_datasets/process_data.json"
    our_data_train = "/new_disk2/zhongxiang_sun/code/explanation_project/NER/data/law_train.json"
    our_data_test = "/new_disk2/zhongxiang_sun/code/explanation_project/NER/data/law_test.json"
    our_data_dev = "/new_disk2/zhongxiang_sun/code/explanation_project/NER/data/law_dev.json"

    ELAM_files = [our_data_train, our_data_dev, our_data_test]
    process_cail_data(cail_file_path, "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/baselines_datasets/CAIL_bert_legal_Thematic_Similarity.json")
    process_ELAM_data(ELAM_files, "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/baselines_datasets/ELAM_bert_legal_Thematic_Similarity.json")
