import json
import random

ELAM_xLIRE_file = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/data_prediction_t5_xLIRE.json"
ELAM_wo_token_file = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/data_prediction_t5_wo_token.json"
ELAM_NILE_file = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/data_prediction_t5_NILE.json"

ELAM_xLIRE_file_save = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/ELAM_sample_data_prediction_t5_xLIRE.json"
ELAM_wo_token_file_save = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/ELAM_sample_data_prediction_t5_wo_token.json"
ELAM_NILE_file_save = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/ELAM_sample_data_prediction_t5_NILE.json"



CAIL_xLIRE_file = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/data_prediction_t5_xLIRE.json"
CAIL_wo_token_file = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/data_prediction_t5_wo_token.json"
CAIL_NILE_file = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/data_prediction_t5_NILE.json"
CAIL_xLIRE_file_save = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/CAIL_sample_data_prediction_t5_xLIRE.json"
CAIL_wo_token_file_save = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/CAIL_sample_data_prediction_t5_wo_token.json"
CAIL_NILE_file_save = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/CAIL_sample_data_prediction_t5_NILE.json"



def proecess(file_path, save_path):
    random.seed(1)
    match_list = []
    midmatch_list = []
    dismatch_list = []

    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            if item['label'] == 2:
                match_list.append({"case_A":item["source_1_dis"][0], "case_B":item["source_1_dis"][1], "explanation":item["exp"][2-item["label"]], "golden_exp":item["explanation"]})
            elif item['label'] == 1:
                midmatch_list.append({"case_A":item["source_1_dis"][0], "case_B":item["source_1_dis"][1], "explanation":item["exp"][2-item["label"]], "golden_exp":item["explanation"]})
            elif item['label'] == 0:
                dismatch_list.append({"case_A":item["source_1_dis"][0], "case_B":item["source_1_dis"][1], "explanation":item["exp"][2-item["label"]], "golden_exp":item["explanation"]})

    dev_test = dismatch_list[int(0.8*len(dismatch_list)):] + midmatch_list[int(0.8*len(midmatch_list)):] #+ match_list[int(0.8*len(match_list)):]
    dev_test_sample = random.sample(dev_test, 100)
    with open(save_path, 'w') as f:
        for line in dev_test_sample:
            f.writelines(json.dumps(line, ensure_ascii=False))
            f.write('\n')

if __name__ == '__main__':
    proecess(ELAM_xLIRE_file, ELAM_xLIRE_file_save)
    proecess(ELAM_NILE_file, ELAM_NILE_file_save)
    proecess(ELAM_wo_token_file, ELAM_wo_token_file_save)
    proecess(CAIL_xLIRE_file, CAIL_xLIRE_file_save)
    proecess(CAIL_NILE_file, CAIL_NILE_file_save)
    proecess(CAIL_wo_token_file, CAIL_wo_token_file_save)








