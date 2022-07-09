import sys
sys.path.append("..")
from models.selector_two_multi_class_ot_v3 import Selector2_mul_class, args, load_checkpoint, OT, load_data, data_extract_npy, data_extract_json, device
import torch
from utils.snippets import *
import torch
import json



def fold_convert_our_data_ot(data, data_x, type, generate=False, generate_mode = 'cluster'):
    """每一fold用对应的模型做数据转换
    """
    max_len = 0
    with torch.no_grad():
        results = []
        print(type+"ing")
        for i, d in enumerate(data):
            if type == 'match' and d["label"] == 2 or type == 'midmatch' and d["label"] == 1 or type == 'dismatch' and d["label"] == 0:
                case_a = d['case_A']
                case_b = d['case_B']
                source_1_a = ''.join(case_a[0])
                source_1_b = ''.join(case_b[0])
                source_2_a = ''.join(case_a[0])
                source_2_b = ''.join(case_b[0])
                max_len = max(max_len, len(source_1_a+source_1_b))
                max_len = max(max_len, len(source_2_a+source_2_b))

                # result = {
                #     'source_1': source_1_a + source_1_b,
                #     'source_2': source_2_a + source_2_b,
                #     'explanation': '；'.join(list(d['explanation'].values())),
                #     'source_1_dis': [source_1_a, source_1_b],
                #     'source_2_dis': [source_2_a, source_2_b],
                #     'label': d['label']
                # }
                result = {
                    'source_1': source_1_a + source_1_b,
                    'source_2': source_2_a + source_2_b,
                    'explanation': d['explanation'],
                    'source_1_dis': [source_1_a, source_1_b],
                    'source_2_dis': [source_2_a, source_2_b],
                    'label': d['label']
                }
                results.append(result)
        print(max_len)
        if generate:
            return results




def convert(filename, data, data_x, type,  generate_mode):
    """转换为生成式数据
    """
    total_results = fold_convert_our_data_ot(data, data_x, type, generate=True, generate_mode=generate_mode)

    with open(filename, 'w') as f:
        for item in total_results:
            f.writelines(json.dumps(item, ensure_ascii=False))
            f.write('\n')



if __name__ == '__main__':
    data_extract_json = '../dataset/data_extract.json'
    data_extract_npy = '../dataset/data_extract.npy'
    data = load_data(data_extract_json)
    data_x = np.load(data_extract_npy)
    da_type = "sort"
    match_data_seq2seq_json = '../dataset/match_data_seq2seq_NILE.json'
    midmatch_data_seq2seq_json = '../dataset/midmatch_data_seq2seq_NILE.json'
    dismatch_data_seq2seq_json = '../dataset/dismatch_data_seq2seq_NILE.json'
    convert(match_data_seq2seq_json, data, data_x, type='match', generate_mode=da_type)
    convert(midmatch_data_seq2seq_json, data, data_x, type='midmatch',  generate_mode=da_type)
    convert(dismatch_data_seq2seq_json, data, data_x, type='dismatch',  generate_mode=da_type)


    print(u'输出over！')
