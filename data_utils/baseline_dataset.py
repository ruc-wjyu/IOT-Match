"""
用于构造：
 Bert_legal (all sents)
 Bert_legal (rational)
 Bert_legal (all sents-rational)
 数据集
"""
import sys
sys.path.append("..")
data_type = 'CAIL'
if data_type == 'CAIL':
    from models.selector_two_multi_class_ot_cail_v2 import Selector2_mul_class, args, load_checkpoint, OT, load_data, \
        data_extract_npy, data_extract_json, device
elif data_type =='ELAM':
    from models.selector_two_multi_class_ot_v3 import Selector2_mul_class, args, load_checkpoint, OT, load_data, data_extract_npy, data_extract_json, device
import torch
from utils.snippets import *
import torch
import json
"""
todo 需要添加一个分类器，对相似句子与不相似句子进行分类
"""
def model_class(model, OT_model, case_A, case_B, seq_len_A, seq_len_B):
    """

    :param model:
    :param OT_model:
    :param case_A:
    :param case_B:
    :return: AO, YO, ZO, AI, YI, ZI
    """

    output_batch_A, batch_mask_A = model(case_A)
    output_batch_B, batch_mask_B = model(case_B)
    plan_list = OT_model(output_batch_A, output_batch_B, case_A, case_B, None,
                            batch_mask_A, batch_mask_B, model_type='valid')
    OT_matrix = torch.ge(plan_list, 1 / case_A.shape[1] / args.threshold_ot).long()
    vec_correct_A = torch.argmax(output_batch_A, dim=-1).long()[0][:seq_len_A]
    vec_correct_B = torch.argmax(output_batch_B, dim=-1).long()[0][:seq_len_B]
    relation_A = torch.sum(OT_matrix[0], dim=1)
    relation_B = torch.sum(OT_matrix[0], dim=0)

    if data_type == 'CAIL':
        return [vec_correct_A+(torch.ge(relation_A[:seq_len_A], 1)*1)*vec_correct_A, vec_correct_B+(torch.ge(relation_B[:seq_len_B], 1)*1)*vec_correct_B]
    elif data_type == 'ELAM':
        return [vec_correct_A+(torch.ge(relation_A[:seq_len_A], 1)*3)*vec_correct_A, vec_correct_B+(torch.ge(relation_B[:seq_len_B], 1)*3)*vec_correct_B]
    else:
        exit()




def get_extract_text_wo_token(case_a, prediction):
    all_sents_a, rationale_a, all_wo_rationale_a = '', '', ''
    for i, output_class in enumerate(prediction):
        if output_class != 0:
             rationale_a += case_a[0][i]
        else:
             all_wo_rationale_a += case_a[0][i]
        all_sents_a += case_a[0][i]
    return all_sents_a, rationale_a, all_wo_rationale_a


def generate_text_wo_token(case_a, case_b, d, prediction):

    all_sents_a, rationale_a, all_wo_rationale_a = get_extract_text_wo_token(case_a, prediction[0])
    all_sents_b, rationale_b, all_wo_rationale_b = get_extract_text_wo_token(case_b, prediction[1])

    result_all_sents = {
        'case_a': all_sents_a,
        'case_b': all_sents_b,
        'label': d['label']
    }
    result_rationale = {
        'case_a': rationale_a,
        'case_b': rationale_b,
        'label': d['label']
    }

    result_all_wo_rationale = {
        'case_a': all_wo_rationale_a,
        'case_b': all_wo_rationale_b,
        'label': d['label']
    }

    return result_all_sents, result_rationale, result_all_wo_rationale


def fold_convert_our_data_ot(data, data_x):
    """每一fold用对应的模型做数据转换
    """

    with torch.no_grad():
        model = Selector2_mul_class(args.input_size, args.hidden_size, kernel_size=args.kernel_size, dilation_rate=[1, 2, 4, 8, 1, 1])
        if data_type == 'CAIL':
            load_checkpoint(model, None, 2, "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/models/weights/extract/cail_extract-criterion-BCEFocal-onehot-1-ot_mode-max-convert_to_onehot-1-weight-100-simot-1-simpercent-1.0.pkl")
        elif data_type == 'ELAM':
            load_checkpoint(model, None, 2, "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/models/weights/extract/extract-criterion-BCEFocal-onehot-1-ot_mode-max-convert_to_onehot-1-weight-100-simot-1-simpercent-1.0.pkl")
        else:
            exit()
        model = model.to(device)
        ot_model = OT()
        ot_model = ot_model.to(device)
        if data_type == 'ELAM':
            load_checkpoint(ot_model, None, 2, "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/models/weights/extract/extract_ot-criterion-BCEFocal-onehot-1-ot_mode-max-convert_to_onehot-1-weight-100-simot-1-simpercent-1.0.pkl")
        elif data_type == 'CAIL':
            load_checkpoint(ot_model, None, 2, "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/models/weights/extract/cail_extract_ot-criterion-BCEFocal-onehot-1-ot_mode-max-convert_to_onehot-1-weight-100-simot-1-simpercent-1.0.pkl")
        else:
            exit()
        results_all_sents, results_rationale, results_wo_rationale = [], [], []
        for i, d in enumerate(data):
            case_a = d['case_A']
            case_b = d['case_B']




            prediction = model_class(model, ot_model, torch.tensor(np.expand_dims(data_x[2*i], axis=0), device=device),
                                                 torch.tensor(np.expand_dims(data_x[2*i+1], axis=0), device=device),len(case_a[0]), len(case_b[0]))


            all_sents, rationale, wo_rationale = generate_text_wo_token(case_a, case_b, d, prediction)
            results_all_sents.append(all_sents)
            results_rationale.append(rationale)
            results_wo_rationale.append(wo_rationale)

        return [results_all_sents, results_rationale, results_wo_rationale]



def convert(filename, data, data_x):
    """转换为生成式数据
    """
    total_results = fold_convert_our_data_ot(data, data_x)
    for i in range(len(total_results)):
        with open(filename[i], 'w') as f:
            for item in total_results[i]:
                f.writelines(json.dumps(item, ensure_ascii=False))
                f.write('\n')



if __name__ == '__main__':

    data = load_data(data_extract_json)
    data_x = np.load(data_extract_npy)
    bert_legal_all_sents_json = '../dataset/baselines_datasets/{}_bert_legal_all_sents.json'.format(data_type)
    bert_legal_rationale_json = '../dataset/baselines_datasets/{}_bert_legal_rationale.json'.format(data_type)
    bert_legal_wo_rationale_json = '../dataset/baselines_datasets/{}_bert_legal_wo_rationale.json'.format(data_type)


    convert([bert_legal_all_sents_json, bert_legal_rationale_json, bert_legal_wo_rationale_json], data, data_x)


    print(u'输出over！')
