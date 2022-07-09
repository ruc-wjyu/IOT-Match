import sys
sys.path.append("..")
from models.selector_two_multi_class_ot_cail_v2 import Selector2_mul_class, args, load_checkpoint, OT, load_data, data_extract_npy, data_extract_json, device
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
    O_a, I_a = [], []
    O_b, I_b = [], []
    output_batch_A, batch_mask_A = model(case_A)
    output_batch_B, batch_mask_B = model(case_B)
    plan_list = OT_model(output_batch_A, output_batch_B, case_A, case_B, None,
                            batch_mask_A, batch_mask_B, model_type="valid")
    OT_matrix = torch.ge(plan_list, 1 / case_A.shape[1] / args.threshold_ot).long()
    vec_correct_A = torch.argmax(output_batch_A, dim=-1).long()[0][:seq_len_A]
    vec_correct_B = torch.argmax(output_batch_B, dim=-1).long()[0][:seq_len_B]
    relation_A = torch.sum(OT_matrix[0], dim=1)
    relation_B = torch.sum(OT_matrix[0], dim=0)

    for i, label in enumerate(vec_correct_A):
        if label == 1:
            if relation_A[i] >= 1:
                I_a.append(i)
            else:
                O_a.append(i)

    for i, label in enumerate(vec_correct_B):
        if label == 1:
            if relation_B[i] >= 1:
                I_b.append(i)
            else:
                O_b.append(i)
    O, I = [O_a, O_b], [I_a, I_b]

    return O, I, [vec_correct_A+(torch.ge(relation_A[:seq_len_A], 1)*1)*vec_correct_A, vec_correct_B+(torch.ge(relation_B[:seq_len_B], 1)*1)*vec_correct_B]




def generate_text_cluster(case_a, case_b, d, O, I, all_true,  I_true):
    source_1_a = ''.join(["[O]" + case_a[0][i] for i in O[0]] + ["[I]" + case_a[0][i] for i in I[0]])

    source_1_b = ''.join(["[O]" + case_b[0][i] for i in O[1]] + ["[I]" + case_b[0][i] for i in I[1]])

    source_2_a = ''.join(["[O]" + case_a[0][i] for i in all_true[0] if i not in I_true[0]] + ["[I]" + case_a[0][i] for i in I_true[0]])

    source_2_b = ''.join(["[O]" + case_b[0][i] for i in all_true[1] if i not in I_true[1]] + ["[I]" + case_b[0][i] for i in I_true[1]])

    result = {
        'source_1': source_1_a + source_1_b,
        'source_2': source_2_a + source_2_b,
        'explanation': d['explanation'],
        'source_1_dis': [source_1_a, source_1_b],
        'source_2_dis': [source_2_a, source_2_b],
        'label': d['label']
    }
    return result


def get_extract_text(case_a, prediction):
    source_1_a = ''
    for i, output_class in enumerate(prediction):
        if output_class == 1:
            source_1_a += "[O]" + case_a[0][i]
        elif output_class == 2:
            source_1_a += "[I]" + case_a[0][i]
        else:
            pass
    return source_1_a


def get_extract_text_wo_token(case_a, prediction):
    source_1_a = ''
    for i, output_class in enumerate(prediction):
        if output_class != 0:
            source_1_a += case_a[0][i]
        else:
            pass
    return source_1_a


def generate_text_sort(case_a, case_b, d, prediction, label):

    source_1_a = get_extract_text(case_a, prediction[0])
    source_1_b = get_extract_text(case_b, prediction[1])
    source_2_a = get_extract_text(case_a, label[0])
    source_2_b = get_extract_text(case_b, label[1])

    result = {
        'source_1': source_1_a + source_1_b,
        'source_2': source_2_a + source_2_b,
        'explanation': d['explanation'],
        'source_1_dis': [source_1_a, source_1_b],
        'source_2_dis': [source_2_a, source_2_b],
        'label': d['label']
    }
    return result


def generate_text_wo_token(case_a, case_b, d, prediction, label):

    source_1_a = get_extract_text_wo_token(case_a, prediction[0])
    source_1_b = get_extract_text_wo_token(case_b, prediction[1])
    source_2_a = get_extract_text_wo_token(case_a, label[0])
    source_2_b = get_extract_text_wo_token(case_b, label[1])

    result = {
        'source_1': source_1_a + source_1_b,
        'source_2': source_2_a + source_2_b,
        'explanation': d['explanation'],
        'source_1_dis': [source_1_a, source_1_b],
        'source_2_dis': [source_2_a, source_2_b],
        'label': d['label']
    }
    return result


def fold_convert_cail_ot(data, data_x, type, generate=False, generate_mode = 'cluster'):
    """每一fold用对应的模型做数据转换
    """

    with torch.no_grad():
        model = Selector2_mul_class(args.input_size, args.hidden_size, kernel_size=args.kernel_size, dilation_rate=[1, 2, 4, 8, 1, 1])
        load_checkpoint(model, None, 2, "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/models/weights/extract/cail_extract-criterion-BCEFocal-onehot-1-ot_mode-max-convert_to_onehot-1-weight-100-simot-1-simpercent-1.0.pkl")
        model = model.to(device)
        ot_model = OT()
        ot_model = ot_model.to(device)
        load_checkpoint(ot_model, None, 2, "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/models/weights/extract/cail_extract_ot-criterion-BCEFocal-onehot-1-ot_mode-max-convert_to_onehot-1-weight-100-simot-1-simpercent-1.0.pkl")
        results = []
        print(type+"ing")
        for i, d in enumerate(data):
            if type == 'match' and d["label"] == 2 or type == 'midmatch' and d["label"] == 1 or type == 'dismatch' and d["label"] == 0:
                case_a = d['case_A']
                case_b = d['case_B']
                important_A, important_B = [], []
                data_y_seven_class_A, data_y_seven_class_B = [0]*len(case_a[0]), [0]*len(case_b[0])
                for pos in d['relation_label']:
                    row, col = pos[0], pos[-1]
                    important_A.append(row)
                    important_B.append(col)

                for j in case_a[1]:
                    if j[0] in important_A:
                        data_y_seven_class_A[j[0]] = j[1] + 1
                    else:
                        data_y_seven_class_A[j[0]] = j[1]

                for j in case_b[1]:
                    if j[0] in important_B:
                        data_y_seven_class_B[j[0]] = j[1] + 1
                    else:
                        data_y_seven_class_B[j[0]] = j[1]
                label = [data_y_seven_class_A, data_y_seven_class_B]



                O, I, prediction = model_class(model, ot_model, torch.tensor(np.expand_dims(data_x[2*i], axis=0), device=device),
                                                     torch.tensor(np.expand_dims(data_x[2*i+1], axis=0), device=device),len(case_a[0]), len(case_b[0]))

                all_true, I_true = [], []

                temp_a, temp_b = [], []
                for i in d['relation_label']:
                    temp_a.append(i[0])
                    temp_b.append(i[1])
                I_true.append(temp_a)
                I_true.append(temp_b)


                case_a_temp = []
                for i in case_a[1]:
                    if i[1] == 1:
                        case_a_temp.append(i[0])
                all_true.append(case_a_temp)

                case_b_temp = []
                for i in case_b[1]:
                    if i[1] == 1:
                        case_b_temp.append(i[0])

                all_true.append(case_b_temp)
                if generate:
                    if generate_mode == 'cluster':
                        results.append(generate_text_cluster(case_a, case_b, d, O, I, all_true, I_true))
                    elif generate_mode == 'sort':
                        results.append(generate_text_sort(case_a, case_b, d, prediction, label))
                    else:
                        results.append(generate_text_wo_token(case_a, case_b, d, prediction, label))

        if generate:
            return results



def convert(filename, data, data_x, type,  generate_mode):
    """转换为生成式数据
    """
    total_results = fold_convert_cail_ot(data, data_x, type, generate=True, generate_mode=generate_mode)

    with open(filename, 'w') as f:
        for item in total_results:
            f.writelines(json.dumps(item, ensure_ascii=False))
            f.write('\n')



if __name__ == '__main__':

    data = load_data(data_extract_json)
    data_x = np.load(data_extract_npy)
    da_type = "wo_token"
    match_data_seq2seq_json = '../dataset/match_data_seq2seq_{}.json'.format(da_type)
    midmatch_data_seq2seq_json = '../dataset/midmatch_data_seq2seq_{}.json'.format(da_type)
    dismatch_data_seq2seq_json = '../dataset/dismatch_data_seq2seq_{}.json'.format(da_type)
    convert(match_data_seq2seq_json, data, data_x, type='match', generate_mode=da_type)
    convert(midmatch_data_seq2seq_json, data, data_x, type='midmatch', generate_mode=da_type)
    convert(dismatch_data_seq2seq_json, data, data_x, type='dismatch', generate_mode=da_type)


    print(u'输出over！')
