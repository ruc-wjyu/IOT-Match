import sys
sys.path.append("..")
from baselines.selector_baselines.Attention_seven_class import Selector2_mul_class, args, load_checkpoint, attention, load_data, data_extract_npy, data_extract_json, device
import torch
from utils.snippets import *
import torch
import json

def get_predition(batch_A, batch_B, case_a_important, case_b_important):
    relation_A = torch.ge(case_a_important, args.threshold).long()
    relation_B = torch.ge(case_b_important, args.threshold).long()
    if args.data_type == 'CAIL':
        vec_correct_A = torch.argmax(batch_A, dim=-1) + (relation_A * 1).squeeze() * torch.ge(torch.argmax(batch_A, dim=-1), 1)
        vec_correct_B = torch.argmax(batch_B, dim=-1) + (relation_B * 1).squeeze() * torch.ge(torch.argmax(batch_B, dim=-1), 1)
    else:
        vec_correct_A = torch.argmax(batch_A, dim=-1) + (relation_A * 3).squeeze() * torch.ge(torch.argmax(batch_A, dim=-1), 1)
        vec_correct_B = torch.argmax(batch_B, dim=-1) + (relation_B * 3).squeeze() * torch.ge(torch.argmax(batch_B, dim=-1), 1)

    return vec_correct_A, vec_correct_B


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
    case_a_important, case_b_important = OT_model(output_batch_A, output_batch_B, case_A, case_B, None, None,
                            batch_mask_A, batch_mask_B)

    seven_prediction_A, seven_prediction_B = get_predition(output_batch_A.clone(), output_batch_B.clone(),
                                                         case_a_important, case_b_important)

    return seven_prediction_A, seven_prediction_B


def get_extract_text(case_a, prediction):
    source_1_a = ''
    for i, output_class in enumerate(prediction[0]):
        if i >= len(case_a[0]):
            break
        if output_class == 0:
            source_1_a += case_a[0][i]
        else:
            source_1_a += '[' + case_a[0][i] + ']'
    return source_1_a





def generate_text_sort(case_a, case_b, d, prediction, label):

    source_1_a = get_extract_text(case_a, prediction[0])
    source_1_b = get_extract_text(case_b, prediction[1])
    source_2_a = get_extract_text(case_a, prediction[0])
    source_2_b = get_extract_text(case_b, prediction[1])
    if args.data_type=='CAIL':
        result = {
            'source_1': source_1_a + source_1_b,
            'source_2': source_2_a + source_2_b,
            'explanation': d['explanation'],
            'source_1_dis': [source_1_a, source_1_b],
            'source_2_dis': [source_2_a, source_2_b],
            'label': d['label']
        }
    else:
        result = {
            'source_1': source_1_a + source_1_b,
            'source_2': source_2_a + source_2_b,
            'explanation': '；'.join(list(d['explanation'].values())),
            'source_1_dis': [source_1_a, source_1_b],
            'source_2_dis': [source_2_a, source_2_b],
            'label': d['label']
        }
    return result




def fold_convert_our_data_ot(data, data_x, type, generate=False):
    """每一fold用对应的模型做数据转换
    """

    with torch.no_grad():
        model = Selector2_mul_class(args.input_size, args.hidden_size, kernel_size=args.kernel_size, dilation_rate=[1, 2, 4, 8, 1, 1])
        load_checkpoint(model, None, 2, "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/baselines/selector_baselines/weights/extract/extract-{}.pkl".format(args.data_type))
        model = model.to(device)
        ot_model = attention()
        ot_model = ot_model.to(device)
        load_checkpoint(ot_model, None, 2, "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/baselines/selector_baselines/weights/extract/extract_attention-{}.pkl".format(args.data_type))
        results = []
        print(type+"ing")
        for i, d in enumerate(data):
            if type == 'match' and d["label"] == 2 or type == 'midmatch' and d["label"] == 1 or type == 'dismatch' and d["label"] == 0:
                case_a = d['case_A']
                case_b = d['case_B']
                important_A, important_B = [], []
                data_y_seven_class_A, data_y_seven_class_B = [0]*len(case_a[0]), [0]*len(case_b[0])
                if args.data_type == 'CAIL':
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
                else:
                    for pos_list in d['relation_label'].values():
                        for pos in pos_list:
                            row, col = pos[0], pos[-1]
                            important_A.append(row)
                            important_B.append(col)

                    for j in case_a[1]:
                        if j[0] in important_A:
                            data_y_seven_class_A[j[0]] = j[1] + 3
                        else:
                            data_y_seven_class_A[j[0]] = j[1]

                    for j in case_b[1]:
                        if j[0] in important_B:
                            data_y_seven_class_B[j[0]] = j[1] + 3
                        else:
                            data_y_seven_class_B[j[0]] = j[1]

                label = [data_y_seven_class_A, data_y_seven_class_B]
                data_x_a = torch.tensor(np.expand_dims(data_x[2*i], axis=0), device=device)
                data_x_b = torch.tensor(np.expand_dims(data_x[2*i+1], axis=0), device=device)

                seven_prediction_A, seven_prediction_B = model_class(model, ot_model, data_x_a, data_x_b, len(case_a[0]), len(case_b[0]))

                prediction = [seven_prediction_A, seven_prediction_B]
                if generate:
                    results.append(generate_text_sort(case_a, case_b, d, prediction, label))

        if generate:
            return results


def convert(filename, data, data_x, type):
    """转换为生成式数据
    """
    total_results = fold_convert_our_data_ot(data, data_x, type, generate=True)

    with open(filename, 'w') as f:
        for item in total_results:
            f.writelines(json.dumps(item, ensure_ascii=False))
            f.write('\n')



if __name__ == '__main__':
    if args.data_type == 'CAIL':
        data_extract_json = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/data_extract.json"
        data_extract_npy = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/data_extract.npy"
    else:
        data_extract_json = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/data_extract.json"
        data_extract_npy = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/data_extract.npy"

    data = load_data(data_extract_json)
    data_x = np.load(data_extract_npy)
    da_type = 'xLIRE'
    match_data_seq2seq_json = '../dataset/our_data/match_data_seq2seq_{}.json'.format(da_type)  # 文件夹位置显示不同的data
    midmatch_data_seq2seq_json = '../dataset/our_data/midmatch_data_seq2seq_{}.json'.format(da_type)
    dismatch_data_seq2seq_json = '../dataset/our_data/dismatch_data_seq2seq_{}.json'.format(da_type)
    convert(match_data_seq2seq_json, data, data_x, type='match')
    convert(midmatch_data_seq2seq_json, data, data_x, type='midmatch')
    convert(dismatch_data_seq2seq_json, data, data_x, type='dismatch')


    print(u'输出over！')
