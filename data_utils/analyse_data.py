import numpy as np
import json
import re
data_extract_json = '../dataset/our_data/data_extract.json'
data_extract_npy = '../dataset/our_data/data_extract.npy'

def load_data(filename):
    """加载数据
    返回：[(texts, labels, exp)]
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            D.append(json.loads(l))
    return D
data = load_data(data_extract_json)
label_0, label_1, label_2 = 0, 0, 0
A_all, Y_all, Z_all, A_I, Y_I, Z_I = 0, 0, 0, 0, 0, 0
# sentence = 0
# for i, d in enumerate(data):
#     if d["label"] == 0:
#         label_0 += 1
#     elif d["label"] == 1:
#         label_1 += 1
#     else:
#         label_2 += 1
#
#     case_A_list = d["case_A"][1]
#     sentence += len(d["case_A"][0])
#     case_B_list = d["case_B"][1]
#     sentence += len(d["case_B"][0])
#     for item in case_A_list + case_B_list:
#         if item[1] == 1:
#             A_all += 1
#         elif item[1] == 2:
#             Y_all += 1
#         elif item[1] == 3:
#             Z_all += 1
#         else:
#             exit()
#     case_A_I = set()
#     case_B_I = set()
#     for pa in d["relation_label"]["relation_label_aqss"]:
#         case_A_I.add(pa[0])
#         case_B_I.add(pa[1])
#     A_I += len(case_A_I) + len(case_B_I)
#     case_A_I = set()
#     case_B_I = set()
#     for pa in d["relation_label"]["relation_label_yjss"]:
#         case_A_I.add(pa[0])
#         case_B_I.add(pa[1])
#     Y_I += len(case_A_I) + len(case_B_I)
#
#     case_A_I = set()
#     case_B_I = set()
#     for pa in d["relation_label"]["relation_label_zyjd"]:
#         case_A_I.add(pa[0])
#         case_B_I.add(pa[1])
#     Z_I += len(case_A_I) + len(case_B_I)
# print("label 0: {} label 1: {} label 2: {}".format(label_0, label_1, label_2))
# print("A_all: {} , Y_all: {}, Z_all: {}, A_I: {}, Y_I： {}, Z_I： {}".format(A_all, Y_all, Z_all, A_I, Y_I, Z_I))
#print(sentence/(label_0+label_1+label_2))
all, I = 0, 0
sentence = 0
for i, d in enumerate(data):
    if d["label"] == 0:
        label_0 += 1
    elif d["label"] == 1:
        label_1 += 1
    else:
        label_2 += 1

    case_A_list = d["case_A"][1]
    sentence += len(d["case_A"][0])
    case_B_list = d["case_B"][1]
    sentence += len(d["case_B"][0])
    all += len(case_A_list + case_B_list)

    case_A_I = set()
    case_B_I = set()
    for pa in d["relation_label"]:
        case_A_I.add(pa[0])
        case_B_I.add(pa[1])
    I += len(case_A_I) + len(case_B_I)
print("pro {}  con {} per {}".format(I, all - I, all/(2*len(data))))



sentence = 0
word_num = 0
for i, d in enumerate(data[:int(0.8*len(data))]):
    sentence += len(d["case_A"][0])
    for item in d["case_A"][0]:
        word_num += len(item)
    sentence += len(d["case_B"][0])
    for item in d["case_B"][0]:
        word_num += len(item)
print("train_avg_sentence_length {} avg_sentence_num {}".format(word_num/sentence, sentence/(2*i)))
sentence = 0
word_num = 0
for i, d in enumerate(data[int(0.8*len(data)):int(0.9*len(data))]):
    sentence += len(d["case_A"][0])
    for item in d["case_A"][0]:
        word_num += len(item)
    sentence += len(d["case_B"][0])
    for item in d["case_B"][0]:
        word_num += len(item)
print("dev_avg_sentence_length {} avg_sentence_num {}".format(word_num/sentence, sentence/(2*i)))

sentence = 0
word_num = 0
for i, d in enumerate(data[int(0.9*len(data)):]):
    sentence += len(d["case_A"][0])
    for item in d["case_A"][0]:
        word_num += len(item)
    sentence += len(d["case_B"][0])
    for item in d["case_B"][0]:
        word_num += len(item)
print("test_avg_sentence_length {} avg_sentence_num {}".format(word_num/sentence, sentence/(2*i)))
sentence = 0
word_num = 0
exp_len = 0
for i, d in enumerate(data):
    sentence += len(d["case_A"][0])
    for item in d["case_A"][0]:
        word_num += len(item)
    sentence += len(d["case_B"][0])
    for item in d["case_B"][0]:
        word_num += len(item)
    exp_len += len(d["explanation"])
print("avg_sentence_length {} avg_sentence_num {} exp_len: {}".format(word_num/sentence, sentence/(2*i), exp_len/i))









