import json
import re
"""
首先按照句号分句，之后对句号中的每个特殊符号进行切分
\001 案情事实 \002 案情事实匹配特征 \003 要件事实 \004要件事实匹配特征 \005争议焦点 \006争议焦点匹配特征
"""

input_path = "/new_disk2/zhongxiang_sun/code/law_project/data/tagger_result/all_data.json"
input_path_raw = "/new_disk2/zhongxiang_sun/code/law_project/data/tagger_result/all_data_raw.json"

output_path = "../data/tagger_ner.json"

count_ky = 0
is_continue = False
former_type = ""
def split_chinese_sentence(sentence, delete_notag_para) -> list:
    """
    delete_notag_para: 是否把没有特征的段落删去（可能是没剔除掉的无用打断）
    """
    if delete_notag_para:
        para = ""
        para_list_raw = sentence.split('\n')
        tag_list = ["\\001", "\\002", "\\003", "\\004", "\\005", "\\006"]
        for pa in para_list_raw:
            for tag in tag_list:
                if tag in pa:
                    para += pa
                    break
    else:
        para = sentence.replace('\n','h')
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('([；])([^”’])', r"\1\n\2", para)  # 司法案例加入中文分号，冒号暂时不加

    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可
    return para.split("\n")

def sentence2ner(case_A_marked):
    count_signal = 0
    marked = False
    start = 0
    last_type = ""
    j = 0
    entities = []
    global is_continue
    while j < len(case_A_marked):
        if case_A_marked[j:j + 4] == '\\001':  # todo 判断一些001咋相等
            if marked == False and not (is_continue==True and start==0):  # 说明是普通句子不加粗不加颜色写进去  这里不可能存在标记不对应的情况
                sentence = case_A_marked[start:j]
                start = j + 4
                j += 4
                marked = True
                last_type = "aqss"
                count_signal += 4
            else:
                sentence = case_A_marked[start:j]
                if start < j:
                    entities.append({"start_idx": start-count_signal, "end_idx": j-1-count_signal, "type": "aqss", "entity":sentence})
                count_signal += 4
                start = j + 4
                j += 4
                marked = False

        elif case_A_marked[j:j + 4] == '\\002':
            if marked == False and not (is_continue==True and start==0):
                sentence = case_A_marked[start:j]
                start = j + 4
                j += 4
                marked = True
                last_type = "aqss"
                count_signal += 4

            else:  # 说明是非关联案情事实不加粗加红写进去
                sentence = case_A_marked[start:j]
                if start < j:
                    entities.append({"start_idx": start-count_signal, "end_idx": j-1-count_signal, "type": "aqss", "entity":sentence})
                count_signal += 4

                start = j + 4
                j += 4
                marked = False

        elif case_A_marked[j:j + 4] == '\\003':
            if marked == False and not (is_continue==True and start==0):
                sentence = case_A_marked[start:j]
                start = j + 4
                j += 4
                marked = True
                last_type = "yjss"
                count_signal += 4


            else:  # 说明是非关联案情事实不加粗加红写进去
                sentence = case_A_marked[start:j]
                if start < j:
                    entities.append({"start_idx": start-count_signal, "end_idx": j-1-count_signal, "type": "yjss", "entity":sentence})
                count_signal += 4
                start = j + 4
                j += 4
                marked = False

        elif case_A_marked[j:j + 4] == '\\004':
            if marked == False and not (is_continue==True and start==0):
                sentence = case_A_marked[start:j]
                start = j + 4
                j += 4
                marked = True
                last_type = "yjss"
                count_signal += 4

            else:  # 说明是非关联案情事实不加粗加红写进去
                sentence = case_A_marked[start:j]
                if start < j:
                    entities.append({"start_idx": start-count_signal, "end_idx": j-1-count_signal, "type": "yjss", "entity":sentence})

                start = j + 4
                j += 4
                marked = False
                count_signal += 4

        elif case_A_marked[j:j + 4] == '\\005':
            if marked == False and not (is_continue==True and start==0):
                sentence = case_A_marked[start:j]
                start = j + 4
                j += 4
                marked = True
                last_type = "zyjd"
                count_signal += 4

            else:  # 说明是非关联案情事实不加粗加红写进去
                sentence = case_A_marked[start:j]
                if start < j:
                    entities.append({"start_idx": start-count_signal, "end_idx": j-1-count_signal, "type": "zyjd", "entity":sentence})

                start = j + 4
                j += 4
                marked = False
                count_signal += 4

        elif case_A_marked[j:j + 4] == '\\006':
            if marked == False and not (is_continue==True and start==0):
                sentence = case_A_marked[start:j]
                start = j + 4
                j += 4
                marked = True
                last_type = "zyjd"
                count_signal += 4

            else:  # 说明是非关联案情事实不加粗加红写进去
                sentence = case_A_marked[start:j]
                if start < j:
                    entities.append({"start_idx": start-count_signal, "end_idx": j-1-count_signal, "type": "zyjd", "entity":sentence})

                start = j + 4
                j += 4
                marked = False
                count_signal += 4

        else:
            j += 1
    global former_type
    if last_type == '':
        pass
    else:
        former_type = last_type
    if marked == True:
        """跨越句号了"""
        print("跨越句号")
        global count_ky
        count_ky += 1
        is_continue = True
        if start < j:
            entities.append({"start_idx": start - count_signal, "end_idx": j - 1 - count_signal, "type": last_type, "entity": case_A_marked[start:j]})
    elif start == 0 and is_continue:
        entities.append({"start_idx": start - count_signal, "end_idx": j - 1 - count_signal, "type": former_type,
                         "entity": case_A_marked[start:j]})
    else:
        is_continue = False
    case_A_marked = case_A_marked.replace('\\001', '').replace('\\002', '').replace('\\003', '').replace('\\004',
                    '').replace('\\005', '').replace('\\006', '')

    return {"text":case_A_marked, "entities": entities}
def get_data(data=None):
    final_data = []
    if data==None:
        with open(input_path, 'r') as f:
            data = json.load(f)
    else:
        pass
    for item in data:
        # if item["pair_ID"] == "848dbfd5-4263-ab70-e105-e8d9586a15b5|981a181f-5269-3604-25ec-b1eb7a75b8ac":
        #     continue
        case_a_marked = item["case_A_marked"]
        case_b_marked = item["case_B_marked"]
        case_a_marked_list = split_chinese_sentence(case_a_marked[0]["content"], delete_notag_para=False) + split_chinese_sentence(case_a_marked[1]["content"], delete_notag_para=False)
        case_b_marked_list = split_chinese_sentence(case_b_marked[0]["content"], delete_notag_para=False) + split_chinese_sentence(case_b_marked[1]["content"], delete_notag_para=False)
        for i in range(len(case_a_marked_list)):
            case_a_marked_list[i] = sentence2ner(case_a_marked_list[i])
        for i in range(len(case_b_marked_list)):
            case_b_marked_list[i] = sentence2ner(case_b_marked_list[i])
        item["case_a_ner"] = case_a_marked_list
        item["case_b_ner"] = case_b_marked_list
        final_data.append(item)
    return final_data

def insert_relation():
    with open(input_path, 'r') as f:
        data = json.load(f)

    with open(input_path_raw, 'r') as f:
        data_raw = json.load(f)
    """
    先把raw 里的 relation 拼接到data上
    """
    for item in data:
        for raw in data_raw:
            if item['pair_ID'] == raw['id']:
                item['relation'] = raw['relationships']
    for item in data:
        for relation in item['relation']:
            case_a_span = [int(relation['entityLeft'].split('#')[0]), int(relation['entityLeft'].split('#')[1])]
            case_b_span = [int(relation['entityRight'].split('#')[0]), int(relation['entityRight'].split('#')[1])]
            sentence_a = split_chinese_sentence(
                "hhhhh" + item['case_A'][0]['content'], delete_notag_para=False) + split_chinese_sentence('hhhhhh' + item['case_A'][1]['content'], delete_notag_para=False)

            sentence_b = split_chinese_sentence(
                "hhhhh" + item['case_B'][0]['content'], delete_notag_para=False) + split_chinese_sentence('hhhhhh' + item['case_B'][1]['content'], delete_notag_para=False)
            if item['pair_ID'] == '5d196c7b-b198-bc69-54e7-e54741bf65d0|39ff52de-b2c4-6b75-cd75-0ab966872c1e':
                print()
            selected_sentence_a = get_relation_label(sentence_a, case_a_span)
            selected_sentence_b = get_relation_label(sentence_b, case_b_span)
            relation['label_Left'] = selected_sentence_a
            relation['label_Right'] = selected_sentence_b
    return data

def get_relation_label(sentence, span1):
    selected_sentence = []
    count_all = 0
    for i, s in enumerate(sentence):
        count_all += len(s)
        if count_all > span1[0]:
            selected_sentence.append(i)
            for j in range(i + 1, len(sentence)):
                count_all += len(sentence[j])
                if count_all < span1[1]:
                    selected_sentence.append(j)
                else:
                    break
            break
    return selected_sentence


if __name__ == '__main__':
    add_relation = insert_relation()
    ner_data = get_data(add_relation)
    train_data = ner_data[:int(len(ner_data)*0.8)]
    test_data = ner_data[int(len(ner_data)*0.8):int(len(ner_data)*0.9)]
    dev_data = ner_data[int(len(ner_data)*0.9):]
    with open("../data/law_train.json", 'w') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    with open("../data/law_test.json", 'w') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)
    with open("../data/law_dev.json", 'w') as f:
        json.dump(dev_data, f, ensure_ascii=False, indent=4)
    print(count_ky)











