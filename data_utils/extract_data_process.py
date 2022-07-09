from utils.snippets import *
import json
data_json = '/new_disk2/zhongxiang_sun/code/law_project/data/CAIL2021/code/persudo_data_new.json'
def load_cail_data(filename, save_path):
    """加载数据
    返回：[(text, summary)]
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            case_a = l['case_A']
            text_a = case_a['content']
            labels_a = [[d['index'], 1] for d in case_a['evidence']]

            case_b = l['case_B']
            text_b = case_b['content']
            labels_b = [[d['index'], 1] for d in case_b['evidence']]
            item = {}
            item["case_A"] = [text_a, labels_a]
            item["case_B"] = [text_b, labels_b]
            item["explanation"] = l["features_exp"]
            item["label"] = l['label']
            item["relation_label"] = l["relation_label"]
            D.append(item)


    with open(save_path, 'w') as f:
        for l in D:
            f.writelines(json.dumps(l,ensure_ascii=False))
            f.write('\n')



def load_our_data(filename, save_path):
    """加载数据
    返回：[(text, summary)]
    """
    D = []
    data = []
    for file in filename:
        with open(file, encoding='utf-8') as f:
            data += json.load(f)

    for l in data:
        relation_label_aqss, relation_label_yjss, relation_label_zyjd = [], [], []
        for it in l["relation"]:
            if it['entityLeft'].split('#')[2] == '案情事实':
                for a in it["label_Left"]:
                    for b in it["label_Right"]:
                        relation_label_aqss.append([a, b])

            if it['entityLeft'].split('#')[2] == '要件事实':
                for a in it["label_Left"]:
                    for b in it["label_Right"]:
                        relation_label_yjss.append([a, b])

            if it['entityLeft'].split('#')[2] == '争议焦点':
                for a in it["label_Left"]:
                    for b in it["label_Right"]:
                        relation_label_zyjd.append([a, b])

        case_a = l['case_a_ner']
        text_a = []
        labels_a = []
        for i, it in enumerate(case_a):
            text_a.append(it["text"])
            if it["entities"] != []:
                if it["entities"][0]["type"] == "aqss":
                    labels_a.append([i, 1])
                elif it["entities"][0]["type"] == "yjss":
                    labels_a.append([i, 2])
                elif it["entities"][0]["type"] == "zyjd":
                    labels_a.append([i, 3])
                else:
                    print(it["entities"])
            else:
                pass

        case_b = l['case_b_ner']
        text_b = []
        labels_b = []
        for i, it in enumerate(case_b):
            text_b.append(it["text"].replace('h', ''))
            if it["entities"] != []:
                if it["entities"][0]["type"] == "aqss":
                    labels_b.append([i, 1])
                elif it["entities"][0]["type"] == "yjss":
                    labels_b.append([i, 2])
                elif it["entities"][0]["type"] == "zyjd":
                    labels_b.append([i, 3])
                else:
                    print(l)
            else:
                pass

        item = {}
        item["case_A"] = [text_a, labels_a]
        item["case_B"] = [text_b, labels_b]
        item["explanation"] = l["explanation"]
        item["label"] = l['gold_label']
        item["relation_label"] = {"relation_label_aqss": relation_label_aqss,
                                  "relation_label_yjss": relation_label_yjss,
                                  "relation_label_zyjd": relation_label_zyjd}
        item['id'] = l['pair_ID']
        D.append(item)


    with open(save_path, 'w') as f:
        for l in D:
            f.writelines(json.dumps(l,ensure_ascii=False))
            f.write('\n')




if __name__ == '__main__':
    # """
    # cail data
    # """
    # data_extract_json = '../dataset/data_extract.json'
    # load_cail_data(data_json, data_extract_json)

    """
    our data
    """
    data_extract_json = '../dataset/our_data/data_extract.json'
    load_our_data([our_data_train,our_data_dev, our_data_test], data_extract_json)


