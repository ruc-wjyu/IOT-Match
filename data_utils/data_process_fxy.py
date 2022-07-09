import json
all_data = []
with open("/home/zhongxiang_sun/code/explanation_project/explanation_model/dataset/fxy/all_clean.json", "r") as f:
    for line in f:
        item = json.loads(line)
        temp_dic = {"case_A":[item["splited_sentence"]]}
        data_list = []
        aleady_data = []
        for span_key in item["spans"].keys():
            if span_key[5:8] == "001":
                label = 1
            if span_key[5:7] == "02":
                label = 2
            if span_key[5:8] == "003":
                label = 3
            for span in item["spans"][span_key]:
                for sent in range(span[0], span[1]+1):
                    data_list.append([sent, label])
                    aleady_data.append(sent)
            for i in range(len(item["splited_sentence"])):
                if i in aleady_data:
                    pass
                else:
                    data_list.append([i, 0])
        temp_dic["case_A"].append(data_list)
        all_data.append(temp_dic)
with open("/home/zhongxiang_sun/code/explanation_project/explanation_model/dataset/fxy/fxy.json",'w') as f:
    for line in all_data:
        f.writelines(json.dumps(line, ensure_ascii=False))
        f.write('\n')


