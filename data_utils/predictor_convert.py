"""
最后的predictor函数 仿照 esnli 写一个，后期用对比学习去做一下，这个文件里产生数据用
"""
import sys
sys.path.append("..")
import json
from models.seq2seq_model import *
from models.seq2seq_model_dismatch import *
from models.seq2seq_model import GenerateModel as G_m
from models.seq2seq_model_dismatch import GenerateModel as G_d
from models.seq2seq_model import AutoSummary as AS_m
from models.seq2seq_model_dismatch import AutoSummary as AS_d

device = torch.device('cuda:'+'1') if torch.cuda.is_available() else torch.device('cpu')

def load_checkpoint_p(model, optimizer, trained_epoch, file_name=None):
    if file_name==None:
        file_name = args.checkpoint + '/' + f"{args.seq2seq_type}-seq2seq-{trained_epoch}.pkl"
    save_params = torch.load(file_name, map_location=device)
    model.load_state_dict(save_params["model"])



def convert(file_list, save_path):
    with torch.no_grad():
        #match_model = G_m()
        #load_checkpoint_p(match_model, None, 20, "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/models/weights/seq2seq_model/match-seq2seq-8.pkl")
        midmatch_model = G_m()
        load_checkpoint_p(midmatch_model, None, 20,
                        "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/models/weights/seq2seq_model/midmatch-seq2seq-16.pkl")
        #dismatch_model = G_d()
        #load_checkpoint_p(dismatch_model, None, 20, "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/models/weights/seq2seq_model/dismatch_v2-seq2seq-15.pkl")
        all_data = []
        for file in file_list[2:3]:
            with open(file, 'r') as f:
                for line in f:
                    all_data.append(json.loads(line))

        autosummary_midmatch = AS_m(
            start_id=midmatch_model.tokenizer.cls_token_id,
            end_id=midmatch_model.tokenizer.sep_token_id,
            maxlen=args.maxlen // 4,
            model=midmatch_model
        )
        # autosummary_match = AS_m(
        #     start_id=match_model.tokenizer.cls_token_id,
        #     end_id=match_model.tokenizer.sep_token_id,
        #     maxlen=args.maxlen // 4,
        #     model=match_model
        # )
        #
        # autosummary_dismatch = AS_d(
        #     start_id=dismatch_model.tokenizer.cls_token_id,
        #     end_id=dismatch_model.tokenizer.sep_token_id,
        #     maxlen=args.maxlen // 4,
        #     model=dismatch_model
        # )

        for d in tqdm(all_data, desc=u'评估中'):
            # match_exp = autosummary_match.generate(d['source_1'], 1)
            # dismatch_exp = autosummary_dismatch.generate(d['source_1_dis'][0], 1)
            # dismatch_exp += autosummary_dismatch.generate(d['source_1_dis'][1], 1)
            midmatch_exp = autosummary_midmatch.generate(d['source_1'], 5)
            d["exp"].append(midmatch_exp)

    with open(save_path, 'w') as f:
        for item in all_data:
            f.writelines(json.dumps(item, ensure_ascii=False))
            f.write('\n')

if __name__ == '__main__':

    match_data_seq2seq_json = '../dataset/match_data_prediction.json'
    midmatch_data_seq2seq_json = '../dataset/midmatch_data_prediction.json'
    dismatch_data_seq2seq_json = '../dataset/dismatch_data_prediction.json'
    save_path = "../dataset/dismatch_data_prediction_v2.json"
    file_list = [match_data_seq2seq_json, midmatch_data_seq2seq_json, dismatch_data_seq2seq_json]
    convert(file_list, save_path)


    print(u'输出over！')



