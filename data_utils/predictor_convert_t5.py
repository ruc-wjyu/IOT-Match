"""
最后的predictor函数 仿照 esnli 写一个，后期用对比学习去做一下，这个文件里产生数据用
"""
import sys
sys.path.append("..")
import json
import torch
from utils.snippets import *
from tqdm import tqdm
import argparse
from transformers import MT5ForConditionalGeneration
parser = argparse.ArgumentParser()
parser.add_argument('--maxlen', type=int, default=1024, help='max_len')
parser.add_argument('--match_type', type=str, default="midmatch", help='[match midmatch dismatch]')
parser.add_argument('--cuda_pos', type=str, default="0", help='[0 1]')
parser.add_argument('--data_type', type=str, default="ELAM", help='ELAM, CAIL')
parser.add_argument('--mode_type', type=str, default="wo_token", help='wo_token NILE xLIRE')
args = parser.parse_args()
print(args)
device = torch.device('cuda:'+args.cuda_pos) if torch.cuda.is_available() else torch.device('cpu')

def generate(text, model, tokenizer, device=device, max_length=30):
    feature = tokenizer.encode(text, return_token_type_ids=True, return_tensors='pt',
                               max_length=args.maxlen, truncation=True)
    feature = {'input_ids': feature}
    feature = {k: v.to(device) for k, v in list(feature.items())}

    gen = model.generate(max_length=max_length, eos_token_id=tokenizer.sep_token_id,
                         decoder_start_token_id=tokenizer.cls_token_id,
                         **feature).cpu().numpy()[0]
    gen = gen[1:]
    gen = tokenizer.decode(gen, skip_special_tokens=True).replace(' ', '')
    return gen

def load_checkpoint_p(model, optimizer, trained_epoch, file_name=None):
    save_params = torch.load(file_name, map_location=device)
    model.load_state_dict(save_params["model"])



def convert(file_list, save_path):
    with torch.no_grad():
        tokenizer = T5PegasusTokenizer.from_pretrained(pretrained_t5_fold)
        tokenizer.add_tokens(["[AO]", "[YO]", "[ZO]", '[AI]', "[YI]", "[ZI]"])
        model = MT5ForConditionalGeneration.from_pretrained(pretrained_t5_fold)
        model.resize_token_embeddings(len(tokenizer))
        model = model.to(device)
        load_checkpoint_p(model, None, None,
                        "../models/weights/seq2seq_model/{}-t5-seq2seq-{}-{}.pkl".format(args.match_type, args.data_type,
                                                                                         args.mode_type))

        all_data = []
        for file in file_list:
            with open(file, 'r') as f:
                for line in f:
                    all_data.append(json.loads(line))

        for d in tqdm(all_data, desc=u'评估中'):
            match_exp = generate(d['source_1'], model, tokenizer, device,max_length=args.maxlen//4)
            d["exp"] = [match_exp]

    with open(save_path, 'w') as f:
        for item in all_data:
            f.writelines(json.dumps(item, ensure_ascii=False))
            f.write('\n')

if __name__ == '__main__':
    if args.data_type=='ELAM':
        match_data_seq2seq_json = '../dataset/our_data/match_data_seq2seq_{}.json'.format(args.mode_type)
        midmatch_data_seq2seq_json = '../dataset/our_data/midmatch_data_seq2seq_{}.json'.format(args.mode_type)
        dismatch_data_seq2seq_json = '../dataset/our_data/dismatch_data_seq2seq_{}.json'.format(args.mode_type)
        save_path = "../dataset/our_data/{}_data_prediction_t5_{}.json".format(args.match_type, args.mode_type)
    elif args.data_type=='CAIL':
        match_data_seq2seq_json = '../dataset/match_data_seq2seq_{}.json'.format(args.mode_type)
        midmatch_data_seq2seq_json = '../dataset/midmatch_data_seq2seq_{}.json'.format(args.mode_type)
        dismatch_data_seq2seq_json = '../dataset/dismatch_data_seq2seq_{}.json'.format(args.mode_type)
        save_path = "../dataset/{}_data_prediction_t5_{}.json".format(args.match_type, args.mode_type)
    else:
        exit()
    file_list = [match_data_seq2seq_json, midmatch_data_seq2seq_json, dismatch_data_seq2seq_json]
    convert(file_list, save_path)


    print(u'输出over！')



