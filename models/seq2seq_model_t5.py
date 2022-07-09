import sys
sys.path.append("..")
import datetime
from transformers import BertTokenizer, AutoTokenizer, GPT2Tokenizer, GPT2Model, T5Tokenizer
import argparse
import torch
from transformers import AdamW
import torch.nn as nn
from tqdm import tqdm
from transformers import MT5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
import logging
from utils.snippets import *

# 基本参数
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--each_test_epoch', type=int, default=1)
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0., help='decay weight of optimizer')
parser.add_argument('--model_name', type=str, default='t5', help='matching model')
parser.add_argument('--checkpoint', type=str, default="./weights/seq2seq_model", help='checkpoint path')
parser.add_argument('--bert_maxlen', type=int, default=512, help='max length of each case')
parser.add_argument('--maxlen', type=int, default=1024, help='max length of each case')
parser.add_argument('--input_size', type=int, default=768)
parser.add_argument('--hidden_size', type=int, default=384)
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--threshold', type=float, default=0.3)
parser.add_argument('--k_sparse', type=int, default=10)
parser.add_argument('--early_stopping_patience', type=int, default=5)
parser.add_argument('--log_name', type=str, default="log_seq2seq")
parser.add_argument('--seq2seq_type', type=str, default='match')
parser.add_argument('--cuda_pos', type=str, default='0', help='which GPU to use')
parser.add_argument('--seed', type=int, default=42, help='max length of each case')
parser.add_argument('--train', action='store_true')
parser.add_argument('--data_type', type=str, default='ELAM', help="data type:[ELAM, CAIL]")
parser.add_argument('--data_format', type=str, default='sort', help="data format")
args = parser.parse_args()
print(args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
device = torch.device('cuda:'+args.cuda_pos) if torch.cuda.is_available() else torch.device('cpu')
log_name = args.log_name
logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
                    filename='../logs/{}-{}-{}-{}.log'.format(log_name, args.model_name, args.seq2seq_type, args.data_type),
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )
logging.info(args)
if args.seq2seq_type == 'match':
    data_seq2seq_json = '../models_v2/data/our_data/match_data_seq2seq_{}.json'.format(args.data_format)
elif args.seq2seq_type == 'midmatch':
    data_seq2seq_json = '../models_v2/data/our_data/midmatch_data_seq2seq_{}.json'.format(args.data_format)
else:
    data_seq2seq_json = '../models_v2/data/our_data/dismatch_data_seq2seq_{}.json'.format(args.data_format)


def load_data(filename):
    """加载数据
    返回：[{...}]
    """
    D = []
    with open(filename) as f:
        for l in f:
            D.append(json.loads(l))
    return D


class DataGenerator(Dataset):
    def __init__(self, input_data, random=True):
        super(DataGenerator, self).__init__()
        self.input_data = input_data
        self.random = random

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):

        i = np.random.choice(2) + 1 if self.random else 1
        source, target = self.input_data[idx]['source_%s' % i], self.input_data[idx]['explanation']
        return [source, target]


class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_seq_len = args.maxlen

    def __call__(self, batch):
        source_batch = []
        target_batch = []
        for item in batch:
            source_batch.append(item[0])
            target_batch.append(item[1])

        enc_source_batch = self.tokenizer(source_batch, max_length=self.max_seq_len, truncation=True, padding=True, return_tensors='pt')
        source_ids = enc_source_batch["input_ids"]
        source_attention_mask = enc_source_batch["attention_mask"]
        enc_target_batch = self.tokenizer(target_batch, max_length=self.max_seq_len, truncation=True, padding=True, return_tensors='pt')
        target_ids = enc_target_batch["input_ids"]
        target_attention_mask = enc_target_batch["attention_mask"]

        features = {'input_ids': source_ids, 'decoder_input_ids': target_ids, 'attention_mask': source_attention_mask,
                    'decoder_attention_mask': target_attention_mask}

        return features


def build_pretrain_dataloader(data, batch_size, tokenizer, shuffle=True, num_workers=0):
    data_generator =DataGenerator(data, random=True)
    collate = Collate(tokenizer)
    return DataLoader(
        data_generator,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate
    )




def load_checkpoint(model, optimizer, trained_epoch, file_name=None):
    if file_name==None:
        file_name = args.checkpoint + '/' + f"{args.seq2seq_type}-seq2seq-{trained_epoch}.pkl"
    save_params = torch.load(file_name, map_location=device)
    model.load_state_dict(save_params["model"])
    #optimizer.load_state_dict(save_params["optimizer"])


def save_checkpoint(model, optimizer, trained_epoch):
    save_params = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch,
    }
    if not os.path.exists(args.checkpoint):
        # 判断文件夹是否存在，不存在则创建文件夹
        os.mkdir(args.checkpoint)
    filename = args.checkpoint + '/' + f"{args.seq2seq_type}-{args.model_name}-seq2seq-{args.data_type}-{args.data_format}.pkl"
    torch.save(save_params, filename)


def train_valid(train_data, valid_data, test_data, model, tokenizer):
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # ema = EMA(model, 0.9999)
    # ema.register()
    early_stop = EarlyStop(args.early_stopping_patience)
    for epoch in range(args.epochs):
        epoch_loss = 0.
        current_step = 0
        model.train()
        # for batch_data in tqdm(train_data_loader, ncols=0):
        pbar = tqdm(train_data, desc="Iteration", postfix='train')
        for batch_data in pbar:
            cur = {k: v.to(device) for k, v in batch_data.items()}
            prob = model(**cur)[0]
            mask = cur['decoder_attention_mask'][:, 1:].reshape(-1).bool()
            prob = prob[:, :-1]
            prob = prob.reshape((-1, prob.size(-1)))[mask]
            labels = cur['decoder_input_ids'][:, 1:].reshape(-1)[mask]
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(prob, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # ema.update()
            loss_item = loss.cpu().detach().item()
            epoch_loss += loss_item/labels.shape[-1]
            current_step += 1
            pbar.set_description("train loss {}".format(epoch_loss / current_step))
            if current_step % 100 == 0:
                logging.info("train step {} loss {}".format(current_step, epoch_loss / current_step))

        epoch_loss = epoch_loss / current_step
        time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('{} train epoch {} loss: {:.4f}'.format(time_str, epoch, epoch_loss))
        logging.info('train epoch {} loss: {:.4f}'.format(epoch, epoch_loss))
        # todo 看一下 EMA是否会让模型准确率提升，如果可以的话在保存模型前加入 ema
        with torch.no_grad():
            model.eval()
            current_val_metric_value = evaluate(valid_data, model, tokenizer, type='valid')['main']
            is_save = early_stop.step(current_val_metric_value, epoch)
            if is_save:
                save_checkpoint(model, optimizer, epoch)
            else:
                pass
            if early_stop.stop_training(epoch):
                logging.info(
                    "early stopping at epoch {} since didn't improve from epoch no {}. Best value {}, current value {}".format(
                        epoch, early_stop.best_epoch, early_stop.best_value, current_val_metric_value
                    ))
                print(
                    "early stopping at epoch {} since didn't improve from epoch no {}. Best value {}, current value {}".format(
                        epoch, early_stop.best_epoch, early_stop.best_value, current_val_metric_value
                    ))
                break
            evaluate(test_data, model, tokenizer, type='test')


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


def evaluate(data, model, tokenizer, filename=None, type='valid'):
    """验证集评估
    """
    if filename is not None:
        F = open(filename, 'w', encoding='utf-8')
    total_metrics = {k: 0.0 for k in metric_keys}
    for d in tqdm(data, desc=u'评估中'):
        pred_summary = generate(d['source_1'], model, tokenizer, max_length=args.maxlen//4)
        metrics = compute_metrics(pred_summary, d['explanation'])
        for k, v in metrics.items():
            total_metrics[k] += v
        if filename is not None:
            F.write(d['explanation'] + '\t' + pred_summary + '\n')
            F.flush()
    if filename is not None:
        F.close()
    print(total_metrics)
    logging.info("~~~~~~~~{}~~~~~~~~~~~".format(type))
    for k, v in total_metrics.items():
        logging.info(k+": {} ".format(v/len(data)))
    logging.info("~~~~~~~~{}~~~~~~~~~~~".format(type))
    return {k: v / len(data) for k, v in total_metrics.items()}

if __name__ == '__main__':
    # 加载数据
    tokenizer = T5PegasusTokenizer.from_pretrained(pretrained_t5_fold)
    if args.data_type == 'CAIL':
        tokenizer.add_tokens(["[O]", '[I]'])
    elif args.data_type == 'ELAM':
        tokenizer.add_tokens(["[AO]", "[YO]", "[ZO]", '[AI]', "[YI]", "[ZI]"])
    else:
        print("data type error")
        exit(-1)
    data = load_data(data_seq2seq_json)
    train_data = data_split(data, 'train')
    valid_data = data_split(data, 'valid')
    test_data = data_split(data, 'test')
    train_data_loader = build_pretrain_dataloader(train_data, args.batch_size, tokenizer)
    G_model = MT5ForConditionalGeneration.from_pretrained(pretrained_t5_fold)
    G_model.resize_token_embeddings(len(tokenizer))  # 扩充embedding
    if args.train:
        G_model = G_model.to(device)
        train_valid(train_data_loader, valid_data, test_data, G_model, tokenizer)
    else:
        load_checkpoint(G_model, None, None, "/home/zhongxiang_sun/code/explanation_project/explanation_model/models/weights/seq2seq_model/match-t5-seq2seq-ELAM-sort.pkl")
        G_model = G_model.to(device)
        with torch.no_grad():
            G_model.eval()
            evaluate(valid_data, G_model, tokenizer)





