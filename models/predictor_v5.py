"""
分类任务 + margin loss
"""
import sys
sys.path.append("..")
import datetime
from transformers import BertTokenizer, AutoTokenizer, BertModel, AutoModel
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import copy
from torch.utils.data import Dataset, DataLoader
import logging
from utils.snippets import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_score
# 基本参数
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size_train', type=int, default=3, help='batch size')
parser.add_argument('--batch_size_test', type=int, default=2, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='decay weight of optimizer')
parser.add_argument('--model_name', type=str, default='legal_bert', help='[nezha, legal_bert, lawformer]')
parser.add_argument('--checkpoint', type=str, default="./weights/predict_model", help='checkpoint path')
parser.add_argument('--bert_maxlen', type=int, default=512, help='max length of each case')
parser.add_argument('--maxlen', type=int, default=1024, help='max length of each case')
parser.add_argument('--input_size', type=int, default=768)
parser.add_argument('--hidden_size', type=int, default=384)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--cuda_pos', type=str, default='1', help='which GPU to use')
parser.add_argument('--seed', type=int, default=1, help='max length of each case')
parser.add_argument('--train', type=bool, default=True, help='whether train')
parser.add_argument('--early_stopping_patience', type=int, default=5, help='whether train')
parser.add_argument('--log_name', type=str, default="predictor_v3_2", help='whether train')
parser.add_argument('--margin', type=float, default=0.01, help='margin')
parser.add_argument('--weight', type=float, default=1., help='gold_weight')
parser.add_argument('--gold_margin', type=float, default=0., help='gold_margin')
parser.add_argument('--gold_weight', type=float, default=1., help='gold_weight')
parser.add_argument('--scale_in', type=float, default=10., help='scale_in')
parser.add_argument('--scale_out', type=float, default=10., help='scale_out')
parser.add_argument('--warmup_steps', type=int, default=10000, help='warmup_steps')
parser.add_argument('--accumulate_step', type=int, default=12, help='accumulate_step')
parser.add_argument('--data_type', type=str, default="ELAM", help='[ELAM, CAIL]')
parser.add_argument('--mode_type', type=str, default="rationale", help='[all_sents, wo_rationale, rationale]')
parser.add_argument('--eval_metric', type=str, default="linear_out", help='[linear_out, cosine_out]')

args = parser.parse_args()
print(args)
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
device = torch.device('cuda:'+args.cuda_pos) if torch.cuda.is_available() else torch.device('cpu')
logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
                    filename='../logs/predictor_v5_data_type_{}_mode_type_{}.log'.format(args.data_type, args.mode_type),
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )


if args.data_type == 'CAIL':
    data_predictor_json = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/stage3/data_prediction_{}.json".format(args.mode_type)
elif args.data_type == 'ELAM':
    data_predictor_json = "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/dataset/our_data/stage3/data_prediction_{}.json".format(args.mode_type)
else:
    exit()
def load_data(filename):
    """加载数据
    返回：[{...}]
    """
    all_data = []
    with open(filename) as f:
        for l in f:
            all_data.append(json.loads(l))
    random.shuffle(all_data)
    return all_data


def load_checkpoint(model, optimizer, trained_epoch, file_name=None):
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
    filename = args.checkpoint + '/' + "predictor_v5_data_type_{}_mode_type_{}.log".format(args.data_type, args.mode_type)
    torch.save(save_params, filename)


class PredictorDataset(Dataset):
    """
    input data predictor convert的输出就OK
    """
    def __init__(self, input_data, random=True):
        super(PredictorDataset, self).__init__()
        self.data = input_data
        self.random = random

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        注意exp的为 match dismatch midmatch
        :param index:
        :return:
        """
        i = np.random.choice(2) + 1 if self.random else 1
        if i == 1:
            return self.data[index]['case_a'], self.data[index]['case_b'], self.data[index]['exp'][0], self.data[index]['exp'][1], self.data[index]['exp'][2], self.data[index]['label'], self.data[index]['explanation']
        else:
            return self.data[index]['source_2_dis'][0], self.data[index]['source_2_dis'][1], self.data[index]['exp'][0], self.data[index]['exp'][1],  self.data[index]['exp'][2], self.data[index]['label'], self.data[index]['explanation']



class Collate:
    def __init__(self):
        if args.model_name=='nezha':
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_nezha_fold)
            self.max_seq_len = args.maxlen
        elif args.model_name == 'legal_bert':
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_fold)
            self.max_seq_len = args.bert_maxlen
        elif args.model_name == 'lawformer':
            self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
            self.max_seq_len = args.maxlen

    def __call__(self, batch):
        text_a, text_b, exp_match, exp_dismatch, exp_midmatch, labels, gold_exp = [], [], [], [], [], [], []
        for item in batch:
            text_a.append(item[0])
            text_b.append(item[1])
            exp_match.append(item[2])
            exp_midmatch.append(item[3])
            exp_dismatch.append(item[4])
            labels.append(item[5])
            gold_exp.append(item[6])
        dic_data_a = self.tokenizer.batch_encode_plus(text_a, padding=True, truncation=True,
                                                      max_length=self.max_seq_len, return_tensors='pt')
        dic_data_b = self.tokenizer.batch_encode_plus(text_b, padding=True, truncation=True,
                                                    max_length=self.max_seq_len, return_tensors='pt')
        dic_match = self.tokenizer.batch_encode_plus(exp_match, padding=True, truncation=True,
                                                      max_length=self.max_seq_len, return_tensors='pt')
        dic_dismatch = self.tokenizer.batch_encode_plus(exp_dismatch, padding=True, truncation=True,
                                                     max_length=self.max_seq_len, return_tensors='pt')
        dic_midmatch = self.tokenizer.batch_encode_plus(exp_midmatch, padding=True, truncation=True,
                                                        max_length=self.max_seq_len, return_tensors='pt')
        dic_gold_exp = self.tokenizer.batch_encode_plus(gold_exp, padding=True, truncation=True,
                                                        max_length=self.max_seq_len, return_tensors='pt')
        return dic_data_a, dic_data_b, dic_match, dic_midmatch, dic_dismatch, torch.tensor(labels), dic_gold_exp


def build_pretrain_dataloader(data, batch_size, shuffle=True, num_workers=0, random=True):
    data_generator =PredictorDataset(data, random=random)
    collate = Collate()
    return DataLoader(
        data_generator,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate
    )


class PredictorModel(nn.Module):
    def __init__(self):
        super(PredictorModel, self).__init__()
        if args.model_name=='nezha':
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_nezha_fold)
            self.model = BertModel.from_pretrained(pretrained_nezha_fold)
        elif args.model_name == 'legal_bert':
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_fold)
            self.model = BertModel.from_pretrained(pretrained_bert_fold)
        elif args.model_name == 'lawformer':
            self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
            self.model = AutoModel.from_pretrained("thunlp/Lawformer")

        self.configuration = self.model.config

        self.n = 2
        self.linear1 = nn.Sequential(
            nn.Linear(self.n*self.configuration.hidden_size, self.configuration.hidden_size),  # self.hidden_dim * 2 for bi-GRU & concat AB
            nn.LeakyReLU(),
            )



        self.linear2_match = nn.Sequential(
                                     nn.Linear(self.configuration.hidden_size+self.configuration.hidden_size, self.configuration.hidden_size),
                                     nn.LeakyReLU(),
                                     nn.Linear(self.configuration.hidden_size, 1),
                                     nn.Sigmoid()
                                     )
        self.linear2_midmatch = nn.Sequential(
                                     nn.Linear(self.configuration.hidden_size+self.configuration.hidden_size, self.configuration.hidden_size),
                                     nn.LeakyReLU(),
                                     nn.Linear(self.configuration.hidden_size, 1),
                                     nn.Sigmoid()
                                     )
        self.linear2_dismatch = nn.Sequential(
                                     nn.Linear(self.configuration.hidden_size+self.configuration.hidden_size, self.configuration.hidden_size),
                                     nn.LeakyReLU(),
                                     nn.Linear(self.configuration.hidden_size, 1),
                                     nn.Sigmoid()
                                     )
        self.cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    def forward(self, text_a, text_b, match, midmatch, dismatch, gold_exp, batch_label, model_type='train'):
        output_text_a = self.model(**text_a)['pooler_output']
        output_text_b = self.model(**text_b)['pooler_output']
        output_exp1 = self.model(**match)['pooler_output']
        output_exp2 = self.model(**midmatch)['pooler_output']
        output_exp3 = self.model(**dismatch)['pooler_output']
        gold_exp_pl = self.model(**gold_exp)['pooler_output']
        data_p = torch.cat([output_text_a, output_text_b], dim=-1)

        query = self.linear1(data_p)
        class_match = self.linear2_match(torch.cat([query, output_exp1], dim=-1))
        class_midmatch = self.linear2_midmatch(torch.cat([query, output_exp2], dim=-1))
        class_dismatch = self.linear2_dismatch(torch.cat([query, output_exp3], dim=-1))
        """
        算一个query与三个exp + golden的cos
        """
        exps = torch.stack([output_exp3, output_exp2, output_exp1], dim=1)  # (batch_size, 3, dim) 还是要把dismatch放前面
        query_1 = query.unsqueeze(1).repeat(1, 3, 1)  # (batch_size, 3, dim)
        in_cos_score = self.cos(exps, query_1)
        golden_cos_similarity = self.cos(gold_exp_pl, query)
        """
        样本间对比操作
        query 与 其他数据的exp算得分
        """
        if model_type == 'train':
            select = exps[:, batch_label.squeeze(), :]
            fi_select = select.permute([1, 0, 2])  # (batch_size, batch_size, dim)
            out_cos_score = self.cos(fi_select, query.unsqueeze(-2))
            output_scores = torch.cat((class_dismatch, class_midmatch, class_match), dim=-1)
            return {"exp_score":output_scores, "in_cos_score":in_cos_score, "golden_cos_score":golden_cos_similarity, "out_cos_score":out_cos_score}   # 需要两个mask 一个对角线mask 另一个label mask
        else:
            output_scores = torch.cat((class_dismatch, class_midmatch, class_match), dim=-1)
            return {"exp_score":output_scores, "in_cos_score":in_cos_score}   # 需要两个mask 一个对角线mask 另一个label mask

def in_class_loss(score, summary_score=None, gold_margin=0, gold_weight=1):
    pos_score = summary_score.unsqueeze(-1).expand_as(score)
    neg_score = score
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    ones = torch.ones_like(pos_score)
    loss_func = torch.nn.MarginRankingLoss(gold_margin)
    TotalLoss = gold_weight * loss_func(pos_score, neg_score, ones)
    return TotalLoss


def out_class_loss(score, summary_score=None, margin=0, weight=1):
    select = torch.le(torch.eye(len(summary_score), device=device), 0)
    pos_score = summary_score.unsqueeze(-1).expand_as(score)
    neg_score = score
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    select = select.contiguous().view(-1)
    ones = torch.ones_like(pos_score, device=device)
    loss_func = torch.nn.MarginRankingLoss(margin, reduction='none')
    TotalLoss = weight * torch.sum(loss_func(pos_score, neg_score, ones)*select)
    return TotalLoss


def train_valid(model, train_dataloader, valid_dataloader, test_dataloader):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()
    #criterion = nn.CrossEntropyLoss()
    early_stop = EarlyStop(args.early_stopping_patience)
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        current_step = 0
        model.train()
        pbar = tqdm(train_dataloader, desc="Iteration", postfix='train')
        for batch_data in pbar:
            text_batch_a, text_batch_b, match_batch, midmatch_batch, dismatch_batch, label_batch, gold_exp_batch = batch_data

            text_batch_a, text_batch_b, match_batch, dismatch_batch, midmatch_batch, label_batch, gold_exp_batch =  \
            text_batch_a.to(device), text_batch_b.to(device), match_batch.to(device), dismatch_batch.to(device), midmatch_batch.to(device), label_batch.to(device), gold_exp_batch.to(device)
            scores = model(text_batch_a, text_batch_b, match_batch, midmatch_batch, dismatch_batch,
                           gold_exp_batch, label_batch, model_type='train')

            """
            match midmatch dismatch
            """

            linear_similarity, gold_similarity, in_cos_score, out_cos_score = scores['exp_score'], scores['golden_cos_score'], scores['in_cos_score'], scores['out_cos_score']

            loss_in_class = args.scale_in * in_class_loss(in_cos_score, gold_similarity, args.gold_margin, args.gold_weight)
            loss_out_class = args.scale_out * out_class_loss(out_cos_score, in_cos_score[list(range(len(in_cos_score))), label_batch.squeeze()], args.margin, args.weight)

            bce_labels = torch.zeros_like(scores['exp_score'])
            bce_labels[list(range(len(bce_labels))), label_batch] = 1
            bce_labels = bce_labels.to(device)

            loss_bce = criterion(scores['exp_score'], bce_labels)
            loss = loss_in_class + loss_out_class + loss_bce  # 多任务了属于是
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_item = loss.cpu().detach().item()
            epoch_loss += loss_item
            current_step += 1
            pbar.set_description("train loss {}".format(epoch_loss / current_step))
            if current_step % 100 == 0:
                logging.info("train step {} loss {}".format(current_step, epoch_loss / current_step))

        epoch_loss = epoch_loss / current_step
        time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('{} train epoch {} loss: {:.4f}'.format(time_str, epoch, epoch_loss))
        logging.info('train epoch {} loss: {:.4f}'.format(epoch, epoch_loss))
        model.eval()

        current_val_metric_value = evaluation(valid_dataloader, model, epoch)

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
            print("early stopping at epoch {} since didn't improve from epoch no {}. Best value {}, current value {}".format(
                    epoch, early_stop.best_epoch, early_stop.best_value, current_val_metric_value
                ))
            break
        evaluation(test_dataloader, model, epoch, type='test')

def evaluation(valid_dataloader, model, epoch, type='valid'):
    with torch.no_grad():
        correct = 0
        total = 0
        current_step = 0
        prediction_batch_list, label_batch_list = [], []
        pbar = tqdm(valid_dataloader, desc="Iteration", postfix=type)
        for batch_data in pbar:
            text_batch_a, text_batch_b, match_batch, midmatch_batch, dismatch_batch, label_batch, gold_exp_batch = batch_data
            text_batch_a = text_batch_a.to(device)
            text_batch_b = text_batch_b.to(device)
            match_batch, dismatch_batch, midmatch_batch, gold_exp_batch = match_batch.to(device), dismatch_batch.to(device), midmatch_batch.to(device), gold_exp_batch.to(device)
            label_batch = label_batch.to(device)
            label_batch_list.append(label_batch)
            """
            todo 这里好好看一下注意改造 mid 0 dis 1 match 2 
            """
            output_batch = model(text_batch_a, text_batch_b, match_batch, midmatch_batch, dismatch_batch, gold_exp_batch, label_batch, model_type=type)
            if args.eval_metric == 'linear_out':
                _, predicted_output = torch.max(output_batch["exp_score"], -1)
            elif args.eval_metric == 'cosine_out':
                _, predicted_output = torch.max(output_batch["in_cos_score"], -1)
            else:
                exit()
            label_batch = label_batch.to(device)
            total += len(label_batch)
            prediction_batch_list.append(predicted_output)
            correct += torch.sum(torch.eq(label_batch, predicted_output))
            pbar.set_description("{} acc {}".format(type, correct / total))
            current_step += 1
            if current_step % 100 == 0:
                logging.info('{} epoch {} acc {}/{}={:.4f}'.format(type, epoch, correct, total, correct / total))
        prediction_batch_list = torch.cat(prediction_batch_list, dim=0).cpu().tolist()
        label_batch_list = torch.cat(label_batch_list, dim=0).cpu().tolist()
        accuracy = accuracy_score(label_batch_list, prediction_batch_list)
        precision_macro = precision_score(label_batch_list, prediction_batch_list, average='macro')
        recall_macro = recall_score(label_batch_list, prediction_batch_list, average='macro')
        f1_macro = f1_score(label_batch_list, prediction_batch_list, average='macro')
        precision_micro = precision_score(label_batch_list, prediction_batch_list, average='micro')
        recall_micro = recall_score(label_batch_list, prediction_batch_list, average='micro')
        f1_micro = f1_score(label_batch_list, prediction_batch_list, average='micro')
        cohen_kappa = cohen_kappa_score(label_batch_list, prediction_batch_list)
        hamming = hamming_loss(label_batch_list, prediction_batch_list)
        jaccard_macro = jaccard_score(label_batch_list, prediction_batch_list, average='macro')
        jaccard_micro = jaccard_score(label_batch_list, prediction_batch_list, average='micro')

        time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('{} {}  acc {} precision ma {} mi {} recall  ma {} mi {} f1  ma {} mi {}'.format(time_str, type, accuracy,
                                                                                               precision_macro,
                                                                                               precision_micro,
                                                                                               recall_macro,
                                                                                               recall_micro, f1_macro,
                                                                                               f1_micro))
        print('cohen_kappa {} hamming {} jaccard_macro {} jaccard_micro {}'.format(cohen_kappa, hamming, jaccard_macro,
                                                                                   jaccard_micro))
        logging.info(
            '{} {}  acc {} precision ma {} mi {} recall  ma {} mi {} f1  ma {} mi {}'.format(time_str, type, accuracy,
                                                                                             precision_macro,
                                                                                             precision_micro,
                                                                                             recall_macro, recall_micro,
                                                                                             f1_macro, f1_micro))
        logging.info(
            'cohen_kappa {} hamming {} jaccard_macro {} jaccard_micro {}'.format(cohen_kappa, hamming, jaccard_macro,
                                                                                 jaccard_micro))
        return accuracy

def frozen_model(P_model, unfreeze_layers):
    """
    用于冻结模型
    :param model:
    :param free_layer:
    :return:
    """
    for name, param in P_model.named_parameters():
        print(name, param.size())
    print("*" * 30)
    print('\n')

    for name, param in P_model.named_parameters():
        param.requires_grad = False
        for ele in unfreeze_layers:
            if ele in name:
                param.requires_grad = True
                break
    # 验证一下
    for name, param in P_model.named_parameters():
        if param.requires_grad:
            print(name, param.size())

if __name__ == '__main__':
    data = load_data(data_predictor_json)
    train_data = prediction_data_split(data, 'train', splite_ratio=0.8)
    valid_data = prediction_data_split(data, 'valid', splite_ratio=0.8)
    test_data = prediction_data_split(data, 'test', splite_ratio=0.8)
    train_data_loader = build_pretrain_dataloader(train_data, args.batch_size_train, shuffle=True, random=True)
    valid_data_loader = build_pretrain_dataloader(valid_data, args.batch_size_test, shuffle=False, random=False)
    test_data_loader = build_pretrain_dataloader(test_data, args.batch_size_test, shuffle=False, random=False)
    P_model = PredictorModel()
    if args.train:
        P_model = P_model.to(device)
        train_valid(P_model, train_data_loader, valid_data_loader, test_data_loader)
    else:
        P_model = P_model.to(device)
        load_checkpoint(P_model, None, None,
                        "/new_disk2/zhongxiang_sun/code/explanation_project/explanation_model/models/weights/predict_model/predictor-0.pkl")
        with torch.no_grad():
            P_model.eval()
            evaluation(valid_data_loader, P_model, 0)

