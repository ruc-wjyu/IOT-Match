import datetime
import sys
sys.path.append("..")
import argparse
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from utils.snippets import *
from utils.sinkhorn import *
import logging
import pandas as pd
from termcolor import colored
import ot
from transformers import AdamW
import nni
from nni.utils import merge_parameter
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epoch_num', type=int, default=40, help='number of epochs')
parser.add_argument('--each_test_epoch', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0., help='decay weight of optimizer')
parser.add_argument('--model_name', type=str, default='bert', help='matching model')
parser.add_argument('--checkpoint', type=str, default="./weights/extract/", help='checkpoint path')
parser.add_argument('--max_length', type=int, default=512, help='max length of each case')
parser.add_argument('--input_size', type=int, default=768)
parser.add_argument('--hidden_size', type=int, default=384)
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--threshold', type=float, default=0.3)
parser.add_argument('--cuda_pos', type=str, default='1', help='which GPU to use')
parser.add_argument('--max_seq_len', type=int, default=115, help='number of max seq length')
parser.add_argument('--coefficient', type=float, default=1., help='balance the dgcnn loss and the ot loss')
parser.add_argument('--coefficient_lrable', type=int, default=0, help='whether to use learnable')  # [true / false]
parser.add_argument('--seed', type=int, default=42, help='which seed to use')
parser.add_argument('--early_stopping_patience', type=int, default=5, help='which seed to use')
parser.add_argument('--threshold_ot', type=float, default=1.9, help='threshold ot')
parser.add_argument('--f_mass', type=float, default=3, help='f mass')
parser.add_argument('--weight', type=float, default=100, help='weight')
parser.add_argument('--convert_to_onehot', type=int, default=1,  help='convert to one hot')  # [0, 1]
parser.add_argument('--ot_mode', type=str, default='max',  help='ot mode')  # [max, gumbel]
parser.add_argument('--reg', type=float, default=0.3, help='regular')
parser.add_argument('--model_usage', type=str, default='eval', help='[train eval search]')
parser.add_argument('--four_class', type=int, default=1, help='if use ot loss')  # [0, 1]
parser.add_argument('--cost_cal_way', type=str, default='class_embedding', help='[l2, attention, class_embedding]')  # [l2, attention, class_embedding]
parser.add_argument('--loss_only_label1', type=int, default=0,  help='if only optimizer 1')  # [0, 1]
parser.add_argument('--criterion', type=str, default="BCEFocal",  help='criterions:[BCEFocal, MSE, BCE]')
parser.add_argument('--simot', type=int, default=0,  help='simot 0 false 1 true')
parser.add_argument('--simpercent', type=float, default=0,  help='percent_label')

args = parser.parse_args()
tuner_params = nni.get_next_parameter()
logging.info(tuner_params)
params = vars(merge_parameter(args, tuner_params))
args = Namespace(**params)
print(args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

log_name = "selector_logs"
logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
                    filename='../logs/{}.log'.format(log_name),
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )

logging.info(args)

# 配置信息

data_extract_json = '../dataset/data_extract.json'
data_extract_npy = '../dataset/data_extract.npy'

device = torch.device('cuda:'+args.cuda_pos) if torch.cuda.is_available() else torch.device('cpu')


def load_checkpoint(model, optimizer, trained_epoch, file_name=None):
    if file_name==None:
        file_name = args.checkpoint + '/' + f"extract-{trained_epoch}.pkl"
    save_params = torch.load(file_name, map_location=device)
    model.load_state_dict(save_params["model"])
    #optimizer.load_state_dict(save_params["optimizer"])

def save_checkpoint(model,optimizer, trained_epoch, ot_model=None):
    save_params = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch,
    }
    if not os.path.exists(args.checkpoint):
        # 判断文件夹是否存在，不存在则创建文件夹
        os.mkdir(args.checkpoint)
    filename = args.checkpoint + '/' + "cail_extract-criterion-{}-onehot-{}-ot_mode-{}-convert_to_onehot-{}-weight-{}-simot-{}-simpercent-{}.pkl".format(args.criterion, args.convert_to_onehot, args.ot_mode, args.convert_to_onehot,
                                                                                                 args.weight, args.simot, args.simpercent)
    torch.save(save_params, filename)

    if ot_model!=None:
        save_params_ot = {
            "model": ot_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "trained_epoch": trained_epoch,
        }
        filename = args.checkpoint + '/' + "cail_extract_ot-criterion-{}-onehot-{}-ot_mode-{}-convert_to_onehot-{}-weight-{}-simot-{}-simpercent-{}.pkl".format(args.criterion, args.convert_to_onehot, args.ot_mode, args.convert_to_onehot,
                                                                                                 args.weight, args.simot, args.simpercent)
        torch.save(save_params_ot, filename)



def load_data(filename):
    """加载数据
    返回：[(texts, labels, exp)]
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            D.append(json.loads(l))
    return D


class ResidualGatedConv1D(nn.Module):
    """门控卷积
    """
    def __init__(self, filters, kernel_size, dilation_rate=1):
        super(ResidualGatedConv1D, self).__init__()
        self.filters = filters  # 输出维度
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.supports_masking = True
        self.padding = self.dilation_rate*(self.kernel_size - 1)//2
        self.conv1d = nn.Conv1d(filters, 2*filters, self.kernel_size, padding=self.padding, dilation=self.dilation_rate)
        self.layernorm = nn.LayerNorm(self.filters)
        self.alpha = nn.Parameter(torch.zeros(1))


    def forward(self, inputs):
        input_cov1d = inputs.permute([0, 2, 1])
        outputs = self.conv1d(input_cov1d)
        outputs = outputs.permute([0, 2, 1])
        gate = torch.sigmoid(outputs[..., self.filters:])
        outputs = outputs[..., :self.filters] * gate
        outputs = self.layernorm(outputs)

        if hasattr(self, 'dense'):
            inputs = self.dense(inputs)

        return inputs + self.alpha * outputs


class Selector2_mul_class(nn.Module):
    def __init__(self, input_size, filters, kernel_size, dilation_rate):
        """
        :param feature_size:每个词向量的长度
        """
        super(Selector2_mul_class, self).__init__()
        self.dense1 = nn.Linear(input_size, filters, bias=False)
        self.ResidualGatedConv1D_1 = ResidualGatedConv1D(filters, kernel_size, dilation_rate=dilation_rate[0])
        self.ResidualGatedConv1D_2 = ResidualGatedConv1D(filters, kernel_size, dilation_rate=dilation_rate[1])
        self.ResidualGatedConv1D_3 = ResidualGatedConv1D(filters, kernel_size, dilation_rate=dilation_rate[2])
        self.ResidualGatedConv1D_4 = ResidualGatedConv1D(filters, kernel_size, dilation_rate=dilation_rate[3])
        self.ResidualGatedConv1D_5 = ResidualGatedConv1D(filters, kernel_size, dilation_rate=dilation_rate[4])
        self.ResidualGatedConv1D_6 = ResidualGatedConv1D(filters, kernel_size, dilation_rate=dilation_rate[5])
        self.dense2 = nn.Linear(filters, 2)

    def forward(self, inputs):
        mask = inputs.ge(0.00001)
        mask = torch.sum(mask, axis=-1).bool()
        x1 = self.dense1(nn.Dropout(0.1)(inputs))
        x2 = self.ResidualGatedConv1D_1(nn.Dropout(0.1)(x1))
        x3 = self.ResidualGatedConv1D_2(nn.Dropout(0.1)(x2))
        x4 = self.ResidualGatedConv1D_3(nn.Dropout(0.1)(x3))
        x5 = self.ResidualGatedConv1D_4(nn.Dropout(0.1)(x4))
        x6 = self.ResidualGatedConv1D_5(nn.Dropout(0.1)(x5))
        x7 = self.ResidualGatedConv1D_6(nn.Dropout(0.1)(x6))
        output = self.dense2(nn.Dropout(0.1)(x7))
        return output, mask


class OT(nn.Module):
    def __init__(self, reg=0.5, mass=0.5, numItermax=10):
        super(OT, self).__init__()
        self.reg = reg
        self.mass = mass
        self.numItermax = numItermax
        if args.criterion == 'MSE':
            self.criterion = nn.MSELoss(reduction='none')
        elif args.criterion == 'BCEFocal':
            self.criterion = BCEFocalLoss(reduction='none')
        elif args.criterion == 'BCE':
            self.criterion = nn.BCELoss(reduction='none')
        else:
            print("cant recognize criterion use mse no!")
            self.criterion = nn.MSELoss(reduction='none')
        self.layer_norm = nn.LayerNorm(args.input_size)
        self.embedding_transformer = nn.Sequential(nn.Linear(args.input_size, args.input_size),
                                                   nn.LeakyReLU(),
                                                   nn.LayerNorm(args.input_size),
                                                   nn.Linear(args.input_size, args.input_size),
                                                   nn.LeakyReLU(),
                                                   nn.LayerNorm(args.input_size)
                                                   )

    def _sample_gumbel(self, shape, device, eps=1e-20):
        U = torch.rand(shape).to(device)
        return -torch.log(-torch.log(U + eps) + eps)

    def _gumbel_softmax_sample(self, logits, temperature):
        y = logits + self._sample_gumbel(logits.size(), logits.device)
        return F.softmax(y / temperature, dim=-1)

    def convert_one_hot(self, logits, temperature=1., hard=False, mode='max', model_type='train'):
        """

        :param logits:
        :param temperature:
        :param hard: if true, the output vector will be 0-1
        :param mode: "max":取最大值，“gumbel ” 使用gumbel可以保证每次优化的不同，不一定优化最大值 todo
        :return:
        """
        if mode == 'gumbel':
            y = self._gumbel_softmax_sample(logits, temperature)
            if not hard:
                return y
        elif mode == 'max':
            y = F.softmax(logits, dim=-1)
        else:
            print("mode type can only in max or gumbel")
            exit()


        shape = y.size()
        _, ind = y.max(dim=-1)
        zero_class_mask = torch.ones(y.shape, device=device)
        zero_class_mask[ind == 0] = 0  # 0 的话也mask掉
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y

        if args.four_class:
            if model_type=='train':
                return y_hard
            else:
                y_hard *= zero_class_mask
                return y_hard
        else:
            y_hard = (y_hard - y).detach() + y
            if model_type=='train':
                return y_hard
            else:
                y_hard *= zero_class_mask
                return y_hard

    def forward(self, case_A_logits, case_B_logits, case_A, case_B, relation_matrix, padding_mask_a, padding_mask_b, convert_to_onehot=True, convert_mode='max', model_type="train"):

        '''
        :param case_A:  shape [bs, num_seq, 4], 4 means logits
        :param case_B:
        :param convert_to_onehot 是否用one hot形式表示mask，还是用softmax形式
        :return:
        '''
        num_seq = case_B_logits.shape[1]  # todo 试验一下 可能没必要转化成one hot呢
        if convert_to_onehot:
            case_A_onehot = self.convert_one_hot(case_A_logits, hard=True, mode=convert_mode, model_type=model_type)
            case_B_onehot = self.convert_one_hot(case_B_logits, hard=True, mode=convert_mode, model_type=model_type)
        else:
            case_A_onehot = F.softmax(case_A_logits, dim=-1)
            case_B_onehot = F.softmax(case_B_logits, dim=-1)


        """
        拼接上原始句子，之后进行OT计算 
        """
        """
        构造一个新的矩阵，可以将0类别全部搞成0
        """
        class_mask = torch.bmm(case_A_onehot, case_B_onehot.permute([0, 2, 1]))
        padding_mask_a, padding_mask_b = padding_mask_a.unsqueeze(-1), padding_mask_b.unsqueeze(1)
        padding_mask = torch.bmm(padding_mask_a.to(float), padding_mask_b.to(float))
        padding_mask_value = -1e9*padding_mask + 1e9    # 1e9 跟bert一样 算loss的时候将这部分梯度截断
        case_A = self.embedding_transformer(case_A)
        case_B = self.embedding_transformer(case_B)
        cost_orig = torch.cdist(case_A, case_B)
        if not convert_to_onehot:
            # 想要让对应位置plan变大，就要让W矩阵对应位置减小，那么就要让DGCNN的输出概率增大， 最大可以大到使得W（cost）变为 0
            class_mask_value = -(
                        args.weight + cost_orig) * class_mask + args.weight  # cdist2 最大根号下786（bert输出维度） 算loss的时候将这部分梯度留下
        else:
            class_mask_value = -args.weight * class_mask + args.weight  # cdist2 最大根号下786（bert输出维度） 算loss的时候将这部分梯度留下

        if model_type=='train':
            cost_matrix_class = cost_orig + padding_mask_value  # 即使两个padding都有的位置也是存在的，合理的
        else:
            cost_matrix_class = cost_orig + class_mask_value + padding_mask_value  # 即使两个padding都有的位置也是存在的，合理的

        margin_A = (torch.ones(num_seq) / num_seq).to(device)
        margin_B = (torch.ones(num_seq) / num_seq).to(device)

        plan_list = []

        for i in range(case_A.shape[0]):
            seq_len_a = torch.sum(padding_mask_a[i])
            seq_len_b = torch.sum(padding_mask_b[i])
            self.mass = 1/num_seq*min(seq_len_a, seq_len_b)/args.f_mass   # mass 动态调整 <=0.5
            plan = sinkhorn_knopp(margin_A, margin_B, cost_matrix_class[i], reg=self.reg, maxIter=self.numItermax) # , m=self.mass
            plan_list.append(plan)
        return_plan_class = torch.stack(plan_list)

        in_class_loss = torch.mean(cost_matrix_class*class_mask*padding_mask)
        out_class_loss = -torch.mean(cost_matrix_class*(1 - class_mask)*padding_mask)
        if args.simot:
            """
            通过sim_mask来mask一定数量的plan-relation loss
            """
            sim_mask = torch.le(torch.rand([return_plan_class.shape[0], 1, 1], device=device), args.simpercent)
            loss_raw = self.criterion(return_plan_class*num_seq, relation_matrix)*sim_mask
            sup_loss = torch.div(torch.sum(torch.multiply(loss_raw, padding_mask_a)), torch.sum(padding_mask_a))
            loss = in_class_loss + out_class_loss + 10*sup_loss
        else:
            loss = in_class_loss + out_class_loss

        if model_type == 'train':
            return loss
        else:
            return return_plan_class


class Selector_Dataset(Dataset):
    def __init__(self, data_x, data_y):
        super(Selector_Dataset, self).__init__()
        self.data_x_tensor = torch.from_numpy(data_x)
        self.data_y_tensor = torch.from_numpy(data_y)
    def __len__(self):
        return len(self.data_x_tensor)
    def __getitem__(self, idx):
        return self.data_x_tensor[idx], self.data_y_tensor[idx]


class Selector_Dataset_Pair(Dataset):
    def __init__(self, data_x_A, data_x_B, data_y_A, data_y_B,  data_y_seven_class_A, data_y_seven_class_B, data_relation_matrix):
        super(Selector_Dataset_Pair, self).__init__()
        self.data_x_tensor_A = torch.from_numpy(data_x_A)
        self.data_x_tensor_B = torch.from_numpy(data_x_B)
        self.data_y_tensor_A = torch.from_numpy(data_y_A)
        self.data_y_tensor_B = torch.from_numpy(data_y_B)
        self.data_y_seven_class_A = torch.from_numpy(data_y_seven_class_A)
        self.data_y_seven_class_B = torch.from_numpy(data_y_seven_class_B)
        self.data_relation_matrix = torch.from_numpy(data_relation_matrix)

    def __len__(self):
        return len(self.data_x_tensor_A)
    def __getitem__(self, idx):
        return self.data_x_tensor_A[idx], self.data_x_tensor_B[idx], self.data_y_tensor_A[idx] ,\
               self.data_y_tensor_B[idx], self.data_y_seven_class_A[idx], self.data_y_seven_class_B[idx], self.data_relation_matrix[idx]


def train_pair(args, model, OT_model, train_dataloader, valid_dataloader, test_dataloader):
    model = model.to(device)
    early_stop = EarlyStop(args.early_stopping_patience)
    no_decay = ['bias', 'gamma', 'beta']
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    if args.coefficient_lrable:
        theta1 = nn.Parameter(torch.tensor(1.))
        theta2 = nn.Parameter(torch.tensor(0.01))
        optimizer_grouped_parameters.append({'params': [theta1, theta2], 'weight_decay_rate': 0.0})

    else:
        pass

    optimizer_grouped_parameters.append({'params': [p for n, p in list(OT_model.named_parameters())], 'weight_decay_rate': 0.0})
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction='none')

    for epoch in range(args.epoch_num):
        epoch_loss, epoch_loss_ot, epoch_loss_dgcnn = 0., 0., 0.
        current_step = 0
        model.train()
        pbar = tqdm(train_dataloader, desc="Iteration", postfix='train', ncols=200)
        for batch_data in pbar:
            x_batch_A, x_batch_B, label_batch_A, label_batch_B, seven_label_batch_A, seven_label_batch_B, relation_matrix_batch = batch_data
            x_batch_A = x_batch_A.to(device)
            x_batch_B = x_batch_B.to(device)
            label_batch_A = label_batch_A.to(device)
            label_batch_B = label_batch_B.to(device)
            seven_label_batch_A = seven_label_batch_A.to(device)
            seven_label_batch_B = seven_label_batch_B.to(device)
            relation_matrix_batch = relation_matrix_batch.to(device)

            output_batch_A, batch_mask = model(x_batch_A)     # output_batch shape [bs, num_seq, 4]
            output_batch = output_batch_A.permute([0, 2, 1])
            loss_A = criterion(output_batch.squeeze(), label_batch_A.type(torch.long).squeeze())
            loss_A = torch.div(torch.sum(loss_A*batch_mask), torch.sum(batch_mask))
            padding_a_mask = batch_mask
            output_batch_B, batch_mask = model(x_batch_B)
            output_batch = output_batch_B.permute([0, 2, 1])
            loss_B = criterion(output_batch.squeeze(), label_batch_B.type(torch.long).squeeze())
            loss_B = torch.div(torch.sum(loss_B * batch_mask), torch.sum(batch_mask))
            padding_b_mask = batch_mask
            loss = loss_B + loss_A
            loss_ot = OT_model(output_batch_A.detach(), output_batch_B.detach(), x_batch_A, x_batch_B, relation_matrix_batch.clone(), padding_a_mask, padding_b_mask, convert_to_onehot=args.convert_to_onehot, convert_mode=args.ot_mode)
            if args.coefficient_lrable:
                loss_all = 1/torch.square(theta1) * loss + 1/torch.square(theta2) * loss_ot + torch.log(theta1 * theta2)  # todo 参数设置看看
            else:
                loss_all = loss + args.coefficient * loss_ot  # todo 参数设置看看
            #loss_all = loss
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            loss_item = loss_all.cpu().detach().item()
            loss_item_ot = loss_ot.cpu().detach().item()
            loss_item_dgcnn = loss.cpu().detach().item()

            epoch_loss += loss_item
            epoch_loss_ot += loss_item_ot
            epoch_loss_dgcnn += loss_item_dgcnn

            current_step += 1
            pbar.set_description("train loss_all {}, dgcnn loss {}, ot loss {}".format(epoch_loss / current_step,
                                                                                       epoch_loss_dgcnn / current_step,
                                                                                       epoch_loss_ot / current_step))
            if current_step % 100 == 0:
                logging.info("train loss_all {}, dgcnn loss {}, ot loss {}".format(epoch_loss / current_step,
                                                                                       epoch_loss_dgcnn / current_step,
                                                                                       epoch_loss_ot / current_step))

        epoch_loss = epoch_loss / current_step
        epoch_loss_dgcnn = epoch_loss_dgcnn / current_step
        epoch_loss_ot = epoch_loss_ot / current_step

        time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('{} train epoch {} loss_all: {:.4f} loss_dgcnn: {:.4f} loss_ot: {:.10f}'.format(time_str, epoch, epoch_loss,
                                                                                             epoch_loss_dgcnn, epoch_loss_ot))
        logging.info('{} train epoch {} loss_all: {:.4f} loss_dgcnn: {:.4f} loss_ot: {:.10f}'.format(time_str, epoch, epoch_loss,
                                                                                             epoch_loss_dgcnn, epoch_loss_ot))

        with torch.no_grad():
            model.eval()
            current_val_metric_value, OT_Acc_valid, OT_Pre_valid, OT_Recall_valid, OT_F1_valid, seven_class_DGCNN_valid = evaluate(model, OT_model, valid_dataloader, epoch, type='valid')
            nni.report_intermediate_result(current_val_metric_value.item())
            is_save = early_stop.step(current_val_metric_value, epoch)
            if is_save:
                save_checkpoint(model, optimizer, epoch, OT_model)
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
                nni.report_final_result(early_stop.best_value.item())
                break
            evaluate(model, OT_model, test_dataloader, epoch, type='test')

def evaluate(model, OT_model, valid_dataloader, epoch, type='valid'):

    correct_DGCNN = 0
    correct_DGCNN_seven = 0
    total_DGCNN = 0
    current_step = 0
    acc_1, acc_2 = 0, 0
    recall_1, recall_2 = 0, 0
    pre_1, pre_2 = 0, 0
    pbar = tqdm(valid_dataloader, desc="Iteration", postfix=type)
    seven_class_predictions = []
    seven_class_labels = []

    for batch_data in pbar:
        x_batch_A, x_batch_B, label_batch_A, label_batch_B, seven_label_batch_A, seven_label_batch_B, relation_matrix_batch = batch_data
        print("max relation_matrix_batch: ", torch.max(relation_matrix_batch),'\n')
        x_batch_A = x_batch_A.to(device)
        x_batch_B = x_batch_B.to(device)
        label_batch_A = label_batch_A.to(device).long()
        label_batch_B = label_batch_B.to(device).long()
        seven_label_batch_A = seven_label_batch_A.to(device).long()
        seven_label_batch_B = seven_label_batch_B.to(device).long()
        seven_class_labels.append(seven_label_batch_A)
        seven_class_labels.append(seven_label_batch_B)
        relation_matrix_batch = relation_matrix_batch.to(device)
        output_batch_A, batch_mask_A = model(x_batch_A)
        output_batch_B, batch_mask_B = model(x_batch_B)
        batch_mask_A_3D, batch_mask_B_3D = batch_mask_A.unsqueeze(-1), batch_mask_B.unsqueeze(1)
        padding_mask = torch.bmm(batch_mask_A_3D.to(float), batch_mask_B_3D.to(float))
        plan_list = OT_model(output_batch_A, output_batch_B, x_batch_A, x_batch_B, relation_matrix_batch,
                              batch_mask_A, batch_mask_B, model_type=type)
        seven_prediction_A, seven_prediction_B = model_class(output_batch_A.clone(), output_batch_B.clone(), plan_list)
        seven_class_predictions.append(seven_prediction_A*batch_mask_A)
        seven_class_predictions.append(seven_prediction_B*batch_mask_B)
        total_DGCNN += torch.sum(batch_mask_A)+torch.sum(batch_mask_B)
        vec_correct_A = (torch.argmax(output_batch_A, dim=-1).long() == label_batch_A.squeeze().long())*batch_mask_A
        vec_correct_B = (torch.argmax(output_batch_B, dim=-1).long() == label_batch_B.squeeze().long())*batch_mask_B

        correct_DGCNN += torch.sum(vec_correct_A).cpu().item()+torch.sum(vec_correct_B).cpu().item()
        vec_correct_seven_A = (seven_prediction_A == seven_label_batch_A.squeeze().long())*batch_mask_A
        vec_correct_seven_B = (seven_prediction_B == seven_label_batch_B.squeeze().long())*batch_mask_B
        correct_DGCNN_seven += torch.sum(vec_correct_seven_A).cpu().item()+torch.sum(vec_correct_seven_B).cpu().item()

        temp_n, temp_d = ot_acc_score(relation_matrix_batch.long(),
                                      torch.ge(plan_list, 1 / batch_mask_A.shape[1]/args.threshold_ot).long(),
                                      padding_mask)  # todo 阈值设置也需要看看
        acc_1 += temp_n
        acc_2 += temp_d
        temp_n, temp_d = ot_precision_score(relation_matrix_batch.long(),
                                      torch.ge(plan_list, 1 / batch_mask_A.shape[1]/args.threshold_ot).long(),
                                      padding_mask)
        pre_1 += temp_n
        pre_2 += temp_d

        temp_n, temp_d = ot_recall_score(relation_matrix_batch.long(),
                                        torch.ge(plan_list, 1 / batch_mask_A.shape[1]/args.threshold_ot).long(),
                                        padding_mask)
        recall_1 += temp_n
        recall_2 += temp_d

        P = pre_1/pre_2
        R = recall_1/recall_2

        pbar.set_description("{} acc DGCNN {} acc_seven {} f1 OT {}".format(type, correct_DGCNN / total_DGCNN, correct_DGCNN_seven/total_DGCNN, 2*R*P/(R+P)))
        current_step += 1
        if current_step % 100 == 0:
            logging.info('{} epoch {} four_class acc {}/{}={:.4f} seven_class acc {}/{}={:.4f}'.format(type, epoch, correct_DGCNN, total_DGCNN, correct_DGCNN / total_DGCNN,
                                                                                                       correct_DGCNN_seven, total_DGCNN, correct_DGCNN_seven/total_DGCNN))
            logging.info('{} epoch {} acc {}/{}={:.4f} precision: {:.4f} recall: {:.4f} f1: {:.4f}'.format(type, epoch, acc_1, acc_2, acc_1 / acc_2,
                                                                                                           pre_1/pre_2, recall_1/recall_2, 2*R*P/(R+P)))
    seven_class_predictions = torch.cat(seven_class_predictions, dim=0).view(-1).cpu().numpy()
    seven_class_labels = torch.cat(seven_class_labels, dim=0).view(-1).cpu().numpy()
    conf_matrix = confusion_matrix(seven_class_labels, seven_class_predictions)

    #Print the confusion matrix using Matplotlib

    # fig, ax = plt.subplots(figsize=(6, 6))
    # ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    # for i in range(conf_matrix.shape[0]):
    #     for j in range(conf_matrix.shape[1]):
    #         ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    #
    # plt.xlabel('Predicted', fontsize=18)
    # plt.ylabel('Actuals', fontsize=18)
    # plt.title('Confusion Matrix {}'.format(type), fontsize=18)
    # plt.show()

    P_sum = pre_1 / pre_2
    R_sum = recall_1 / recall_2
    print('{} epoch {} four_class acc {}/{}={:.4f} '.format(type, epoch, correct_DGCNN, total_DGCNN, correct_DGCNN / total_DGCNN))
    print(colored('{} epoch {} acc {}/{}={:.4f} precision: {:.4f} recall: {:.4f} f1: {:.4f} seven_class acc {}/{}={:.4f} '.format(type, epoch, acc_1, acc_2,
                                                                                          acc_1 / acc_2,
                                                                                          pre_1 / pre_2,
                                                                                          recall_1 / recall_2,
                                                                                          2*P_sum*R_sum/(P_sum+R_sum),correct_DGCNN_seven, total_DGCNN, correct_DGCNN_seven/total_DGCNN), color='yellow'))
    logging.info('{} epoch {} four_class acc {}/{}={:.4f} seven_class acc {}/{}={:.4f}'.format(type, epoch, correct_DGCNN, total_DGCNN, correct_DGCNN / total_DGCNN,
                                                                                                       correct_DGCNN_seven, total_DGCNN, correct_DGCNN_seven/total_DGCNN))
    logging.info(
        '{} epoch {} acc {}/{}={:.4f} precision: {:.4f} recall: {:.4f} f1: {:.4f}'.format(type, epoch, acc_1, acc_2,
                                                                                          acc_1 / acc_2,
                                                                                          pre_1 / pre_2,
                                                                                          recall_1 / recall_2,
                                                                                          2 * P_sum * R_sum / (P_sum + R_sum)))
    return correct_DGCNN/total_DGCNN, acc_1 / acc_2, pre_1 / pre_2, recall_1 / recall_2, 2*P_sum*R_sum/(P_sum+R_sum), correct_DGCNN_seven/total_DGCNN


def model_class(batch_A, batch_B, plan_matrix):
    """
    :param model:
    :param OT_model:
    :param case_A:
    :param case_B:
    :return: AO, YO, ZO, AI, YI, ZI
    """
    relation_A = torch.ge(torch.sum(torch.ge(plan_matrix, 1 / batch_A.shape[1]/args.threshold_ot).long(), dim=-1), 1).long()
    relation_B = torch.ge(torch.sum(torch.ge(plan_matrix, 1 / batch_A.shape[1]/args.threshold_ot).long(), dim=-2), 1).long()

    vec_correct_A = torch.argmax(batch_A, dim=-1) + (relation_A*1)*torch.ge(torch.argmax(batch_A, dim=-1), 1)  # todo
    vec_correct_B = torch.argmax(batch_B, dim=-1) + (relation_B*1)*torch.ge(torch.argmax(batch_B, dim=-1), 1)  # todo

    return vec_correct_A, vec_correct_B


def OT_param_search(model, valid_dataloader, test_dataloader):
    """
    threshold_ot: [1.0, ... 3.0] 20
    f_mass: [1.0-3.0] 20
    weight: [100, 200, 500] 3
    reg:    [0.3 - 1.] 7
    """
    threshold_ot_list = list(np.arange(1, 2, 0.1))
    weight_list = [0, 10, 100, 200]
    f_mass_list = [2]
    reg_list = list(np.arange(0.1, 1, 0.1))
    pd_dic_valid = {"threshold_ot": [], "f_mass": [], "weight": [], "reg": [], "DGCNN_Acc": [], "OT_Acc": [],
                    "OT_Pre": [], "OT_Recall": [], "OT_F1": [],'seven_class_DGCNN':[]}
    pd_dic_test = {"threshold_ot": [], "f_mass": [], "weight": [], "reg": [], "DGCNN_Acc": [], "OT_Acc": [],
                   "OT_Pre": [], "OT_Recall": [], "OT_F1": [], 'seven_class_DGCNN':[]}
    filename_OT = args.checkpoint + '/' + "cail_extract_ot-criterion-{}-onehot-{}-ot_mode-{}-convert_to_onehot-{}-weight-{}-simot-{}-simpercent-{}.pkl".format(
        args.criterion, args.convert_to_onehot, args.ot_mode, args.convert_to_onehot,
        args.weight, args.simot, args.simpercent)
    for threshold_ot in threshold_ot_list:
        for f_mass in f_mass_list:
            for weight in weight_list:
                for reg in reg_list:
                    args.threshold_ot = threshold_ot
                    args.f_mass = f_mass
                    args.reg = reg
                    OT_model = OT(reg=args.reg).to(device)
                    args.weight = weight
                    load_checkpoint(OT_model, None, -1, filename_OT)
                    DGCNN_Acc_valid, OT_Acc_valid, OT_Pre_valid, OT_Recall_valid, OT_F1_valid, seven_class_DGCNN_valid = evaluate(model,
                                                                                                         OT_model,
                                                                                                         valid_dataloader,
                                                                                                         22, 'valid')
                    pd_dic_valid["threshold_ot"].append(threshold_ot)
                    pd_dic_valid["f_mass"].append(f_mass)
                    pd_dic_valid["weight"].append(weight)
                    pd_dic_valid["reg"].append(reg)
                    pd_dic_valid["DGCNN_Acc"].append(DGCNN_Acc_valid.cpu().item())
                    pd_dic_valid["OT_Acc"].append(OT_Acc_valid.cpu().item())
                    pd_dic_valid["OT_Pre"].append(OT_Pre_valid.cpu().item())
                    pd_dic_valid["OT_Recall"].append(OT_Recall_valid.cpu().item())
                    pd_dic_valid["OT_F1"].append(OT_F1_valid.cpu().item())
                    pd_dic_valid['seven_class_DGCNN'].append(seven_class_DGCNN_valid.cpu().item())
                    DGCNN_Acc_test, OT_Acc_test, OT_Pre_test, OT_Recall_test, OT_F1_test, seven_class_DGCNN_test = evaluate(model, OT_model,
                                                                                                    test_dataloader, 22,
                                                                                                    'test')
                    pd_dic_test["threshold_ot"].append(threshold_ot)
                    pd_dic_test["f_mass"].append(f_mass)
                    pd_dic_test["weight"].append(weight)
                    pd_dic_test["reg"].append(reg)
                    pd_dic_test["DGCNN_Acc"].append(DGCNN_Acc_test.cpu().item())
                    pd_dic_test["OT_Acc"].append(OT_Acc_test.cpu().item())
                    pd_dic_test["OT_Pre"].append(OT_Pre_test.cpu().item())
                    pd_dic_test["OT_Recall"].append(OT_Recall_test.cpu().item())
                    pd_dic_test["OT_F1"].append(OT_F1_test.cpu().item())
                    pd_dic_test["seven_class_DGCNN"].append(seven_class_DGCNN_test.cpu().item())
    pd_valid = pd.DataFrame(pd_dic_valid)
    pd_test = pd.DataFrame(pd_dic_test)
    # todo
    pd_valid.to_csv("../dataset/transfer_cail_valid_unsup_{}_semi_percent_{}.csv".format(args.simot, args.simpercent), index=False)
    pd_test.to_csv("../dataset/transfer_cail_test_unsup_{}_semi_percent_{}.csv".format(args.simot, args.simpercent), index=False)




if __name__ == '__main__':

    # 加载数据
    data = load_data(data_extract_json)
    data_x = np.load(data_extract_npy, allow_pickle=True) # [2*num_data, max_num_seq, dim]
    data_y = np.zeros_like(data_x[..., :1])  # [2*num_data, max_num_seq, 1]
    data_y_seven_class = np.zeros_like(data_x[..., :1])  # [2*num_data, max_num_seq, 1]
    data_matrix = np.zeros([data_x.shape[0]//2, data_x.shape[1], data_x.shape[1]])

    for i, d in enumerate(data):
        important_A, important_B = [], []
        case_a = d['case_A']
        for j in case_a[1]:
            data_y[2*i, j[0]] = j[1]
        case_b = d['case_B']
        for j in case_b[1]:
            data_y[2*i+1, j[0]] = j[1]
        for pos in d['relation_label']:
            row, col = pos[0], pos[-1]
            data_matrix[i][row][col] = 1.
            important_A.append(row)
            important_B.append(col)
        # for pos_list in d['relation_label'].values():
        #     for pos in pos_list:
        #         row, col = pos[0], pos[-1]
        #         data_matrix[i][row][col] = 1.
        #         important_A.append(row)
        #         important_B.append(col)

        for j in case_a[1]:
            if j[0] in important_A:
                data_y_seven_class[2*i, j[0]] = j[1]+1 # todo
            else:
                data_y_seven_class[2*i, j[0]] = j[1]

        for j in case_b[1]:
            if j[0] in important_B:
                data_y_seven_class[2*i+1, j[0]] = j[1]+1 # todo
            else:
                data_y_seven_class[2*i+1, j[0]] = j[1]


    train_data = data_split(data, 'train', if_random=True)
    valid_data = data_split(data, 'valid',  if_random=True)
    test_data = data_split(data, 'test',  if_random=True)

    data_x_A = data_x[0:data_x.shape[0]:2]
    data_x_B = data_x[1:data_x.shape[0]:2]
    data_y_A = data_y[0:data_x.shape[0]:2]
    data_y_B = data_y[1:data_x.shape[0]:2]
    data_y_seven_class_A = data_y_seven_class[0:data_x.shape[0]:2]
    data_y_seven_class_B = data_y_seven_class[1:data_x.shape[0]:2]

    train_x_A = data_split(data_x_A, 'train',  if_random=True)
    train_x_B = data_split(data_x_B, 'train',  if_random=True)
    train_y_A = data_split(data_y_A, 'train',  if_random=True)
    train_y_B = data_split(data_y_B, 'train',  if_random=True)
    train_y_seven_class_A = data_split(data_y_seven_class_A, 'train',  if_random=True)
    train_y_seven_class_B = data_split(data_y_seven_class_B, 'train',  if_random=True)
    train_relation_matrix = data_split(data_matrix, 'train',  if_random=True)

    valid_x_A = data_split(data_x_A, 'valid',  if_random=True)
    valid_x_B = data_split(data_x_B, 'valid',  if_random=True)
    valid_y_A = data_split(data_y_A, 'valid',  if_random=True)
    valid_y_B = data_split(data_y_B, 'valid',  if_random=True)
    valid_y_seven_class_A = data_split(data_y_seven_class_A, 'valid',  if_random=True)
    valid_y_seven_class_B = data_split(data_y_seven_class_B, 'valid',  if_random=True)
    valid_relation_matrix = data_split(data_matrix, 'valid',  if_random=True)

    test_x_A = data_split(data_x_A, 'test',  if_random=True)
    test_x_B = data_split(data_x_B, 'test',  if_random=True)
    test_y_A = data_split(data_y_A, 'test',  if_random=True)
    test_y_B = data_split(data_y_B, 'test',  if_random=True)
    test_y_seven_class_A = data_split(data_y_seven_class_A, 'test',  if_random=True)
    test_y_seven_class_B = data_split(data_y_seven_class_B, 'test',  if_random=True)
    test_relation_matrix = data_split(data_matrix, 'test',  if_random=True)



    train_dataloader = DataLoader(Selector_Dataset_Pair(train_x_A, train_x_B, train_y_A, train_y_B, train_y_seven_class_A, train_y_seven_class_B, train_relation_matrix), batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(Selector_Dataset_Pair(valid_x_A, valid_x_B, valid_y_A, valid_y_B, valid_y_seven_class_A, valid_y_seven_class_B, valid_relation_matrix), batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(Selector_Dataset_Pair(test_x_A, test_x_B, test_y_A, test_y_B,  test_y_seven_class_A, test_y_seven_class_B, test_relation_matrix), batch_size=args.batch_size, shuffle=False)

    model = Selector2_mul_class(args.input_size, args.hidden_size, kernel_size=args.kernel_size, dilation_rate=[1, 2, 4, 8, 1, 1]).to(device)

    if args.model_usage == 'train':
        OT_model = OT(reg=args.reg).to(device)
        train_pair(args, model, OT_model, train_dataloader, valid_dataloader, test_dataloader)
        filename_DGCNN = args.checkpoint + '/' + "cail_extract-criterion-{}-onehot-{}-ot_mode-{}-convert_to_onehot-{}-weight-{}-simot-{}-simpercent-{}.pkl".format(args.criterion, args.convert_to_onehot, args.ot_mode, args.convert_to_onehot,
                                                                                                 args.weight, args.simot, args.simpercent)

        load_checkpoint(model, None, 22, filename_DGCNN)

        OT_param_search(model, valid_dataloader, test_dataloader)
    elif args.model_usage == 'eval':
        filename_DGCNN = args.checkpoint + '/' + "cail_extract-criterion-{}-onehot-{}-ot_mode-{}-convert_to_onehot-{}-weight-{}-simot-{}-simpercent-{}.pkl".format(
            args.criterion, args.convert_to_onehot, args.ot_mode, args.convert_to_onehot,
            args.weight, args.simot, args.simpercent)
        load_checkpoint(model, None, 22, filename_DGCNN)
        OT_model = OT(reg=args.reg).to(device)
        filename_OT = args.checkpoint + '/' + "cail_extract_ot-criterion-{}-onehot-{}-ot_mode-{}-convert_to_onehot-{}-weight-{}-simot-{}-simpercent-{}.pkl".format(
            args.criterion, args.convert_to_onehot, args.ot_mode, args.convert_to_onehot,
            args.weight, args.simot, args.simpercent)
        load_checkpoint(OT_model, None, -1, filename_OT)
        evaluate(model, OT_model, train_dataloader, 0, 'valid')
        evaluate(model, OT_model, valid_dataloader, 0, 'valid')
        evaluate(model, OT_model, test_dataloader, 0, 'test')
    elif args.model_usage == 'search':
        filename_DGCNN = args.checkpoint + '/' + "cail_extract-criterion-{}-onehot-{}-ot_mode-{}-convert_to_onehot-{}-weight-{}-simot-{}-simpercent-{}.pkl".format(args.criterion, args.convert_to_onehot, args.ot_mode, args.convert_to_onehot,
                                                                                                 args.weight, args.simot, args.simpercent)

        load_checkpoint(model, None, 22, filename_DGCNN)

        OT_param_search(model, valid_dataloader, test_dataloader)




