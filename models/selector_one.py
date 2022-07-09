import sys
sys.path.append("..")
import argparse
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from utils.snippets import *
import torch.nn as nn
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_pos', type=str, default='0', help='which GPU to use')
parser.add_argument('--seed', type=int, default=42, help='max length of each case')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device('cuda:'+args.cuda_pos) if torch.cuda.is_available() else torch.device('cpu')



class GlobalAveragePooling1D(nn.Module):
    """自定义全局池化
    对一个句子的pooler取平均，一个长句子用短句的pooler平均代替
    """
    def __init__(self):
        super(GlobalAveragePooling1D, self).__init__()


    def forward(self, inputs, mask=None):
        if mask is not None:
            mask = mask.to(torch.float)[:, :, None]
            return torch.sum(inputs * mask, dim=1) / torch.sum(mask, dim=1)
        else:
            return torch.mean(inputs, dim=1)


class Selector_1(nn.Module):
    def __init__(self):
        super(Selector_1, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_fold, mirror='tuna', do_lower_case=True)
        self.Pooling = GlobalAveragePooling1D()
        self.encoder = BertModel.from_pretrained(pretrained_bert_fold)
        self.max_seq_len = 512


    def predict(self, texts):
        """句子列表转换为句向量
        """
        with torch.no_grad():
            output_1s = []
            bert_output = self.tokenizer.batch_encode_plus(texts, padding=True, truncation=True, max_length=self.max_seq_len)
            for p in range(len(texts)):
                bert_output_p = {"input_ids": torch.tensor([bert_output["input_ids"][p]], device=device), "token_type_ids": torch.tensor([bert_output["token_type_ids"][p]], device=device),
                                 "attention_mask": torch.tensor([bert_output["attention_mask"][p]], device=device)}
                output_1 = self.encoder(**bert_output_p)["last_hidden_state"]
                output_1s.append(output_1[0])
            output_1_final = torch.stack(output_1s, dim=0)
            outputs = self.Pooling(output_1_final)
        return outputs



def load_data(filename):
    """加载数据
    返回：[texts]
    """
    D = []
    with open(filename) as f:
        for l in f:
            texts_a = json.loads(l)['case_A'][0]
            texts_b = json.loads(l)['case_B'][0]
            D.append([texts_a, texts_b])

    return D




def convert(data):
    """转换所有样本
    """
    embeddings = []
    model = Selector_1()
    model.to(device)
    for texts in tqdm(data, desc=u'向量化'):
        outputs_a = model.predict(texts[0])
        outputs_b = model.predict(texts[1])
        embeddings.append(outputs_a)
        embeddings.append(outputs_b)
    embeddings = sequence_padding(embeddings)
    return embeddings


if __name__ == '__main__':
    #
    data_extract_json = '../dataset/our_data/data_extract_old.json'
    data_extract_npy = '../dataset/our_data/data_extract_old.npy'
    data = load_data(data_extract_json)
    embeddings = convert(data)
    np.save(data_extract_npy, embeddings)
    print(u'输出路径：%s' % data_extract_npy)
