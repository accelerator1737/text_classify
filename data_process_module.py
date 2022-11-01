import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
from torch.utils.data import Dataset, DataLoader



MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def get_data(config, ues_word):
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # 词级别

    vocab = pkl.load(open(config.vocab_path, 'rb'))

    print(f"Vocab size: {len(vocab)}")

    train = load_dataset(config.train_path, config.pad_size, tokenizer, vocab)
    dev = load_dataset(config.dev_path, config.pad_size, tokenizer, vocab)
    test = load_dataset(config.test_path, config.pad_size, tokenizer, vocab)
    return vocab, train, dev, test


def load_dataset(path, pad_size, tokenizer, vocab):
    '''
    将路径文本文件分词并转为三元组返回
    :param path: 文件路径
    :param pad_size: 每个序列的大小
    :param tokenizer: 转为词级别或字级别
    :param vocab: 词向量模型
    :return: 二元组，含有字ID，标签
    '''
    contents = []
    with open(path, 'r', encoding='UTF-8') as f:
        # tqdm可以看进度条
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content, label = lin.split('\t')

            # word_line存储每个字的id
            words_line = []
            # 分割器，分词每个字
            token = tokenizer(content)
            # 字的长度
            seq_len = len(token)
            if pad_size:
                # 如果字长度小于指定长度，则填充，否则截断
                if len(token) < pad_size:
                    token.extend([vocab.get(PAD)] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # 将每个字映射为ID，本来就是ID的不变
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append((words_line, int(label)))
    return contents

class TextDataset(Dataset):
    def __init__(self, data, config):
        self.device = config.device
        self.x = torch.LongTensor([x[0] for x in data]).to(self.device)
        self.y = torch.LongTensor([x[1] for x in data]).to(self.device)


    def __getitem__(self,index):
        self.text = self.x[index]
        self.label = self.y[index]
        return self.text, self.label


    def __len__(self):
        return len(self.x)


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


