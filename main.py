import torch
from RNNmodel import *
import time
from data_process_module import *
from torch.utils.data import DataLoader
from train_best_model import *

# 存放数据的文件夹
dataset = 'text_classify_data'

# 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
embedding = 'embedding_SougouNews.npz'

config = Config(dataset, embedding)
# 设置随机数种子，保证每次运行结果一致，不至于不能复现模型
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

start_time = time.time()
print("Loading data...")
vocab, train_data, dev_data, test_data = get_data(config, False)
dataloaders = {
    'train': DataLoader(TextDataset(train_data, config), 128, shuffle=True),
    'dev': DataLoader(TextDataset(dev_data, config), 128, shuffle=True),
    'test': DataLoader(TextDataset(test_data, config), 128, shuffle=True)
}

time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)

# train
config.n_vocab = len(vocab)
model = RNNModel(config).to(config.device)
# writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
init_network(model)
print(model.parameters)
train_best(config, model, dataloaders)