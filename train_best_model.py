import torch.nn as nn
import time
import torch
import copy
import pandas as pd
import datetime
from sklearn import metrics
import numpy as np


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train_best(config, model, dataloaders, log_step=100):
    '''
    训练模型
    :param config: 超参数
    :param model: 模型
    :param dataloaders: 处理后的数据，包含trian,dev,test
    :param log_step: 每隔多少个batch打印一次数据，默认100
    :return: 训练的指标
    '''

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()
    best_acc = 0
    # 最优模型
    best_model = copy.deepcopy(model.state_dict())

    total_step = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    # 保存每一个epoch的信息
    dfhistory = pd.DataFrame(columns=["epoch", "train_loss", "train_acc", "dev_loss", "dev_acc"])

    device = config.device

    print("Start Training...\n")
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("==========" * 8 + "%s\n" % nowtime)

    for i in range(config.num_epochs):

        # 1，训练循环----------------------------------------------------------------


        # 将数据全部取完

        # 记录每一个batch
        step = 0

        print('Epoch [{}/{}]\n'.format(i + 1, config.num_epochs))

        for inputs, labels in dataloaders['train']:
            # 训练模式，可以更新参数
            model.train()

            inputs = inputs.to(device)
            labels = labels.to(device)
            # 梯度清零，防止累加
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            total_step += 1
            step += 1

            if step % log_step == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_loss = loss.item()
                train_acc = metrics.accuracy_score(true, predic)

                # 2，开发集验证----------------------------------------------------------------
                dev_acc, dev_loss = dev_eval(model, dataloaders['dev'], loss_function)

                dfhistory.loc[i] = (i, train_loss, train_acc, dev_loss, dev_acc)

                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    last_improve = total_step

                print("[step = {} batch]  train_loss = {:.3f}, train_acc = {:.2%}, dev_loss = {:.3f}, dev_acc = {:.2%}".
                      format(step, train_loss, train_acc, dev_loss, dev_acc))

            if total_step - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break

    # 3，验证循环----------------------------------------------------------------
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss = dev_eval(model, dataloaders['test'], loss_function)
    print('================'*8)
    print('test_loss: {:.3f}      test_acc: {:.2%}'.format(test_loss, test_acc))

    return dfhistory


def dev_eval(model, data, loss_function):
    '''
    得到开发集和测试集的准确率和loss
    :param model: 模型
    :param data: 测试集集和开发集的数据
    :param loss_function: 损失函数
    :return: 损失和准确率
    '''
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data:
            outputs = model(texts)
            loss = loss_function(outputs, labels)
            loss_total += loss.item()
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    return acc, loss_total / len(data)
