# -*- coding: utf-8 -*-
# -------------------------------------
# Description:
# @Author: tiniwu (tiniwu@tencent.com)
# CopyRight: Tencent Company


import os
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from models.bilstm import SiaGRU
from utils.data_helper import data_process_similarity, data_process_class
from torch.utils.data import DataLoader, TensorDataset

encoded_a, encoded_b, encoded_labels = data_process_similarity("data/ATEC/train.txt")
train_data = TensorDataset(torch.from_numpy(encoded_a), torch.from_numpy(encoded_b), torch.from_numpy(encoded_labels))
train_loader = DataLoader(train_data, shuffle=True, batch_size=32)

valid_encoded_a, valid_encoded_b, valid_encoded_labels = data_process_similarity("data/ATEC/dev.txt", 16, False)
valid_data = TensorDataset(torch.from_numpy(valid_encoded_a), torch.from_numpy(valid_encoded_b), torch.from_numpy(valid_encoded_labels))
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=16)

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def train_model(model, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters())) # filter(lambda p: p.requires_grad, model.parameters())
    steps = 0
    model.train()
    for idx, (q1, q2, y) in enumerate(tqdm(train_loader, desc='Iteration')):
        target = y
        # target = torch.autograd.Variable(target).long()
        # if (text.size()[0] is not 32):  # One of the batch returned by BucketIterator has length different than 32.
        #     continue
        optim.zero_grad()
        prediction = model(q1, q2)

        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1] == target).sum().item()
        acc = 100.0 * num_corrects / len(y)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        if steps % 100 == 0:
            print(f'Epoch: {epoch + 1}, Idx: {idx + 1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc: .2f}%')

        total_epoch_loss += loss.item()
        total_epoch_acc += acc

    return total_epoch_loss / len(train_loader), total_epoch_acc / len(train_loader)


def eval_model(model):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, (q1, q2, y) in enumerate(valid_loader):
            # target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            prediction = model(q1,q2)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects / len(y)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss / len(valid_loader), total_epoch_acc / len(valid_loader)

params = {
    'is_training':True,
    'seq_length':32,
    'class_num':2,
    'embedding_size':200,
    'hidden_num':2,
    'hidden_size':100,
    'vocab_size':100000,
    'batch_size':16,
    'learning_rate':0.0001,
    'l2_lambda':0.01
}


learning_rate = 2e-5
batch_size = 32
output_size = 2
hidden_size = 256
embedding_length = 300

model = SiaGRU(params)
loss_fn = F.cross_entropy

for epoch in range(10):
    train_loss, train_acc = train_model(model, epoch)
    val_loss, val_acc = eval_model(model)

    print(f'Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')
# Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%
# test_loss, test_acc = eval_model(model, test_iter)
# print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')

# ''' Let us now predict the sentiment on a single sentence just for the testing purpose. '''
# test_sen1 = "This is one of the best creation of Nolan. I can say, it's his magnum opus. Loved the soundtrack and especially those creative dialogues."
# test_sen2 = "Ohh, such a ridiculous movie. Not gonna recommend it to anyone. Complete waste of time and money."
#
# test_sen1 = TEXT.preprocess(test_sen1)
# test_sen1 = [[TEXT.vocab.stoi[x] for x in test_sen1]]
#
# test_sen2 = TEXT.preprocess(test_sen2)
# test_sen2 = [[TEXT.vocab.stoi[x] for x in test_sen2]]
#
# test_sen = np.asarray(test_sen1)
# test_sen = torch.LongTensor(test_sen)
# test_tensor = Variable(test_sen, volatile=True)
# test_tensor = test_tensor.cuda()
# model.eval()
# output = model(test_tensor, 1)
# out = F.softmax(output, 1)
# if (torch.argmax(out[0]) == 1):
#     print("Sentiment: Positive")
# else:
#     print("Sentiment: Negative")
#
#
