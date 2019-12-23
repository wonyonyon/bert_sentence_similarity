# -*- coding: utf-8 -*-
# -------------------------------------
# Description:
# @Author: tiniwu (tiniwu@tencent.com)
# CopyRight: Tencent Company

import os
import torch
from torch.nn import functional as F
from torch.autograd import Variable
import torch.nn as nn


class SiaGRU(nn.Module):
    def __init__(self, config):
        super(SiaGRU, self).__init__()
        self.config = config
        self.word_embedding = nn.Embedding(config['vocab_size'], config['embedding_size'])
        # whether assign pre_trained word vector models or not
        # self.word_embedding.weight = nn.Parameter(weights, requires_grad=True)
        # self.norm_embedding = nn.LayerNorm(config['embedding_size'])
        self.gru = nn.LSTM(config['embedding_size'], config['hidden_size'], batch_first=True, bidirectional=True, num_layers=2)
        # self.h0 = self.init_hidden((2*config['hidden_num'], 1, config['hidden_size']))
        # self.fc = nn.Linear(8*config['hidden_size'], config['class_num'])
        self.classification = nn.Sequential(nn.Linear(2*2*4*config['hidden_size'],config['hidden_size']), nn.Tanh(), nn.Dropout(0.5), nn.Linear(config['hidden_size'], config['class_num']),nn.Softmax(dim=-1))

    def init_hidden(self, size):
        h0 = nn.Parameter(torch.randn(size))
        nn.init.xavier_normal_(h0)
        return h0

    def forward(self, text_a, text_b):
        text_a_embedding = self.word_embedding(text_a)
        text_b_embedding = self.word_embedding(text_b)
        # print(text_a_embedding.size())
        # print(text_a_embedding.transpose(1, 2).size())
        output_a, (final_hidden_state_a, final_cell_state_a) = self.gru(text_a_embedding)
        # print(output_a.size(), final_hidden_state_a.size(), final_cell_state_a.size())
        output_b, (final_hidden_state_b, final_cell_state) = self.gru(text_b_embedding)
        enhanced_output = torch.cat([output_a, output_b, output_a - output_b, output_a*output_b], dim=-1)
        # print(enhanced_output.size())
        p2 = F.max_pool1d(enhanced_output.transpose(1, 2), enhanced_output.size(1)).squeeze(-1)
        p1 = F.avg_pool1d(enhanced_output.transpose(1, 2), enhanced_output.size(1)).squeeze(-1)
        # torch.cat([p1, p2], 1)
        # print(p2.size())
        # sim = torch.exp(-torch.norm(final_hidden_state_a.squeeze() - final_hidden_state_b.squeeze(), p=2, dim=-1, keepdim=True))

        return self.classification(torch.cat([p1, p2], 1))

# params = {
#     'is_training':True,
#     'seq_length':32,
#     'class_num':2,
#     'embedding_size':200,
#     'hidden_num':2,
#     'hidden_size':100,
#     'vocab_size':100000,
#     'batch_size':16,
#     'learning_rate':0.0001,
#     'l2_lambda':0.01
# }
# sigru = SiaGRU(config=params)
# print(sigru)
