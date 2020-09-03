import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

from modules_.tanh_attention import TanhAttention
from modules_.dynamic_rnn import DynamicGRU
from modules_.ConvGRU import ConvGRUCell
from modules_.Coattention import CoAttention, CoAttention_intra
from modules_.multihead_attention import MultiHeadAttention
from modules_.position import RelTemporalEncoding
from utils import generate_anchors


class VideoEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.args = args
        if args.dataset == 'TACOS':
            self.tran = nn.Linear(4096, args.frame_dim)
        #else:
        #    self.tran = nn.Linear(args.frame_dim, args.frame_dim)
        self.max_num_frames = args.max_num_frames
        self.attn_layers = nn.ModuleList([
            MultiHeadAttention(args.frame_dim, args.num_heads)
            for _ in range(args.num_attn_layers)
        ])
        self.rnn = DynamicGRU(args.frame_dim, args.d_model >> 1, bidirectional=True, batch_first=False)
        self.attn_width = 3
        self.self_attn_mask = torch.empty(self.max_num_frames, self.max_num_frames) \
            .float().fill_(float(-1e10)).cuda()
        for i in range(0, self.max_num_frames):
            low = i - self.attn_width
            low = 0 if low < 0 else low
            high = i + self.attn_width + 1
            high = self.max_num_frames if high > self.max_num_frames else high
            # attn_mask[i, low:high] = 0
            self.self_attn_mask[i, low:high] = 0

    def forward(self, x, mask):
        if self.args.dataset == 'TACOS':
            x = self.tran(x)
        #else:
        #    x = self.tran(x)
        x = x.transpose(0, 1)
        length = mask.sum(dim=-1)

        for a in self.attn_layers:
            res = x
            x, _ = a(x, x, x, None, attn_mask=self.self_attn_mask)
            x = F.dropout(x, self.dropout, self.training)
            x = res + x

        x = self.rnn(x, length, self.max_num_frames)
        x = F.dropout(x, self.dropout, self.training)

        x = x.transpose(0, 1)
        return x


class SentenceEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.max_num_words = args.max_num_words

        self.unigram_conv = nn.Conv1d(args.word_dim, args.word_dim, 1, stride=1, padding=0)
        self.bigram_conv  = nn.Conv1d(args.word_dim, args.word_dim, 2, stride=1, padding=1, dilation=2)
        self.trigram_conv = nn.Conv1d(args.word_dim, args.word_dim, 3, stride=1, padding=2, dilation=2)
        self.max_pool = nn.MaxPool2d((3, 1))
        # self.tanh = nn.Tanh()
        self.bilstm = nn.LSTM(input_size=args.word_dim,
                              hidden_size=args.word_dim // 2,
                              num_layers=2,
                              batch_first=True,
                              dropout=self.dropout,
                              bidirectional=True)
        self.concat = DynamicGRU(args.word_dim*3, args.d_model >> 1, bidirectional=True, batch_first=True)

    def forward(self, x, mask, node_pos, node_mask, adj_mat):
        length = mask.sum(dim=-1)
         
        #1
        words = x.permute(0, 2, 1) #128, 300, 20
        unigrams = torch.unsqueeze(self.unigram_conv(words), 2) # B x 512 x L
        bigrams  = torch.unsqueeze(self.bigram_conv(words), 2)  # B x 512 x L
        trigrams = torch.unsqueeze(self.trigram_conv(words), 2) # B x 512 x L
        words = words.permute(0, 2, 1) #128, 20, 300

        phrase = torch.squeeze(self.max_pool(torch.cat((unigrams, bigrams, trigrams), 2)))
        phrase = phrase.permute(0, 2, 1) #128, 20, 300

        self.bilstm.flatten_parameters()
        sentence, _ = self.bilstm(phrase)
        
        concate = torch.cat((words, phrase, sentence), 2)
        x = self.concat(concate, length, self.max_num_words)
        x = F.dropout(x, self.dropout, self.training)
        return x


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.dropout = args.dropout
        self.max_num_frames = args.max_num_frames

        self.anchors = generate_anchors(dataset=args.dataset)
        self.num_anchors = self.anchors.shape[0]
        widths = (self.anchors[:, 1] - self.anchors[:, 0] + 1)  # [num_anchors]
        centers = np.arange(0, args.max_num_frames)  # [video_len]
        start = np.expand_dims(centers, 1) - 0.5 * (np.expand_dims(widths, 0) - 1)
        end = np.expand_dims(centers, 1) + 0.5 * (np.expand_dims(widths, 0) - 1)
        self.proposals = np.stack([start, end], -1)  # [video_len, num_anchors, 2]

        # VideoEncoder
        self.video_encoder = VideoEncoder(args)

        # SentenceEncoder
        self.sentence_encoder = SentenceEncoder(args)

        #attentive graph
        self.atten = CoAttention(args.d_model, args.d_model)
        self.intra_v = CoAttention_intra(args.max_num_frames, args.d_model)
        self.intra_s = CoAttention_intra(args.max_num_words, args.d_model) 

        self.update_v = ConvGRUCell(args.d_model, args.d_model)
        self.update_s = ConvGRUCell(args.d_model, args.d_model)
        self.update_v_intra = ConvGRUCell(args.d_model, args.d_model)
        self.update_s_intra = ConvGRUCell(args.d_model, args.d_model)
        self.v2s = TanhAttention(args.d_model)
        
        self.rnn = DynamicGRU(args.d_model << 1, args.d_model >> 1, bidirectional=True, batch_first=True)

        self.fc_score = nn.Conv1d(args.d_model, self.num_anchors, kernel_size=1, padding=0, stride=1)
        self.fc_reg = nn.Conv1d(args.d_model, self.num_anchors << 1, kernel_size=1, padding=0, stride=1)

        # loss function
        self.criterion1 = nn.BCELoss()
        self.criterion2 = nn.SmoothL1Loss()

    def forward(self, frames, frame_mask, words, word_mask,
                label, label_mask, gt,
                node_pos, node_mask, adj_mat):
        frames_len = frame_mask.sum(dim=-1)

        frames = F.dropout(frames, self.dropout, self.training)
        words = F.dropout(words, self.dropout, self.training)

        frames = self.video_encoder(frames, frame_mask)
        x = self.sentence_encoder(words, word_mask, node_pos, node_mask, adj_mat)

        
        #attentive
        x1_att, x2_att, _, _ = self.atten(frames, x, node_mask)
        x1_m, x2_m = x1_att, x2_att#self.message_v(x1_att), self.message_s(x2_att)
        frames1 = self.update_v(x1_m, frames)
        x1 = self.update_s(x2_m, x)

        x1_m, _, a1, _ = self.intra_v(frames1, frames1, node_mask)
        x2_m, _, a2, _ = self.intra_s(x1, x1, node_mask)
        frames1 = self.update_v_intra(x1_m, frames1)
        x1 = self.update_s_intra(x2_m, x1)
        
        #layer 2
        #x1_att, x2_att, a1, a2 = self.atten(frames1, x1, node_mask)
        #x1_m, x2_m = x1_att, x2_att#self.message_v(x1_att), self.message_s(x2_att)
        #frames1 = self.update_v(x1_m, frames1)
        #x1 = self.update_s(x2_m, x1)
        #x1_m, _, a1, _ = self.intra_v(frames1, frames1, node_mask)
        #x2_m, _, a2, _ = self.intra_s(x1, x1, node_mask)
        #frames1 = self.update_v_intra(x1_m, frames1)
        #x1 = self.update_s_intra(x2_m, x1)
        
        #frames1, x1 = frames, x
        #a1, a2 = 1, 1
        # interactive
        x1 = self.v2s(frames1, x1, node_mask)
        x = torch.cat([frames1, x1], -1) #x1
        x = self.rnn(x, frames_len, self.max_num_frames)
        x = F.dropout(x, self.dropout, self.training)

        # loss
        predict = torch.sigmoid(self.fc_score(x.transpose(-1, -2))).transpose(-1, -2)
        # [batch, max_num_frames, num_anchors]
        reg = self.fc_reg(x.transpose(-1, -2)).transpose(-1, -2)
        reg = reg.contiguous().view(-1, self.args.max_num_frames * self.num_anchors, 2)
        # [batch, max_num_frames, num_anchors, 2]
        predict_flatten = predict.contiguous().view(predict.size(0), -1) * label_mask.float()
        cls_loss = self.criterion1(predict_flatten, label)
        # gt_box: [batch, 2]
        proposals = torch.from_numpy(self.proposals).type_as(gt).float()  # [max_num_frames, num_anchors, 2]
        proposals = proposals.view(-1, 2)
        if not self.training:
            ##if evaluation, open this
            batch_now = reg.shape[0]
            proposals = proposals.expand(batch_now, 800, 2)#1400,800
            predict_box = proposals
            predict_reg = reg # [nb, 2]
            refine_box = predict_box + predict_reg
            gt = gt.expand(800, batch_now, 2).transpose(0, 1).contiguous()
            reg_loss = self.criterion2(refine_box, gt.float())
            loss = cls_loss + 5e-3 * reg_loss #1e-3 5e-3
            predict_flatten = (predict.contiguous().view(predict.size(0), -1) * label_mask.float())
            
            # if train, open this
            # indices = torch.argmax(predict_flatten, -1)
            # predict_box = proposals[indices]  # [nb, 2]
            # predict_reg = reg[range(reg.size(0)), indices]  # [nb, 2]
            # refine_box = predict_box + predict_reg
            # reg_loss = self.criterion2(refine_box, gt.float())
            # if self.args.dataset == 'TACOS':
            #    loss = cls_loss + 5e-3 * reg_loss #1e-3 5e-3
            # else:
            #    loss = cls_loss + 1e-3 * reg_loss #1e-3 5e-3
        else:
            # indices = torch.argmax(label, -1)
            indices = torch.where(adj_mat > 0)
            batch_now = reg.shape[0]
            if self.args.dataset == 'TACOS':
                proposals = proposals.expand(batch_now, 800, 2)#1400,800
            else:
                proposals = proposals.expand(batch_now, 1400, 2)#1400,800
            predict_box = proposals[indices]  # [nb, 2]
            predict_reg = reg[indices]  # [nb, 2]
            refine_box = predict_box + predict_reg
            if self.args.dataset == 'TACOS':
                gt = gt.expand(800, batch_now, 2).transpose(0, 1).contiguous()
            else:         
                gt = gt.expand(1400, batch_now, 2).transpose(0, 1).contiguous()
            gt = gt[indices]
            reg_loss = self.criterion2(refine_box, gt.float())
            if self.args.dataset == 'TACOS':
                loss = cls_loss + 5e-3 * reg_loss #1e-3 5e-3
            else:
                loss = cls_loss + 1e-3 * reg_loss #1e-3 5e-3
        return refine_box, loss, predict_flatten, a1, a2
