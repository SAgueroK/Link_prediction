import os
import time

import numpy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torch.utils.data as Data
from ptflops import get_model_complexity_info

import config
from SelfAttention_Family import AttentionLayer, ProbAttention

FACTOR = 3
NUM_CLASS = 1
NUM_LAYER = 2
TOP_K = 1
NUM_KERNELS = 2
WINDOW_SIZE = 7
N_HEADS = 2
ATTN_NUM_LAYER = 1
period_weight = np.zeros([50])
D_MODEL = 1
SEQ_LEN = 2


def FFT_for_Period(x, k=TOP_K):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    # 这里对所有batch的频率求平均  代替了对每一个输入数据的频率求前k大的， 就是为了节省时间，前提是同一batch的数据周期具有相似性
    frequency_list[0] = -1
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    for i, x in enumerate(period):
        if x < 50:
            period_weight[x] += k - i
    return period, abs(xf).mean(-1)[:, top_list]


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class TimesBlock(nn.Module):
    def __init__(self, seq_len, top_k, d_model, num_kernels):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.k = top_k
        # parameter-efficient design
        self.conv = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if period != 0 and self.seq_len % period != 0:
                length = ((self.seq_len // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - self.seq_len), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :self.seq_len, :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class RTime(nn.Module):
    def __init__(self, seq_len, d_model, num_layer=NUM_LAYER, top_k=TOP_K, num_kernels=NUM_KERNELS,
                 window_size=WINDOW_SIZE):
        self.seq_len = seq_len
        self.d_model = d_model
        super(RTime, self).__init__()
        time_list = []
        for i in range(num_layer):
            time_list.append(TimesBlock(seq_len, top_k, d_model, num_kernels))
        decomp_list = []
        for i in range(num_layer):
            decomp_list.append(series_decomp(window_size))
        self.time_layer_list = nn.ModuleList(time_list)
        self.decomp_layer_list = nn.ModuleList(decomp_list)
        self.linear = nn.Linear(NUM_LAYER + 1, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # if m.bias is not None:
                #     nn.init.constant_(m.bias, 0)

    # seasonal data 为 seq*d_model 维度
    def forward(self, seasonal_data):
        res = None
        for x, _ in enumerate(self.time_layer_list):
            seasonal_data = self.time_layer_list[x](seasonal_data)
            seasonal_data, tmp_trend_data = self.decomp_layer_list[x](seasonal_data)
            if res is None:
                res = tmp_trend_data
            else:
                res = torch.concatenate([res, tmp_trend_data], dim=0)
        res = torch.concatenate([res, seasonal_data], dim=0)
        res = res.view(NUM_LAYER + 1, self.seq_len, self.d_model)
        res = res.permute(1, 0, 2)
        res = res.permute(0, 2, 1)
        res = self.linear(res)
        return res
