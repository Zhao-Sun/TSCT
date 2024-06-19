import torch
import torch.nn as nn
import math

class temporal_gate(nn.Module):
    def __init__(self, in_channel, is_adapt, ks, b=1, gama=2):
        super(temporal_gate, self).__init__()
        kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
        if is_adapt:
            if kernel_size % 2:
                kernel_size = kernel_size
            else:
                kernel_size = kernel_size + 1
        else:
            kernel_size = ks
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size,
                              bias=False, padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        b, c, l = inputs.shape
        x = self.avg_pool(inputs)
        x = x.view([b, 1, c])
        x = self.conv(x)
        x = self.sigmoid(x)
        Wt = x.view([b, c, 1])
        outputs = Wt * inputs
        return outputs

class channel_gate(nn.Module):
    def __init__(self, channel, reduction, r):
        super(channel_gate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        if reduction:
            self.fc = nn.Sequential(
                nn.Linear(channel, channel // r, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // r, channel, bias=False),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(channel, channel, bias=False),
                nn.ReLU(inplace=True),
                nn.Sigmoid()
            )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        W_c = self.fc(y).view(b, c, 1)
        return x * W_c.expand_as(x)

class series_decomp_conv(nn.Module):

    def __init__(self, ks=None, ch_in=None, reduction=None, r=None, is_adapt=None):
        super(series_decomp_conv, self).__init__()
        self.conv = nn.Conv1d(ch_in, ch_in, groups=ch_in, kernel_size=ks, padding='same')
        self.channel_gate = channel_gate(ch_in, reduction, r)
        self.temporal_gate = temporal_gate(ch_in, is_adapt, ks)

    def forward(self, x):
        moving_mean = self.conv(x.permute(0, 2, 1))
        channel_map = self.channel_gate(moving_mean).permute(0, 2, 1)
        temporal_map = self.temporal_gate(channel_map)
        res = x - temporal_map
        return res, temporal_map


class TSCTNet(nn.Module):

    def __init__(self, ks=None, ch_in=None, reduction=None, r=None, seq_len=None, pred_len=None, is_adapt=None):
        super(TSCTNet, self).__init__()
        self.decompsition = series_decomp_conv(ks, ch_in, reduction, r, is_adapt)
        self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
        self.Linear_Trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.kernel_size = configs.kernel_size
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.reduction = configs.reduction
        self.r = configs.r
        self.is_adapt = configs.is_adapt
        self.TSCTNet = TSCTNet(self.kernel_size, self.channels, self.reduction, self.r, self.seq_len, self.pred_len, self.is_adapt)

    def forward(self, x):

        seq_last = torch.mean(x, dim=1, keepdim=True)
        x = x - seq_last
        x = self.TSCTNet(x)

        return x.permute(0,2,1) + seq_last
