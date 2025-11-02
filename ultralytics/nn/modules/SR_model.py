import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class Decoder(nn.Module):
    def __init__(self, c1, c2):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(c1, c1 // 2, 1, bias=False)
        self.conv2 = nn.Conv2d(c2, c2 // 2, 1, bias=False)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d((c1 + c2) // 2, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       # BatchNorm(256),
                                       nn.ReLU(),
                                       # nn.Dropout(0.5),
                                       nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                       # BatchNorm(128),
                                       nn.ReLU(),
                                       # nn.Dropout(0.1),
                                       nn.Conv2d(128, 64, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x, low_level_feat, factor):
        # x, low_level_feat = input[-2],input[-1]
        low_level_feat = self.conv1(low_level_feat)
        # low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        # x = F.interpolate(x, size=[i*2 for i in low_level_feat.size()[2:]], mode='bilinear', align_corners=True)
        # low_level_feat = F.interpolate(low_level_feat, size=[i*2 for i in low_level_feat.size()[2:]], mode='bilinear', align_corners=True)
        # x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = self.relu(x)
        x = F.interpolate(x, size=[i * (factor // 2) for i in low_level_feat.size()[2:]], mode='bilinear',
                          align_corners=True)
        if factor > 1:
            low_level_feat = F.interpolate(low_level_feat, size=[i * (factor // 2) for i in low_level_feat.size()[2:]],
                                           mode='bilinear', align_corners=True)
        # x = self.pixel_shuffle(x)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, SynchronizedBatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def make_model(args, parent=False):
    return EDSR(args)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class EDSR(nn.Module):
    def __init__(self, num_channels=3,input_channel=64, factor=4, width=64, depth=16, kernel_size=3, conv=default_conv):
        super(EDSR, self).__init__()

        n_resblock = depth
        n_feats = width
        kernel_size = kernel_size
        scale = factor
        act = nn.ReLU()

        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = common.MeanShift(1.0, rgb_mean, rgb_std)

        # define head module
        m_head = [conv(input_channel, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=1.
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, num_channels, kernel_size)
        ]

        # self.add_mean = common.MeanShift(1.0, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


class EDSRConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EDSRConv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            )

        self.residual_upsampler = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            )

    def forward(self, input):
        return self.conv(input)+self.residual_upsampler(input)


class SR_model(nn.Module):
    def __init__(self, ch, c1=128, c2=512,factor=2, sync_bn=True, freeze_bn=False):
        super(SR_model, self).__init__()

        self.sr_decoder = Decoder(c1,c2)
        self.edsr = EDSR(num_channels=ch,input_channel=64, factor=factor)
        self.factor = factor


        # self.freeze_bn = freeze_bn

    def forward(self, low_level_feat,x):
        x_sr= self.sr_decoder(x, low_level_feat,self.factor)
        x_sr_up = self.edsr(x_sr)


        return x_sr_up



if __name__ == "__main__":
    model = SR_model(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


