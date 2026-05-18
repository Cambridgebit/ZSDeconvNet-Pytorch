import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_conv=3):
        super().__init__()
        layers = []
        for i in range(n_conv):
            layers.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, depth=4, n_conv=3):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_ch
        for i in range(depth):
            out_ch = base_ch * (2**i)
            self.blocks.append(ConvBlock(ch, out_ch, n_conv))
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch
        self.out_ch = ch

    def forward(self, x):
        skips = []
        for block, pool in zip(self.blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)
        return x, skips


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, n_conv=3):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        layers = [nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True)]
        cur = out_ch
        for _ in range(n_conv - 1):
            next_ch = max(out_ch // 2, 1)
            layers.extend([nn.Conv2d(cur, next_ch, 3, padding=1), nn.ReLU(inplace=True)])
            cur = next_ch
        self.conv = nn.Sequential(*layers)
        self.out_ch = cur

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return self.conv(torch.cat([x, skip], dim=1))


class UNetStage(nn.Module):
    def __init__(self, in_ch=1, base_ch=32, depth=4, n_conv=3):
        super().__init__()
        self.encoder = Encoder(in_ch, base_ch, depth, n_conv)
        mid_ch = self.encoder.out_ch * 2
        self.mid = nn.Sequential(
            nn.Conv2d(self.encoder.out_ch, mid_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, self.encoder.out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoders = nn.ModuleList()
        cur_ch = self.encoder.out_ch
        for i in reversed(range(depth)):
            skip_ch = base_ch * (2**i)
            block = DecoderBlock(cur_ch, skip_ch, skip_ch, n_conv)
            self.decoders.append(block)
            cur_ch = block.out_ch
        self.out_ch = cur_ch

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.mid(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)
        return x


class DirectDeconvNet(nn.Module):
    def __init__(
        self,
        base_ch=32,
        depth=4,
        n_conv=3,
        upsample=True,
        detail_branch=False,
        detail_scale=0.2,
        gated_detail=True,
    ):
        super().__init__()
        self.upsample = upsample
        self.detail_branch = detail_branch
        self.detail_scale = detail_scale
        self.gated_detail = gated_detail
        self.last_detail = None
        self.stage2 = UNetStage(1, base_ch, depth, n_conv)
        self.refine1 = nn.Conv2d(self.stage2.out_ch, 128, 3, padding=1)
        self.refine2 = nn.Conv2d(128, 128, 3, padding=1)
        self.out2 = nn.Conv2d(128, 1, 3, padding=1)
        if self.detail_branch:
            self.detail1 = nn.Conv2d(128, 64, 3, padding=1)
            self.detail2 = nn.Conv2d(64, 1, 3, padding=1)
            if self.gated_detail:
                self.detail_gate = nn.Conv2d(128, 1, 3, padding=1)
                nn.init.zeros_(self.detail_gate.weight)
                nn.init.constant_(self.detail_gate.bias, -1.0)
            nn.init.zeros_(self.detail2.weight)
            nn.init.zeros_(self.detail2.bias)

    def detail_regularization(self):
        if self.last_detail is None:
            return None
        return self.last_detail.abs().mean()

    def forward(self, x):
        self.last_detail = None
        f2 = self.stage2(x)
        if self.upsample:
            f2 = F.interpolate(f2, scale_factor=2, mode="nearest")
        refined = F.relu(self.refine2(F.relu(self.refine1(f2))))
        out = self.out2(refined)
        if self.detail_branch:
            detail = self.detail2(F.relu(self.detail1(refined)))
            detail = detail - F.avg_pool2d(detail, kernel_size=5, stride=1, padding=2)
            if self.gated_detail:
                detail = torch.sigmoid(self.detail_gate(refined)) * detail
            self.last_detail = detail
            out = out + self.detail_scale * detail
        return F.relu(out)
