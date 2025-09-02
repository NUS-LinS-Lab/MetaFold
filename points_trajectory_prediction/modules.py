
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # print(f"SelfAttention input x.shape:{x.shape}")
        batch_size, channels, height, width = x.shape
        # x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x = x.view(batch_size, channels, height * width).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        # return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)
        return attention_value.swapaxes(2, 1).view(batch_size, channels, height, width)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        # print(f"DoubleConv input x.shape:{x.shape}")
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        # print(f"Down input x.shape:{x.shape}, t.shape:{t.shape}")
        x = self.maxpool_conv(x)
        # print(x.device) # cuda
        # print(t.device) # cuda
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        # print(f"Up input x.shape:{x.shape}, t.shape:{t.shape}")
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=64, c_out=64, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        # self.sa1 = SelfAttention(128, 32)
        self.sa1 = SelfAttention(128, 16)
        self.down2 = Down(128, 256)
        # self.sa2 = SelfAttention(256, 16)
        self.sa2 = SelfAttention(256, 8)
        self.down3 = Down(256, 256)
        # self.sa3 = SelfAttention(256, 8)
        self.sa3 = SelfAttention(256, 4)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        # self.sa4 = SelfAttention(128, 16)
        self.sa4 = SelfAttention(128, 8)
        self.up2 = Up(256, 64)
        # self.sa5 = SelfAttention(64, 32)
        self.sa5 = SelfAttention(64, 16)
        self.up3 = Up(128, 64)
        # self.sa6 = SelfAttention(64, 64)
        self.sa6 = SelfAttention(64, 32)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        ).to(t.device)
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        ## print(f"Unet input x.shape:{x.shape}, t.shape:{t.shape}")
        t = t.unsqueeze(-1).type(torch.float)
        ## print(f"Unet input t.shape:{t.shape}")
        t = self.pos_encoding(t, self.time_dim)
        ## print(f"Unet input t.shape:{t.shape}")

        x1 = self.inc(x)
        ## print(f"x1.shape:{x1.shape}")
        x2 = self.down1(x1, t)
        ## print(f"x2.shape:{x2.shape}")
        x2 = self.sa1(x2)
        ## print(f"x2.shape:{x2.shape}")
        x3 = self.down2(x2, t)
        ## print(f"x3.shape:{x3.shape}")
        x3 = self.sa2(x3)
        ## print(f"x3.shape:{x3.shape}")
        x4 = self.down3(x3, t)
        ## print(f"x4.shape:{x4.shape}")
        x4 = self.sa3(x4)
        ## print(f"x4.shape:{x4.shape}")

        x4 = self.bot1(x4)
        ## print(f"x4.shape:{x4.shape}")
        x4 = self.bot2(x4)
        ## print(f"x4.shape:{x4.shape}")
        x4 = self.bot3(x4)
        ## print(f"x4.shape:{x4.shape}")

        x = self.up1(x4, x3, t)
        ## print(f"x.shape:{x.shape}")
        x = self.sa4(x)
        ## print(f"x.shape:{x.shape}")
        x = self.up2(x, x2, t)
        ## print(f"x.shape:{x.shape}")
        x = self.sa5(x)
        ## print(f"x.shape:{x.shape}")
        x = self.up3(x, x1, t)
        ## print(f"x.shape:{x.shape}")
        x = self.sa6(x)
        ## print(f"x.shape:{x.shape}")
        output = self.outc(x)
        ## print(f"UNet output.shape:{x.shape}")
        return output


if __name__ == '__main__':
    net = UNet(device="cuda")
    # net = UNet_conditional(num_classes=10, device="cpu")
    ## print(f"parameters sum:{sum([p.numel() for p in net.parameters()])}")
    x = torch.randn(3, 3, 64, 64)
    t = x.new_tensor([500] * x.shape[0]).long()
    ## print(f"input x.shape:{x.shape}, t.shape:{t.shape}")
    # y = x.new_tensor([1] * x.shape[0]).long()
    ## print(f"ouput net shape:{net(x, t).shape}")
    # print(net(x, t, y).shape)