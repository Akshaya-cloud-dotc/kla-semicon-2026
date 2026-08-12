import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        
        y = inp + x * self.beta
        
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        
        return y + x * self.gamma

class NAFNet(nn.Module):
    def __init__(self, img_channel=1, width=16, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 1], dec_blk_nums=[1, 1, 1, 1], upscale=2):
        super().__init__()
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        
        self.upscale = upscale
        if self.upscale > 1:
            self.up = nn.Sequential(
                nn.Conv2d(width, img_channel * (upscale ** 2), 3, padding=1, bias=True),
                nn.PixelShuffle(upscale)
            )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            self.downs.append(
                nn.Conv2d(chan, chan * 2, 2, 2)
            )
            chan = chan * 2
            
        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )
        
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            
        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, inp):
        B, C, H, W = inp.shape
        H_pad = (self.padder_size - H % self.padder_size) % self.padder_size
        W_pad = (self.padder_size - W % self.padder_size) % self.padder_size
        if H_pad > 0 or W_pad > 0:
            inp_padded = F.pad(inp, (0, W_pad, 0, H_pad), mode='reflect')
        else:
            inp_padded = inp
            
        x = self.intro(inp_padded)
        
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
            
        x = self.middle_blks(x)
        
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            
        if self.upscale > 1:
            x = self.up(x)
            # Bilinear upsample the input to add as global residual connection
            inp_up = F.interpolate(inp_padded, scale_factor=self.upscale, mode='bilinear', align_corners=False)
            x = x + inp_up
            
            # Crop back to original dimensions scaled by upscale
            if H_pad > 0 or W_pad > 0:
                x = x[:, :, :H * self.upscale, :W * self.upscale]
        else:
            x = self.ending(x)
            x = x + inp_padded
            if H_pad > 0 or W_pad > 0:
                x = x[:, :, :H, :W]
            
        return torch.clamp(x, 0.0, 1.0)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Testing on device:", device)
    
    # Test same resolution
    print("\n--- Testing Same Resolution (upscale=1) ---")
    model_same = NAFNet(img_channel=1, width=16, middle_blk_num=1, enc_blk_nums=[1, 1, 1], dec_blk_nums=[1, 1, 1], upscale=1).to(device)
    x = torch.randn(2, 1, 128, 128).to(device)
    out_same = model_same(x)
    print("Input shape:", x.shape)
    print("Output shape:", out_same.shape)
    assert out_same.shape == x.shape
    
    # Test upscaling by 2
    print("\n--- Testing Upscale by 2 (upscale=2) ---")
    model_up = NAFNet(img_channel=1, width=16, middle_blk_num=1, enc_blk_nums=[1, 1, 1], dec_blk_nums=[1, 1, 1], upscale=2).to(device)
    out_up = model_up(x)
    print("Input shape:", x.shape)
    print("Output shape:", out_up.shape)
    assert out_up.shape == (2, 1, 256, 256)
    
    print("\nSelf-check passed successfully!")
