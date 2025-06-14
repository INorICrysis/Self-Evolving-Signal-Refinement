import torch
import torch.nn as nn

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义 UNet 模型
class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_rate=0):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels, dropout_rate):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            )

        self.enc1 = nn.Sequential(CBR(in_channels, 16, dropout_rate))
        self.enc2 = nn.Sequential(CBR(16, 32, dropout_rate))
        self.enc3 = nn.Sequential(CBR(32, 64, dropout_rate))
        self.enc4 = nn.Sequential(CBR(64, 128, dropout_rate))

        self.center = nn.Sequential(CBR(128, 256, dropout_rate))

        self.dec4 = CBR(256, 128, dropout_rate)
        self.dec3 = CBR(128, 64, dropout_rate)
        self.dec2 = CBR(64, 32, dropout_rate)
        self.dec1 = CBR(32, 16, dropout_rate)

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        center = self.center(self.pool(enc4))

        dec4 = self.dec4(torch.cat([self.upconv4(center), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.upconv3(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.upconv2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.upconv1(dec2), enc1], dim=1))

        return self.final(dec1)
