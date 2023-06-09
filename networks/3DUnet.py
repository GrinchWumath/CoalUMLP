import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        self.enc1 = nn.Sequential(
            DoubleConv(in_channels, 64),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.enc2 = nn.Sequential(
            DoubleConv(64, 128),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.enc3 = nn.Sequential(
            DoubleConv(128, 256),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.enc4 = nn.Sequential(
            DoubleConv(256, 512),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.middle = DoubleConv(512, 1024)
        self.dec4 = nn.Sequential(
            nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2),
            DoubleConv(1024, 512)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2),
            DoubleConv(512, 256)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),
            DoubleConv(256, 128)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            DoubleConv(128, 64)
        )
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        mid = self.middle(enc4)
        dec4 = self.dec4(mid)
        dec4 = torch.cat([enc4, dec4], dim=1)
        dec3 = self.dec3(dec4)
        dec3 = torch.cat([enc3, dec3], dim=1)
        dec2 = self.dec2(dec3)
        dec2 = torch.cat([enc2, dec2], dim=1)
        dec1 = self.dec1(dec2)
        dec1 = torch.cat([enc1, dec1], dim=1)
        out = self.final_conv(dec1)
        return out