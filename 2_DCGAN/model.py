import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1), # N x C x 64 x 64 --> N x C x 32 x 32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),                        # 32x32 --> 16x16
            self._block(features_d * 2, features_d * 4, 4, 2, 1),                        # 16x16 --> 8x8
            self._block(features_d * 4, features_d * 8, 4, 2, 1),                        # 8x8 --> 4x4
            nn.Conv2d(features_d * 8, 1, 4, 2, 0), # 1x1
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels, 
                kernel_size,
                stride,
                padding,
                bias = False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2), 
        )       
    
    def forward(self, x):
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super().__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, features_g * 16, kernel_size=4, stride=1, padding=0), # N x 100 x 1 x 1 --> N x f_g*16 x 4 x 4
            self._block(features_g * 16, features_g * 8, kernel_size=4, stride=2, padding=1), # 8 x 8
            self._block(features_g * 8, features_g * 4, kernel_size=4, stride=2, padding=1), # 16x16
            self._block(features_g * 4, features_g * 2, kernel_size=4, stride=2, padding=1), # 32x32
            nn.ConvTranspose2d(features_g * 2, channels_img, 4, 2, 1), # 1x1
            nn.Tanh(),
        )

    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels, 
                kernel_size,
                stride,
                padding,
                bias = False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.gen(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

            
def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    model_d = Discriminator(in_channels, 8)
    initialize_weights(model_d)
    assert model_d(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    model_g = Generator(z_dim, in_channels, 8)
    z = torch.randn((N, z_dim, 1, 1))
    initialize_weights(model_g)
    assert model_g(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("Success: tests passed!")


if __name__ == "__main__":
    test()
