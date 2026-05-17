import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    """
    A Convolutional Autoencoder for Image Compression.
    It takes an image tensor of shape (B, C, H, W) and compresses it into a latent tensor.
    """
    def __init__(self, in_channels=3, latent_channels=16):
        super(ConvAutoencoder, self).__init__()
        
        # ENCODER
        # Using strides to downsample the spatial dimensions (H, W) by a factor of 4 (2x2)
        # and compressing the channels to the bottleneck size (latent_channels).
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1), # -> (B, 32, H/2, W/2)
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (B, 64, H/4, W/4)
            nn.ReLU(True),
            nn.Conv2d(64, latent_channels, kernel_size=3, stride=1, padding=1) # -> (B, latent_channels, H/4, W/4)
            # We use Sigmoid at the end of encoder to bound the latent representation between 0 and 1.
            # This makes uniform quantization to uint8 (0-255) much easier and more stable.
            ,nn.Sigmoid() 
        )
        
        # DECODER
        # Using ConvTranspose2d to upsample the spatial dimensions back to original size.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (B, 32, H/2, W/2)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (B, C, H, W)
            nn.Sigmoid() # Output pixels are between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
