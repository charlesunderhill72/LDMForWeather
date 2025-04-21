import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        
        # Encoder backbone
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # 256 → 128
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),           # 128 → 64
            nn.ReLU(),
        )
        
        # Separate convs for mu and log_var
        self.conv_mu = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)       # Output: mu
        self.conv_logvar = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)   # Output: log_var
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # 64 → 128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),           # 128 → 256
            nn.ReLU(),
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        
        mu = self.conv_mu(x)         # Mean of latent distribution
        log_var = self.conv_logvar(x)  # Log variance

        # Don't sample — just use mu as deterministic latent
        z = mu
        
        x_recon = self.decoder(z)

        return x_recon, mu, log_var



# Regularization term to enforce standard normal latent space
def final_loss(mse_loss, mu, log_var):
    """
    Total loss is MSE loss of reconstructed image + KL Divergence 
    for a standard normal prior (mean=0, variance=1) in the latent
    space.
    """
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return kl_loss + mse_loss


if __name__ == '__main__':
    x = torch.randn(4, 2, 256, 256)
    model = ConvAutoencoder()
    reconstructed, latent, _ = model(x)
    
    print("Latent shape: ", latent.shape)         # Should be [4, 2, 64, 64]
    print("Reconstructed shape: ", reconstructed.shape)  # Should be [4, 2, 128, 256]



