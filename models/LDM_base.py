import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentDiffusionModel(nn.Module):
    def __init__(self, autoencoder, unet, noise_scheduler, beta_schedule='linear'):
        super().__init__()
        self.encoder = autoencoder.encoder
        self.decoder = autoencoder.decoder
        self.unet = unet
        self.noise_scheduler = noise_scheduler
        self.beta_schedule = beta_schedule

    def forward(self, x, t):
        with torch.no_grad():
            z = self.encoder(x)  # [B, 2, 64, 64]

        # Apply noise to the latent code
        noise = torch.randn_like(z)
        z_noisy = self.noise_scheduler.add_noise(z, noise, t)

        # Predict noise
        noise_pred = self.unet(z_noisy, t)

        # Regularize the predicted noise to prevent high variance
        noise_loss = F.mse_loss(noise_pred, noise)

        return noise_pred, noise_loss

    def sample(self, num_steps=1000, shape=(1, 2, 64, 64), device='cuda'):
        z = torch.randn(shape).to(device)
        for t in reversed(range(num_steps)):
            t_tensor = torch.full((z.size(0),), t, device=z.device, dtype=torch.long)
            noise_pred = self.unet(z, t_tensor)
            z = self.noise_scheduler.step(z, noise_pred, t)
        x_recon = self.decoder(z)
        return x_recon
    
    
