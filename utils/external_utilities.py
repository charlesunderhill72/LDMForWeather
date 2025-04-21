"""
A collection of external utilities.
"""
import numpy as np
import torch.nn.functional as F
import tqdm
import torch
import xarray as xr

# Function to calculate the next power of 2 greater than or equal to the size
def next_power_of_2(x):
    return 2 ** np.ceil(np.log2(x)).astype(int)

# Function to pad an image to the next power of 2 in both dimensions
def pad_to_power_of_2(x):
    batch_size, channels, height, width = x.size()

    # Calculate the next power of 2 for height and width
    target_height = next_power_of_2(height)
    target_width = next_power_of_2(width)
    
    target = max(target_height, target_width)

    # Calculate padding needed
    pad_height = target - height
    pad_width = target - width

    # Padding should be applied symmetrically
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Apply padding
    padded_x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    return padded_x

def undo_padding(x, height, width):
    batch_size, channels, im_height, im_width = x.size()

    # Calculate padding to remove
    pad_height = im_height - height
    pad_width = im_width - width

    # Padding should be applied symmetrically
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    
    # Slice the tensor to remove padding
    return x[:, :, pad_top: -pad_bottom, pad_left: -pad_right]


# This needs to be fixed so that it computes min/max of each channel
def compute_global_min_max(images):
    r"""
    Iterates over image attributes from Dataset class and returns global min 
    and max.
    """
    global_min = float('inf')
    global_max = float('-inf')

    for path in tqdm(images, desc="Computing global min/max"):
        im = xr.open_dataset(path)["z"].values.astype("float32")
        current_min = im.min()
        current_max = im.max()
        global_min = min(global_min, current_min)
        global_max = max(global_max, current_max)
    
    return global_min, global_max


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
    print(undo_padding(x, 91, 180).shape)
    


