import torch
import torchvision
import argparse
import yaml
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm
from models.unet_base import Unet
from models.autoencoder_base import ConvAutoencoder
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.external_utilities import undo_padding


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, autoencoder, scheduler, train_config, model_config, diffusion_config, global_min, global_max):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    xt = torch.randn((train_config['num_samples'],
                      model_config['im_channels'],
                      model_config['im_size'],
                      model_config['im_size'])).to(device)
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        
        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        
    xt, x0_pred = autoencoder.decoder(xt), autoencoder.decoder(x0_pred)
    xt, x0_pred = undo_padding(xt, 91, 180), undo_padding(x0_pred, 91, 180)
    
    
    # Save x0
    ims = torch.clamp(xt, -1., 1.).detach().cpu()
    ims = (ims + 1) / 2
    ims = (global_max - global_min + 1e-8)*ims + global_min
    #grid = make_grid(ims, nrow=train_config['num_grid_rows'])
    #img = torchvision.transforms.ToPILImage()(grid)
    if not os.path.exists(os.path.join(train_config['task_name'], 'samples')):
        os.mkdir(os.path.join(train_config['task_name'], 'samples'))
    
    for sample_idx in range(ims.shape[0]):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # One row, two columns
        for ch in range(2):
            data = ims[sample_idx, ch].numpy()
            axs[ch].imshow(data, cmap='viridis', origin='lower')
            axs[ch].set_title(f"Channel {ch}")
            axs[ch].axis('off')
        
        plt.suptitle(f"Generated Sample {sample_idx}")
        plt.tight_layout()
        out_path = os.path.join(train_config['task_name'], 'samples', f"x0_{i}.png")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        #img.save(os.path.join(train_config['task_name'], 'samples', 'x0_{}.png'.format(i)))
        #img.close()


def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']
    autoencoder_config = config['autoencoder_params']
    
    if os.path.exists(os.path.join(train_config['task_name'], train_config['min_max_name'])):
        with open(os.path.join(train_config['task_name'], train_config['min_max_name']), "r") as f:
            min_max_dict = json.load(f)
        
        global_min = min_max_dict["global_min"]
        global_max = min_max_dict["global_max"]
    
    # Load autoencoder with checkpoint
    autoencoder = ConvAutoencoder().to(device)
    autoencoder.load_state_dict(torch.load(os.path.join(autoencoder_config['task_name'],
                                                  autoencoder_config['ckpt_name']), map_location=device))
    autoencoder.eval()
    
    # Load model with checkpoint
    model = Unet(model_config).to(device)
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                  train_config['ckpt_name']), map_location=device))
    model.eval()
    
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    with torch.no_grad():
        sample(model, autoencoder, scheduler, train_config, model_config, diffusion_config, global_min, global_max)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    infer(args)
