import os
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import json
import argparse
from torch.utils.data import DataLoader
from dataset.mnist_dataset import MnistDataset
from torch.optim import Adam
from tqdm import tqdm
from models.autoencoder_base import ConvAutoencoder, final_loss
from utils.external_utilities import compute_global_min_max, next_power_of_2, pad_to_power_of_2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_autoencoder(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    autoencoder_config = config['autoencoder_params']
    dataset_config = config['dataset_params']
    train_config = config['train_params']
    
    if os.path.exists(os.path.join(train_config['task_name'], train_config['min_max_name'])):
        with open(os.path.join(train_config['task_name'], train_config['min_max_name']), "r") as f:
            min_max_dict = json.load(f)
        
        global_min = min_max_dict["global_min"]
        global_max = min_max_dict["global_max"]
    
    else:
        # Load image paths using temp Dataset instance
        temp_dataset = MnistDataset(split="train", im_path=dataset_config['im_path'], global_min=0, global_max=1)
        image_paths = temp_dataset.images
        
        # Compute global min and max
        global_min, global_max = compute_global_min_max(image_paths)
        print(f"Global min: {global_min}, max: {global_max}")
        
        min_max_dict = {
            "global_min": float(global_min),
            "global_max": float(global_max)
        }
        
        with open(os.path.join(train_config['task_name'], train_config['min_max_name']), "w") as f:
            json.dump(min_max_dict, f)
    
    # Create the dataset
    mnist = MnistDataset('train', im_path=dataset_config['im_path'], global_min=global_min, global_max=global_max)
    mnist_loader = DataLoader(mnist, batch_size=autoencoder_config['batch_size'], shuffle=True, num_workers=4)
    
    # Instantiate the model
    model = ConvAutoencoder().to(device)
    model.train()
    
    # Create output directories
    if not os.path.exists(autoencoder_config['task_name']):
        os.mkdir(autoencoder_config['task_name'])
    
    # Load checkpoint if found
    if os.path.exists(os.path.join(autoencoder_config['task_name'],autoencoder_config['ckpt_name'])):
        print('Loading checkpoint as found one')
        model.load_state_dict(torch.load(os.path.join(autoencoder_config['task_name'],
                                                      autoencoder_config['ckpt_name']), map_location=device))
    # Specify training parameters
    num_epochs = autoencoder_config['num_epochs']
    optimizer = Adam(model.parameters(), lr=autoencoder_config['lr'])
    criterion = torch.nn.MSELoss()
    
    model.to(device)
    model.train()

    for epoch_idx in range(num_epochs):
        losses = []
        for im in tqdm(mnist_loader):
            optimizer.zero_grad()
            im = im.float().to(device)
            im = pad_to_power_of_2(im)

            recon, mu, log_var = model(im)
            mse_loss = criterion(recon, im)
            loss = final_loss(mse_loss, mu, log_var)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        print('Finished epoch:{} | Loss : {:.4f}'.format(
                epoch_idx + 1,
                np.mean(losses),
            ))

        torch.save(model.state_dict(), os.path.join(autoencoder_config['task_name'],
                                                autoencoder_config['ckpt_name']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for autoencoder training')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    train_autoencoder(args)
