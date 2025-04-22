"""Test module for autoencoder"""

import unittest
import os
import sys
import yaml
import json
import torch
import argparse
import xarray as xr
import matplotlib.pyplot as plt
from models.autoencoder_base import ConvAutoencoder
from utils.external_utilities import next_power_of_2, pad_to_power_of_2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TestAutoencoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Read the config file
        with open(cls.config_path, 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
        
        cls.autoencoder_config = config['autoencoder_params']
        cls.dataset_config = config['dataset_params']
        cls.train_config = config['train_params']
        cls.test_config = config['test_params']
        
        cls.checkpoint_path = os.path.join(cls.autoencoder_config['task_name'],
                                           cls.autoencoder_config['ckpt_name'])
        
        cls.model = ConvAutoencoder().to(device)
        
        if os.path.exists(cls.checkpoint_path):
            cls.model.load_state_dict(torch.load(cls.checkpoint_path, map_location=device))
            cls.model.eval()
            cls.has_checkpoint = True
        else:
            cls.has_checkpoint = False

    def test_checkpoint_exists(self):
        """Check if the autoencoder checkpoint exists"""
        self.assertTrue(self.has_checkpoint, f"Checkpoint not found at {self.checkpoint_path}")

    def test_output_shape(self):
        """Ensure that autoencoder output shape matches input shape"""
        if not self.has_checkpoint:
            self.skipTest("Skipping shape test: no checkpoint available.")
        x = torch.randn(2, 2, 256, 256).to(device)
        x_recon, mu, log_var = self.model(x)
        self.assertEqual(x_recon.shape, x.shape)

    def visualize_reconstruction(self):
        if not self.has_checkpoint:
            print("Checkpoint not found. Cannot visualize reconstruction.")
            return
       ## Need to modify to test samples of generated latents
       ## But first test on an actual data file
       # if not os.path.exists(os.path.join(self.train_config['task_name'], 'samples', 'x0_1.png')):
        #    print("Sample not found. Cannot visualize reconstruction.")
         #   return
         
        in_path = os.path.join(self.test_config['im_path'], '1985', 'output_chunk_1985_0.nc')
        if not os.path.exists(in_path):
            print("Sample not found. Cannot visualize reconstruction.")
            return
        
        if os.path.exists(os.path.join(self.train_config['task_name'], self.train_config['min_max_name'])):
            with open(os.path.join(self.train_config['task_name'], self.train_config['min_max_name']), "r") as f:
                min_max_dict = json.load(f)
            
            global_min = min_max_dict["global_min"]
            global_max = min_max_dict["global_max"]
        
        im = xr.open_dataset(in_path)["z"].values
        im_tensor = torch.squeeze(torch.tensor(im, dtype=torch.float32))
        # Global min-max normalization to [0, 1]
        im_tensor = (im_tensor - global_min) / (global_max - global_min + 1e-8)
        sample = im_tensor
        # Convert input to [-1, 1] range.
        #sample = (2 * im_tensor) - 1
        sample = sample[None, :, :, :]
        sample = pad_to_power_of_2(sample).to(device)
        assert sample.shape == (1, 2, 256, 256)
        #sample = torch.randn(1, 2, 256, 256).to(device)
        with torch.no_grad():
            recon, _, _ = self.model(sample)

        fig, axs = plt.subplots(2, 2, figsize=(10, 5))
        for ch in range(2):
            axs[0, ch].imshow(sample[0, ch].cpu(), cmap='viridis', origin='lower')
            axs[0, ch].set_title(f"Original - Channel {ch}")
            axs[0, ch].axis('off')
            axs[1, ch].imshow(recon[0, ch].cpu(), cmap='viridis', origin='lower')
            axs[1, ch].set_title(f"Reconstruction - Channel {ch}")
            axs[1, ch].axis('off')

        plt.tight_layout()
        if not os.path.exists(os.path.join(self.test_config['task_name'], self.test_config['vis_name'])):
            os.mkdir(os.path.join(self.test_config['task_name'], self.test_config['vis_name']))
        
        out_path = os.path.join(self.test_config['task_name'], self.test_config['vis_name'], "recon_test.png")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()

# Manual entry point
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_path', type=str, default='config/default.yaml')
    parser.add_argument('--vis', action='store_true', help='Manually visualize output')
    args, unknown = parser.parse_known_args()

    # Inject config path into class before test discovery
    TestAutoencoder.config_path = args.config_path

    if args.vis:
        # Manually visualize
        test_case = TestAutoencoder(methodName='visualize_reconstruction')
        test_case.setUpClass()
        test_case.visualize_reconstruction()
    else:
        # Run standard unit tests
        unittest.main(argv=[sys.argv[0]] + unknown)

if __name__ == '__main__':
    main()

