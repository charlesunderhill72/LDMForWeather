import glob
import os

import torch
import xarray as xr
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

class MnistDataset(Dataset):
    r"""
    Nothing special here. Just a simple dataset class for mnist images.
    Created a dataset class rather using torchvision to allow
    replacement with any other image dataset
    """
    def __init__(self, split, im_path, global_min=None, global_max=None, im_ext='nc'):
        r"""
        Init method for initializing the dataset properties
        :param split: train/test to locate the image files
        :param im_path: root folder of images
        :param global_min/max: 
        :param im_ext: image extension. assumes all
        images would be this type.
        """
        self.split = split
        self.im_ext = im_ext
        self.images, self.labels = self.load_images(im_path)
        
        # Set global min/max if provided (assume computed externally)
        self.global_min = global_min
        self.global_max = global_max

        if self.global_min is None or self.global_max is None:
            raise ValueError("global_min and global_max must be provided for global normalization.")
    
    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        :param im_path:
        :return:
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        labels = []
        for d_name in tqdm(os.listdir(im_path)):
            for fname in glob.glob(os.path.join(im_path, d_name, '*.{}'.format(self.im_ext))):
                ims.append(fname)
               # labels.append(int(d_name))
        print('Found {} images for split {}'.format(len(ims), self.split))
        return ims, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        im = xr.open_dataset(self.images[index])["z"].values
        im_tensor = torch.squeeze(torch.tensor(im, dtype=torch.float32))
        
        # Global min-max normalization to [0, 1]
        im_tensor = (im_tensor - self.global_min) / (self.global_max - self.global_min + 1e-8)
        
        # Convert input to [-1, 1] range.
        im_tensor = (2 * im_tensor) - 1
        return im_tensor
    