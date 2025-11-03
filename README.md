Latent Diffusion Model Implementation in PyTorch to Develop Generative Images of Global Geopotential
========

This repository modifies explainingai's [DDPM](https://arxiv.org/abs/2006.11239) to function as a Latent Diffusion Model [LDM](https://arxiv.org/pdf/2112.10752) with training and sampling methods of DDPM and unet architecture mimicking the stable diffusion unet used in diffusers library from huggingface from scratch. The LDM modification involves using an autoencoder to downscale larger images so the unet can be trained on larger images without enforcing large memory requirements. The adaptation was made so the model can be trained on higher resolution geopotential data.

<!--## DDPM Explanation Videos
<a href="https://www.youtube.com/watch?v=H45lF4sUgiE">
   <img alt="DDPM Math Video" src="https://github.com/explainingai-code/DDPM-Pytorch/assets/144267687/27627399-ca3e-4491-a3f3-34d36e05b9cb"
   width="300">
</a><a href="https://www.youtube.com/watch?v=vu6eKteJWew">
   <img alt="DDPM Implementation Video" src="https://github.com/explainingai-code/DDPM-Pytorch/assets/144267687/ebcf6a07-c84a-4219-bb2a-31fc7d60cffa"
   width="300">
</a><img width="1400" height="1000" alt="example_data_image" src="https://github.com/user-attachments/assets/f1996a56-5cc7-4957-832e-e36c22ef8d21" />
.-->

## Example Image Used for Training Data and Generative Target

<img width="1122" height="268" alt="example_data_image" src="https://github.com/user-attachments/assets/c089f313-7eef-4584-bdb2-d488e5b2fac4" />

<!--
## Data preparation
For setting up the mnist dataset:

Follow - https://github.com/explainingai-code/Pytorch-VAE#data-preparation
-->

<!--
## Training on your own images
For this one would need to make the following changes
* Put the image files in a folder created within the repo root (example: data/images/*.png ). The data folder should only have one directory 'images'
* Comment https://github.com/explainingai-code/DDPM-Pytorch/blob/main/dataset/mnist_dataset.py#L42 as this is only valid for mnist
* Update the expected number of channels here and image dimensions(assumed square images) here - https://github.com/explainingai-code/DDPM-Pytorch/blob/main/config/default.yaml#L10
* Change the config path here to point to 'data' directory('data' and not 'data/images') - https://github.com/explainingai-code/DDPM-Pytorch/blob/main/config/default.yaml#L2
* Right now the code has been written for picking up png files in mnist data directory format, so I assume there are subdirectories inside the directory mentioned in config and these sub-directories have .png files. 
This would work if you have .png files. If the images are of other formats or combination of different formats then one would have to change the load_images function correspondingly here - https://github.com/explainingai-code/DDPM-Pytorch/blob/main/dataset/mnist_dataset.py#L29C9-L29C9
* As of now code is written assuming square images, if thats not the case then just changing the dimensions to desired one during sampling should work - https://github.com/explainingai-code/DDPM-Pytorch/blob/main/tools/sample_ddpm.py#L20
-->

# Quickstart
* Create a new conda environment with python 3.8 then run below commands
* ```git clone https://github.com/charlesunderhill72/LDMForWeather.git```
* ```cd LDMForWeather```
* ```pip install -r requirements.txt```
* For training/sampling use the below commands passing the desired configuration file as the config argument in case you want to play with it. 
* ```python -m tools.train_ddpm``` for training ddpm
* ```python -m tools.sample_ddpm``` for generating images

## Configuration
* ```config/default.yaml``` - Allows you to play with different components of the ddpm, autoencoder, and training/sampling parameters. 


## Output 
Outputs will be saved according to the configuration present in yaml files.

For every run a folder of ```task_name``` key in config will be created

During training of DDPM the following output will be saved 
* Latest Model checkpoint in ```task_name``` directory

During sampling the following output will be saved
* Sampled image grid for all timesteps in ```task_name/samples/*.png``` 

## Citations
```
@misc{ho2020denoising,
      title={Denoising Diffusion Probabilistic Models}, 
      author={Jonathan Ho and Ajay Jain and Pieter Abbeel},
      year={2020},
      eprint={2006.11239},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
@misc{rombach2022highresolutionimagesynthesislatent,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2022},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2112.10752}, 
}
```
