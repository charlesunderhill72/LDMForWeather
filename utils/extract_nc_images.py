r"""
File to extract nc time steps from yearly nc files.
"""

import os
from tqdm import tqdm
import numpy as np
import xarray as xr

def extract_images(save_dir, nc_fname):
    assert os.path.exists(save_dir), "Directory {} to save images does not exist".format(save_dir)
    assert os.path.exists(nc_fname), "nc file {} does not exist".format(nc_fname)
    with xr.open_dataset(nc_fname) as ds:
        time_dim_size = len(ds['time'])
        chunk_size = 1
        date = ds["time"][0].values
        ystr = np.datetime64(date, 'Y').astype(str)
        for start_idx in tqdm(range(0, time_dim_size, chunk_size)):
            # Define the end index for the chunk
            end_idx = min(start_idx + chunk_size, time_dim_size)
            
            # Slice the data along the time dimension
            chunk_ds = ds.isel(time=slice(start_idx, end_idx))
            
            # Define the output file name
            if not os.path.exists(os.path.join(save_dir, ystr)):
                os.mkdir(os.path.join(save_dir, ystr))
                
            output_file = f"{save_dir}/{ystr}/output_chunk_{ystr}_{start_idx}.nc"
            
            # Save the chunk to a new NetCDF file
            chunk_ds.to_netcdf(output_file)
            print(f"Saved chunk {ystr}-{start_idx} to {output_file}")

        # Close the dataset
        ds.close()
            
            
if __name__ == '__main__':
    extract_images('data/train/images', 'data/train/z1979.nc')
    extract_images('data/train/images', 'data/train/z1980.nc')
    extract_images('data/train/images', 'data/train/z1981.nc')
    extract_images('data/train/images', 'data/train/z1983.nc')
    extract_images('data/test/images', 'data/test/z1985.nc')