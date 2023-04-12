'''
calculate_std.py - calculate the standard deviation for different frequency bands
The goal of this is to get a first order analysis on the acoustic linking at different distances
'''

import xarray as xr
from DAStools import tools
from scipy import signal
import os
from dask.distributed import Client
from tqdm import tqdm

if __name__ == "__main__":
    client = Client()
    print(client)

    wbands = [(0.005, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5,0.9), (0.15, 0.2)]
    fband_text = ['p5_5', '5_20', '20_50', '50_90', '15_25']

    # open zarr store
    storage_options = {'account_name':'dasdata', 'account_key':os.environ['AZURE_KEY_dasdata']}
    ds = xr.open_zarr('abfs://zarr/ooi_South_Tx.zarr/ooi_South_Tx.zarr', storage_options=storage_options)

    stds = {}

    for k, wband in enumerate(tqdm(wbands)):
        b,a = signal.butter(4, wband, btype='bandpass')

        da_filt = tools.filtfilt(ds.RawData, dim='time', b=b, a=a)
        da_filt_std = da_filt.std('time')

        # compute filtered standard deviation for fband
        stds[fband_text[k]] = da_filt_std.compute()

    stds_x = xr.Dataset(stds)
    stds_x.to_zarr('/datadrive/DAS/stds/fband_stds.zarr', mode='w')

    client.close()