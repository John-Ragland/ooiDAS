import xarray as xr
from xrsignal import xrsignal
import os
from dask.distributed import Client
import numpy as np
from scipy import signal
import dask

from matplotlib import pyplot as plt

# open zarr store
storage_options = {'account_name':'dasdata', 'account_key':os.environ['AZURE_KEY_dasdata']}
ds = xr.open_zarr('abfs://zarr/ooi_South_Tx.zarr/ooi_South_Tx.zarr', storage_options=storage_options)

das_psd = xrsignal.welch(ds['RawData'], dim='time', fs=200)
das_psd_distance = das_psd.mean('time')

das_psd_distance.to_netcdf('/datadrive/DAS/psd_distance.nc')