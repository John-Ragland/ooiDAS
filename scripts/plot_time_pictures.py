import xarray as xr
import ODLintake
from xrsignal import xrsignal
from scipy import signal
import pandas as pd
import numpy as np
from dask.distributed import Client, LocalCluster
from matplotlib import pyplot as plt
from tqdm import tqdm
import dask

if __name__ == "__main__":
    dask.config.set({'temporary_directory': '/datadrive/tmp'})

    cluster = LocalCluster(n_workers=8)
    print(cluster.dashboard_link)
    client = Client(cluster)

    ds,geo = ODLintake.open_ooi_DAS_SouthTx_RawData()
    time_coord = pd.to_datetime(ds['RawDataTime'], unit='us')
    ds = ds.assign_coords({'time':time_coord})

    for k in tqdm(range(3264)):

        start_idx = k*3000*4
        end_idx = (k+1)*3000*4

        ds_slice = ds.isel({'distance': slice(0, 47500, 20),
                        'time': slice(start_idx, end_idx)})['RawData']

        b, a = signal.butter(4, [0.15, 0.27], btype='bandpass')
        ds_filt = xrsignal.filtfilt(ds_slice, dim='time', b=b, a=a)

        ds_c = xrsignal.hilbert_mag(ds_filt, dim='time')

        ds_downsamp = ds_c.isel({'time':slice(0,6000,10)})

        fig = plt.figure(figsize=(6,4))
        ax = plt.gca()
        (20*np.log10(ds_downsamp)).plot(x='distance',cmap='Blues', cbar_kwargs={'label':'dB'}, vmax=70, vmin=40, ax=ax)
        _=plt.xlabel('distance [km]')
        plt.title(time_coord[start_idx])

        fn = f'/datadrive/DAS/time_pictures/{k:04}.png'
        fig.savefig(fn, dpi=300, bbox_inches='tight')
        plt.close(fig)
