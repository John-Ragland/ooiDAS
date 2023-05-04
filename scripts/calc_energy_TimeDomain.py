import xarray as xr
import numpy as np
import pandas as pd
import ODLintake
from xrsignal import xrsignal
import matplotlib.pyplot as plt
from scipy import signal
from dask.distributed import Client, LocalCluster
import hvplot.xarray
import dask
from DAStools import tools as dt

if __name__ == '__main__':
    dask.config.set({'temporary_directory': '/datadrive/tmp'})

    cluster = LocalCluster(n_workers=8)
    print(cluster.dashboard_link)
    client = Client(cluster)

    fcs = np.array([15, 28])
    ds = ODLintake.open_ooi_DAS_SouthTx()
    b,a = signal.butter(4, [fcs[0]/100, fcs[1]/100], btype='bandpass')
    da_filt = da_filt = xrsignal.filtfilt(ds['RawData'], b=b, a=a, dim='time')

    DAS_energy = dt.energy_TimeDomain(da_filt, time_dim='time')

    fn = f'/datadrive/DAS/DAS_energy_{fcs[0]}-{fcs[1]}.nc'
    DAS_energy.to_netcdf(fn)