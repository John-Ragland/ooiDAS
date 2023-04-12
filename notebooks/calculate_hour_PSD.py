import xarray as xr
from xrsignal import xrsignal
import pandas as pd
import dask
from dask.distributed import Client, LocalCluster
from tqdm import tqdm
if __name__ == "__main__":
    dask.config.set({'temporary_directory': '/datadrive/tmp'})

    cluster = LocalCluster(n_workers=8)
    print(cluster.dashboard_link)
    client = Client(cluster)

    # open zarr store
    storage_options = {'account_name': 'dasdata'}
    ds = xr.open_zarr('az://zarr/ooi_South_Tx.zarr/ooi_South_Tx.zarr',
                    storage_options=storage_options)
    da = ds['RawData']

    print('getting time coordinate...')
    time_coord_raw = pd.to_datetime(ds['RawDataTime'], unit='us')

    print('calculating psd computation tree....')
    fn = '/datadrive/DAS/psd_hour.zarr'
    # loop over each hour and caclulate psd
    for k in tqdm(range(55)):
        da_slice = da[:,k*200*3600:(k+1)*200*3600]
        psd = xrsignal.welch(da_slice, dim='time', fs=200, nperseg=1024).mean('time', keepdims=True)
        psd = psd.assign_coords(coords={'time':[time_coord_raw[k]]})
        psd_ds = xr.Dataset({'psd':psd})

        if k == 0:
            psd_ds.to_zarr(fn, mode='w-')
        else:
            psd_ds.to_zarr(fn, append_dim='time')