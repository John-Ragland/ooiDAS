import xarray as xr
from xrsignal import xrsignal
from dask.distributed import Client
from dask.distributed import LocalCluster, Client
import dask


if __name__ == '__main__':
    
    dask.config.set({'temporary_directory': '/datadrive/tmp'})

    cluster = LocalCluster(n_workers=8)
    print(cluster.dashboard_link)
    client = Client(cluster)

    # open zarr store
    storage_options = {'account_name':'dasdata'}
    ds = xr.open_zarr('az://zarr/ooi_South_Tx.zarr/ooi_South_Tx.zarr', storage_options=storage_options)

    da = ds['RawData']
    psd = xrsignal.welch(da, dim='time', fs=200, nperseg=1024)
    
    fn = '/datadrive/DAS/psd.nc'
    xr.Dataset({'psd':psd}).to_zarr(fn)

    #psd_distance = psd.mean('time')
    #psd_distance = psd_distance.compute()

    #fn = '/datadrive/DAS/psd_distance.nc'
    #psd_distance.to_netcdf(fn)

    client.close()