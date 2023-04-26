from xrsignal import xrsignal
import xarray as xr
import ODLintake
from NI_tools.NI_tools import calculate
import numpy as np
from dask.distributed import Client, LocalCluster
import dask
from tqdm import tqdm

if __name__ == "__main__":
    dask.config.set({'temporary_directory': '/datadrive/tmp2'})

    cluster = LocalCluster(n_workers=8, memory_limit='40GB')
    print(cluster.dashboard_link)
    client = Client(cluster)
    
    dxs = [5]
    ref_idx = 300
    fcs = [1, 90]

    dx = 5 # in index not meters
    
    da = ODLintake.open_ooi_DAS_SouthTx()['RawData']
    # rechunk time dimension to 30 s
    #da = da.chunk({'time': 6000})

    k = 9

    da_slice = da.isel(distance=slice((k)*3000, (k+1)*3000, dx))

    da_pp = calculate.preprocess(da_slice, dim='time', fcs=fcs, W=15)
    NCCFs_me = calculate.compute_MultiElement_NCCF_PhaseWeight(da_pp, W=15, ref_idx=ref_idx).compute()
    NCCF_me = NCCFs_me.mean('time')
    NCCF_me.to_netcdf(f'/datadrive/DAS/NCCFs/phase_weight/distance_idx_{(k)*3000}-{(k+1)*3000}_.nc')