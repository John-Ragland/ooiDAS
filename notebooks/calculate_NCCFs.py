from xrsignal import xrsignal
import xarray as xr
import ODLintake
from NI_tools.NI_tools import calculate
import numpy as np
from dask.distributed import Client, LocalCluster
import dask
from tqdm import tqdm

if __name__ == "__main__":
    dask.config.set({'temporary_directory': '/datadrive/tmp'})

    cluster = LocalCluster(n_workers=8)
    print(cluster.dashboard_link)
    client = Client(cluster)
    
    depth_chunk_idxs = [9,]
    ref_idxs = [300]
    dxs = [5]


    fcs = [15, 25]

    dx = 10 # in index not meters
    
    da = ODLintake.open_ooi_DAS_SouthTx()['RawData']
    # rechunk time dimension to 30 s
    #da = da.chunk({'time': 6000})

    for depth_chunk_idx in tqdm(depth_chunk_idxs):
        for ref_idx in ref_idxs:
            for dx in dxs:
                da_slice = da.isel(distance=slice(depth_chunk_idx*3000, (depth_chunk_idx+1)*3000, dx))
                
                da_pp = calculate.preprocess(da_slice, dim='time', fcs=fcs, W=15)

                NCCF_me = calculate.compute_MultiElement_NCCF(da_pp, W=15, ref_idx=ref_idx).mean('time')
                #NCCF_me_pw = np.abs(calculate.compute_MultiElement_NCCF_PhaseWeight(da_pp, W=15, ref_idx=ref_idx).mean('time'))

                NCCF_me.to_netcdf(f'/datadrive/DAS/NCCFs/{depth_chunk_idx*3000}_{(depth_chunk_idx+1)*3000}_{fcs[0]}-{fcs[1]}Hz_dx{dx}_refidx{ref_idx}.nc')
                #NCCF_me_pw.to_netcdf(f'/datadrive/DAS/NCCFs/{depth_chunk_idx*3000}_{(depth_chunk_idx+1)*3000}_{fcs[0]}-{fcs[1]}Hz_pw_dx{dx}_refidx{ref_idx}.nc')