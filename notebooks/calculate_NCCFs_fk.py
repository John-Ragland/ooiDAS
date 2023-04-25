from xrsignal import xrsignal
import xarray as xr
import ODLintake
from NI_tools.NI_tools import calculate
import numpy as np
from dask.distributed import Client, LocalCluster
import dask
from tqdm import tqdm
from DAStools import tools as dt

if __name__ == "__main__":
    dask.config.set({'temporary_directory': '/datadrive/tmp'})

    cluster = LocalCluster(n_workers=8)
    print(cluster.dashboard_link)
    client = Client(cluster)
    
    distance_chunk_idxs = [10]
    fcs = [5, 90]

    ds = ODLintake.open_ooi_DAS_SouthTx()


    for distance_chunk_idx in tqdm(distance_chunk_idxs):

        distance1, distance2 = distance_chunk_idx*3000, (distance_chunk_idx+1)*3000

        da_slice = ds.isel(distance=slice(distance1, distance2, 30))['RawData']
        
        tint = 1
        fs = 200
        xint=1
        dx=2
        c_min=1200
        c_max=1800

        da_fk_filt = np.real(dt.fk_filt(da_slice, tint, fs, xint, dx, c_min, c_max))

        da_pp = calculate.preprocess(da_fk_filt, dim='time', W=15, fcs=fcs)

        NCCF_me = calculate.compute_MultiElement_NCCF(da_pp, W=15, ref_idx=50).mean('time')
        #NCCF_me_pw = np.abs(calculate.compute_MultiElement_NCCF_PhaseWeight(da_pp, W=15).mean('time'))

        NCCF_me.to_netcdf(f'/datadrive/DAS/NCCFs/fk/{distance1}_{distance2}_{fcs[0]}-{fcs[1]}_middle_Hz_fk.nc')
        #NCCF_me_pw.to_netcdf(f'/datadrive/DAS/NCCFs/{distance1}_{distance2}_{fcs[0]}-{fcs[1]}Hz_pw_fk.nc')