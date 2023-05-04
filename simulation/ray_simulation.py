import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import arlpy.uwapm as pm
import arlpy.plot as aplt
import numpy as np
from oceans.sw_extras import sw_extras as sw
from pyat_tools import pyat_tools
from scipy import signal
import scipy
from tqdm import tqdm
import ODLintake
from scipy import interpolate
from tqdm import tqdm
import pickle

fn = '/datadrive/DAS/south_DAS_latlondepth.txt'
geo = pd.read_csv(fn)
geo['distance'] = geo['index']*2.0419046878814697

hycom = xr.open_mfdataset(
    '/datadrive/HYCOM_data/DAS_ooi_South_Tx/*.nc', decode_times=False)

for k in tqdm(range(1, 15)):
    arrivals_forward = []
    arrivals_backward = []
    
    idx1 = (geo['index']-k*3000).abs().argmin()
    idx2 = (geo['index']-(k+1)*3000).abs().argmin()

    lat = geo.loc[idx1:idx2].lat.mean()
    lon = geo.loc[idx1:idx2].lon.mean()

    # Get SSP
    sspx = sw.soundspeed(hycom.salinity, hycom.water_temp, hycom.depth).sel(
        {'lat': lat, 'lon': lon}, method='nearest')
    ssp = pyat_tools.convert_SSP_arlpy(sspx, 0)[1:, :]
    ssp = np.vstack((ssp, np.array([5000, ssp[-1, 1]])))

    # Get Bathy
    bathy_pd = geo.loc[idx1:idx2][['distance', 'depth']]
    bathy_pd['depth'] = bathy_pd['depth']*-1
    bathy_pd['distance'] = bathy_pd['distance'] - \
        bathy_pd.loc[idx1+1500]['distance']
    bathy_pd['distance'] = bathy_pd['distance']
    bathy = np.array(bathy_pd)

    bathy_backward = bathy[:1501, :]
    bathy_backward[:, 0] = -1*bathy_backward[:, 0]
    bathy_backward = np.flip(bathy_backward, axis=0)
    bathy_forward = bathy[1500:-1, :]

    depth_interp_forward = interpolate.interp1d(
        bathy_forward[:, 0], bathy_forward[:, 1], kind='cubic')
    depth_interp_backward = interpolate.interp1d(
        bathy_backward[:, 0], bathy_backward[:, 1], kind='cubic')

    ranges = np.linspace(5,2999,50)

    for n in range(len(ranges)):
        # Define environment
        env_forward = pm.create_env2d(
            depth=bathy_forward,
            soundspeed=ssp,
            bottom_soundspeed=1450,
            bottom_density=1200,
            bottom_absorption=1.0,
            tx_depth=bathy_forward[0, 1]-0.01,
            rx_range=ranges[n],
            rx_depth=depth_interp_forward(ranges[n])-0.01
        )

        # Define environment
        env_backward = pm.create_env2d(
            depth=bathy_backward,
            soundspeed=ssp,
            bottom_soundspeed=1450,
            bottom_density=1200,
            bottom_absorption=1.0,
            tx_depth=bathy_backward[0, 1]-0.01,
            rx_range=ranges[n],
            rx_depth=depth_interp_backward(ranges[n])-0.01
        )

        try:
            arrivals_forward.append(pm.compute_arrivals(env_forward))
        except ValueError:
            arrivals_forward.append(None)
        
        try:
            arrivals_backward.append(pm.compute_arrivals(env_backward))
        except ValueError:
            arrivals_backward.append(None)

    results = {'arrivals_forward': arrivals_forward,
            'arrivals_backward': arrivals_backward}

    fn = f'/datadrive/simulation/DAS/{k*3000}-{(k+1)*3000}_arrivals.pkl'

    with open(fn, 'wb') as f:
        pickle.dump(results, f)
