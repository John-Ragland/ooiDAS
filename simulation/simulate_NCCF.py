import pandas as pd
import hvplot.pandas

import holoviews as hv
import xarray as xr
from tqdm import tqdm
from matplotlib import pyplot as plt

from oceans.sw_extras import sw_extras as sw
import pickle

from pyat_tools import pyat_tools
import numpy as np
import scipy
from scipy import signal

# Load SSP and cable geometry
fn = '/datadrive/HYCOM_data/DAS_ooi_South_Tx/*.nc'
ds = xr.open_mfdataset(fn, decode_times=False)

fn = '/datadrive/DAS/south_DAS_latlondepth.txt'
geo = pd.read_csv(fn)
geo['distance'] = geo['index']*2.0419046878814697/1000

geo_start_idx = (geo['index']-27000).abs().argmin()
geo_end_idx = geo_start_idx + 3000

middle_point = geo.loc[geo_start_idx:geo_end_idx].mean()
lat = middle_point['lat']
lon = middle_point['lon']
depth = middle_point['depth']*-1

print('lat\t\t\t lon\t\t\t depth')
print(f'({lat}, \t{lon})\t{depth} m')

ds_slice = ds.mean('time', keepdims=True).sel({'lat':lat, 'lon':lon+360}, method='nearest')

sspx = sw.soundspeed(ds_slice.salinity, ds_slice.water_temp, ds_slice.depth)
ssp = pyat_tools.convert_SSP_arlpy(sspx, time_idx=0)[1:,:]

ssp = np.vstack((ssp, ssp[-1,:]))
ssp[-1,0] = 600

fn = 'kraken_files/NCCF'
ranges = np.arange(0,30,0.002) #sampling every 2 m out to 30 km

## Write Environment File and flp files
Fs = 200
To = 30
t, freq_half, freq_full = pyat_tools.get_freq_time_vectors(Fs, To)
pyat_tools.write_env_file_pyat(ssp, depth, depth+0.1, ranges, np.array([depth]), 20, 'p(range)', fn, verbose=True)

# Write field flp file
s_depths = np.array([depth+0.1])  # meters
r_depths = np.array([20])  # meters

pyat_tools.write_flp_file(s_depths, ranges, r_depths, fn)

data_lens = {'s_depths': len(s_depths), 'ranges': len(
    ranges), 'r_depths': len(r_depths)}

pf = pyat_tools.simulate_FDGF(
    fn, freq_half, [0,100], 'multiprocessing/', data_lens, True)

pt = np.real(scipy.fft.ifft(pf[:,0,0,:], axis=0))

fn = '/datadrive/simulation/DAS/pt_range.pkl'
with open(fn, 'wb') as f:
    pickle.dump(pt, f)