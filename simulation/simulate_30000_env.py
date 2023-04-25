import pyat
from pyat_tools import pyat_tools
import xarray as xr
import pandas as pd
import numpy as np
from numpy import matlib
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm
import scipy
from geopy.distance import geodesic
from oceans.sw_extras import sw_extras as sw

## Load SSP
ds = xr.open_mfdataset('/datadrive/HYCOM_data/DAS_30000/*.nc', decode_times=False).mean('time', keepdims=True)
sspx = sw.soundspeed(ds.salinity, ds.water_temp, ds.depth)

## Write Environment File and flp files
Fs = 200
To = 30
t, freq_half, freq_full = pyat_tools.get_freq_time_vectors(Fs, To)

ssp = pyat_tools.convert_SSP_arlpy(sspx, 0)
ssp = np.vstack((ssp, ssp[-1, :]))[1:]
ssp[-1, 0] = 600

fn = 'kraken_files/das_30000'
ranges = np.arange(0,6000,2)/1000

pyat_tools.write_env_file_pyat(ssp, 546.5, 546, ranges, np.array([546]), 20, 'das_30000_simulation', fn=fn, verbose=False)

# Write field flp file
s_depths = np.array([546])  # meters
r_depths = np.array([546])  # meters

pyat_tools.write_flp_file(s_depths, ranges, r_depths, fn)

data_lens = {'s_depths': len(s_depths), 'ranges': len(
    ranges), 'r_depths': len(r_depths)}

pf = pyat_tools.simulate_FDGF(
    fn, freq_half, [0, 100], 'multiprocessing/', data_lens, True)

pt = np.real(scipy.fft.ifft(pf[:,0,0,:], axis=0))

fn = '/datadrive/simulation/DAS/30000_env.pkl'
with open(fn, 'wb') as f:
    pickle.dump(pt, f)