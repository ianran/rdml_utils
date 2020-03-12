import datetime, math, urllib, os, pdb, itertools
from scipy.interpolate import griddata
from tqdm import tqdm
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from .utils import dateRange



def getROMSData(datafile_path, feature):
  #Load a single Roms Feature as a scalar field, return the field and its bounds
  #Note that the field is a masked array, so all locations within the bounds are not guaranteed to be valid

  if 'txla' in datafile_path:
    #ROMS Data is from Texas - Lousisiana Dataset
    return loadTXLAROMSData(datafile_path, feature)

  elif 'ocean_his' in datafile_path:
    #ROMS Data is from Oregon Dataset
    return loadOregonROMSData(datafile_path, feature)

  elif 'ca300m' in datafile_path:
    #ROMS Data is from Monterey Dataset
    return loadMontereyROMSData(datafile_path, feature)



def loadOregonROMSData(datafile_path, feature="temperature"):
  # Returns Ocean Surface Temperature
  roms_dataset = nc.Dataset(datafile_path)

  base_time = datetime.datetime.strptime(str(roms_dataset['ocean_time'].units), "seconds since %Y-%m-%d %H:%M:%S")
  times = np.array([(base_time - datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=int(dt))).total_seconds() for dt in roms_dataset['ocean_time'][:]])

  if feature == 'temp' or feature == 'temperature':
    scalar_field = roms_dataset['temp'][:,-1,:,:] # I'm pretty the depth variable goes bottom -> surface which is different than other roms datasets
    lat = roms_dataset['lat_rho'][:]
    lon = roms_dataset['lon_rho'][:]

  if feature == 'h' or feature == 'depth':
    lat = roms_dataset['lat_rho'][:]
    lon = roms_dataset['lon_rho'][:]
    scalar_field = roms_dataset['h'][:]
    scalar_field = np.repeat(scalar_field[np.newaxis,:,:], roms_dataset['temp'].shape[0], axis=0)

  elif feature == 'current_u' or feature == "u":
    scalar_field = roms_dataset['u'][:,-1,:,:]
    lat = roms_dataset['lat_u'][:]
    lon = roms_dataset['lon_u'][:]

  elif feature == 'current_v' or feature == "v":
    scalar_field = roms_dataset['v'][:,-1,:,:]
    lat = roms_dataset['lat_v'][:]
    lon = roms_dataset['lon_v'][:]

  elif feature == 'salinity' or feature == "salt":
    scalar_field = roms_dataset['salt'][:,-1,:,:]
    lat = roms_dataset['lat_rho'][:]
    lon = roms_dataset['lon_rho'][:]


  return scalar_field, lat, lon, times



def loadMontereyROMSData(datafile_path, feature="temperature"):
  roms_dataset = nc.Dataset(datafile_path)

  #Convert dumb time units into UTC Timestamp
  base_time = datetime.datetime.strptime(str(roms_dataset['time'].units), "hour since %Y-%m-%d %H:%M:%S")
  times = np.array([(base_time - datetime.datetime(1970, 1, 1) + datetime.timedelta(hours=int(dt))).total_seconds() for dt in roms_dataset['time'][:]])

  if feature == 'h' or feature == 'depth':
    raise Exception("Depth Not in Monterey Dataset, You're outta luck")

  if feature == "temp" or feature == "temperature":
    scalar_field = roms_dataset['temp'][:,0,:,:]

  elif feature == "salinity" or feature == "salt":
    scalar_field = roms_dataset['salt'][:,0,:,:]

  elif feature == "current_u" or feature == "u":
    scalar_field = roms_dataset['u'][:,0,:,:]

  elif feature == "current_v" or feature == "v":
    scalar_field = roms_dataset['v'][:,0,:,:]

  lon, lat = np.meshgrid(roms_dataset['lon'][:], roms_dataset['lat'][:])

  return scalar_field, lat, lon, times



def loadTXLAROMSData(datafile_path, feature='temperature'):
  roms_dataset = nc.Dataset(datafile_path)
  base_time = datetime.datetime.strptime(str(roms_dataset['ocean_time'].units), "seconds since %Y-%m-%d %H:%M:%S")
  times = np.array([(base_time - datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=int(dt))).total_seconds() for dt in roms_dataset['ocean_time'][:]])

  if feature == 'temp' or feature == 'temperature':
    lat = roms_dataset['lat_rho'][:]
    lon = roms_dataset['lon_rho'][:]
    # times = roms_dataset['ocean_time'][:]
    scalar_field = roms_dataset['temp'][:,0,:,:]

  if feature == 'h' or feature == 'depth':
    lat = roms_dataset['lat_rho'][:]
    lon = roms_dataset['lon_rho'][:]
    scalar_field = roms_dataset['h'][:]
    scalar_field = np.repeat(scalar_field[np.newaxis,:,:], roms_dataset['temp'].shape[0], axis=0)

  elif feature == 'salt' or feature == 'salinity':
    lat = roms_dataset['lat_rho'][:]
    lon = roms_dataset['lon_rho'][:]
    # times = roms_dataset['ocean_time'][:]
    scalar_field = roms_dataset['salt'][:,0,:,:]

  elif feature == 'current_u' or feature == "u":
    lat = roms_dataset['lat_u'][:]
    lon = roms_dataset['lon_u'][:]
    # times = roms_dataset['ocean_time'][:]
    scalar_field = roms_dataset['u'][:,0,:,:]

  elif feature == 'current_v' or feature == "v":
    lat = roms_dataset['lat_v'][:]
    lon = roms_dataset['lon_v'][:]
    # times = roms_dataset['ocean_time'][:]
    scalar_field = roms_dataset['v'][:,0,:,:]


  return scalar_field, lat, lon, times


def reshapeROMS(roms_field, roms_lat, roms_lon, bounds, output_shape):
  n_bound = bounds[0]
  s_bound = bounds[1]
  e_bound = bounds[2]
  w_bound = bounds[3]

  filtered_lon = roms_lon + -360*(roms_lon>180)
  filtered_lat = roms_lat

  if n_bound > np.nanmax(filtered_lat):
    print( "[Warning] North bound %.3f out of range of ROMS data (%.2f, %.2f)" % (n_bound, np.nanmin(filtered_lat), np.nanmax(filtered_lat)) )
  if s_bound < np.nanmin(filtered_lat):
    print( "[Warning] South bound %.3f out of range of ROMS data (%.2f, %.2f)" % (s_bound, np.nanmin(filtered_lat), np.nanmax(filtered_lat)) )
  if e_bound > np.nanmax(filtered_lon):
    print( "[Warning] East bound %.3f out of range of ROMS data (%.2f, %.2f)" % (e_bound, np.nanmin(filtered_lon), np.nanmax(filtered_lon)) )
  if w_bound < np.nanmin(filtered_lon):
    print( "[Warning] West bound %.3f out of range of ROMS data (%.2f, %.2f)" % (w_bound, np.nanmin(filtered_lon), np.nanmax(filtered_lon)))


  lonlon, latlat = np.mgrid[w_bound:e_bound:output_shape[0]*1j, s_bound:n_bound:output_shape[1]*1j]

  if filtered_lat.ndim == 1 and filtered_lon.ndim == 1:
    filtered_lat, filtered_lon = np.meshgrid(filtered_lat, filtered_lon)

  lat_coords = filtered_lat.flatten()
  lon_coords = filtered_lon.flatten()

  pts = np.vstack((lon_coords, lat_coords)).transpose()

  reshaped_field = np.empty(output_shape)

  for t_idx in tqdm(range(output_shape[2])):
    if isinstance(roms_field, np.ma.masked_array):
      data = roms_field[t_idx].data.flatten()
    else:
      data = roms_field[t_idx].flatten()

    zz = griddata(pts, data, (lonlon,latlat), fill_value=9999.)
    reshaped_field[:,:,t_idx] = zz

  data_range = np.max(roms_field) - np.min(roms_field)

  masked_field = np.ma.masked_less(np.ma.masked_greater(reshaped_field, np.max(roms_field) + .1*data_range), np.min(roms_field) - .1*data_range)

  return masked_field


def getMoneteryROMS(start_date, end_date, datafile_path):

  print( "Loading ROMS Data" )
  files = []
  dates = []
  for dt in dateRange(start_date, end_date, datetime.timedelta(hours=1)):
    filename = "ca300m_das_%04d%02d%02d%02d.nc" % (dt.year, dt.month, dt.day, dt.hour)
    url = "http://west.rssoffice.com:8080/thredds/fileServer/roms/CA300m-nowcast/" + filename
    file_path = datafile_path + filename
    if not os.path.isfile(file_path):
      try:
        testfile = urllib.URLopener()
        testfile.retrieve(url, file_path)
        print( dt, "\tLoading Data" )
        files.append(file_path)
        dates.append(dt)
        # print url
        print( file_path )
      except IOError:
        pass
      print( dt, "\tNo Data" )
      # pass
    else:
      # pass
      print( dt, "\tFile Already Exists" )
      files.append(file_path)
      dates.append(dt)
  return dates, files



if __name__ == '__main__':
  d1 = datetime.datetime(2018, 1, 1)
  d2 = datetime.datetime(2018, 1, 3)

  getMoneteryROMS(d1, d2, os.path.expandvars("$HOME/Desktop/"))
