import numpy as np
import os, pdb, random, math, cmath, time, datetime, re
from scipy.interpolate import RegularGridInterpolator, interp2d, griddata, RectBivariateSpline
from scipy.signal import convolve2d
from scipy.stats import multivariate_normal
from location import Location, Observation, LocDelta
from utils import dateLinspace, dateRange, getBox, getLatLon
from roms import getROMSData, reshapeROMS
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import deepdish as dd

class World(object):

  """docstring for World"""
  def __init__(self, sci_type, scalar_field, current_u_field, current_v_field, x_ticks, y_ticks, t_ticks, lon_ticks, lat_ticks, cell_x_size, cell_y_size, bounds):
    self.science_variable_type = sci_type
    self.scalar_field = scalar_field
    self.current_u_field = current_u_field
    self.current_v_field = current_v_field

    self.x_ticks = x_ticks # km
    self.y_ticks = y_ticks # km
    self.t_ticks = t_ticks # Time (s) since the world began
    self.lon_ticks  = lon_ticks # Decimal Degrees
    self.lat_ticks = lat_ticks # Decimal Degrees
    self.cell_y_size = cell_y_size
    self.cell_x_size = cell_x_size


    self.bounds = bounds
    self.n_bound = bounds[0]
    self.s_bound = bounds[1]
    self.e_bound = bounds[2]
    self.w_bound = bounds[3]



  def __str__(self):
    return "X-axis: " + str(self.x_ticks) + "\nY-axis: " + str(self.y_ticks) + "\nWorld:\n" + str(self.scalar_field)

  def __repr__(self):
    return "World Class Object"


  def xy2latlon(self, query_xy):
    x2lon_ratio = (self.lon_ticks[1] - self.lon_ticks[0]) / (self.x_ticks[1] - self.x_ticks[0])  
    y2lat_ratio = (self.lat_ticks[1] - self.lat_ticks[0]) / (self.y_ticks[1] - self.y_ticks[0])  

    xy_reference = Location(xlon=self.x_ticks[0], ylat=self.y_ticks[0])
    latlon_reference = Location(xlon=self.lon_ticks[0], ylat=self.lat_ticks[0])

    dxdy = query_xy - xy_reference

    dlatdlon = LocDelta(d_ylat=dxdy.d_ylat*y2lat_ratio, d_xlon=dxdy.d_xlon*x2lon_ratio)

    return latlon_reference + dlatdlon


  def latlon2xy(self, query_latlon):
    lon2x_ratio = (self.x_ticks[1] - self.x_ticks[0]) / (self.lon_ticks[1] - self.lon_ticks[0])  
    lat2y_ratio = (self.y_ticks[1] - self.y_ticks[0]) / (self.lat_ticks[1] - self.lat_ticks[0])  

    xy_reference = Location(xlon=self.x_ticks[0], ylat=self.y_ticks[0])
    latlon_reference = Location(xlon=self.lon_ticks[0], ylat=self.lat_ticks[0])

    dlatdlon = query_latlon - latlon_reference

    dxdy = LocDelta(d_ylat=dlatdlon.d_ylat*lat2y_ratio, d_xlon=dlatdlon.d_xlon*lon2x_ratio)

    return xy_reference + dxdy


  def withinBounds(self, query_loc, loc_type='xy'):
    if loc_type == "xy":
      if query_loc.x < np.min(self.x_ticks):
        return False
      if query_loc.x > np.max(self.x_ticks):
        return False
      if query_loc.y < np.min(self.y_ticks):
        return False
      if query_loc.y > np.max(self.y_ticks):
        return False
      return True
    elif loc_type == "latlon":
      if query_loc.lon < np.min(self.lon_ticks):
        return False
      if query_loc.lon > np.max(self.lon_ticks):
        return False
      if query_loc.lat < np.min(self.lat_ticks):
        return False
      if query_loc.lat > np.max(self.lat_ticks):
        return False
      return True


  def makeObservations(self, query_locs, query_times, query_type='sci', loc_type='xy'):
    query_times = [min(time, self.t_ticks[-1]) for time in query_times]

    if loc_type == "xy":
      if query_type == 'sci':
        sci_interp = RegularGridInterpolator((self.x_ticks, self.y_ticks, self.t_ticks), self.scalar_field, fill_value=float('NaN'), bounds_error=False)
        return [Observation(query_loc, float(sci_interp((query_loc.x, query_loc.y, query_time))), query_time) for query_loc, query_time in zip(query_locs, query_times) if self.withinBounds(query_loc, loc_type=loc_type)]

      elif query_type == 'current':
        u_interp = RegularGridInterpolator((self.x_ticks, self.y_ticks, self.t_ticks), self.current_u_field, fill_value=float('NaN'), bounds_error=False)
        v_interp = RegularGridInterpolator((self.x_ticks, self.y_ticks, self.t_ticks), self.current_v_field, fill_value=float('NaN'), bounds_error=False)

        u_obs = [Observation(query_loc, float(u_interp((query_loc.x, query_loc.y, query_time))), query_time) for query_loc, query_time in zip(query_locs, query_times) if self.withinBounds(query_loc, loc_type=loc_type)]
        v_obs = [Observation(query_loc, float(v_interp((query_loc.x, query_loc.y, query_time))), query_time) for query_loc, query_time in zip(query_locs, query_times) if self.withinBounds(query_loc, loc_type=loc_type)]

        return u_obs, v_obs
    elif loc_type == "latlon":
      if query_type == 'sci':
        sci_interp = RegularGridInterpolator((self.lon_ticks, self.lat_ticks, self.t_ticks), self.scalar_field, fill_value=float('NaN'), bounds_error=False)
        return [Observation(query_loc, float(sci_interp((query_loc.lon, query_loc.lat, query_time))), query_time) for query_loc, query_time in zip(query_locs, query_times) if self.withinBounds(query_loc, loc_type=loc_type)]

      elif query_type == 'current':
        u_interp = RegularGridInterpolator((self.lon_ticks, self.lat_ticks, self.t_ticks), self.current_u_field, fill_value=float('NaN'), bounds_error=False)
        v_interp = RegularGridInterpolator((self.lon_ticks, self.lat_ticks, self.t_ticks), self.current_v_field, fill_value=float('NaN'), bounds_error=False)

        u_obs = [Observation(query_loc, float(u_interp((query_loc.lon, query_loc.lat, query_time))), query_time) for query_loc, query_time in zip(query_locs, query_times) if self.withinBounds(query_loc, loc_type=loc_type)]
        v_obs = [Observation(query_loc, float(v_interp((query_loc.lon, query_loc.lat, query_time))), query_time) for query_loc, query_time in zip(query_locs, query_times) if self.withinBounds(query_loc, loc_type=loc_type)]

        return u_obs, v_obs

  def getSnapshot(self, ss_time, snapshot_type='scalar_field'):
    time_dist = [abs(ss_time - x) for x in self.t_ticks]#
    snapshot_time_idx = time_dist.index(min(time_dist))

    if snapshot_type == 'scalar_field':
      return self.scalar_field[:,:,snapshot_time_idx]

    elif snapshot_type == 'current_u_field':
      return self.current_u_field[:,:,snapshot_time_idx]

    elif snapshot_type == 'current_v_field':
      return self.current_v_field[:,:,snapshot_time_idx]

  def getUVcurrent(self, loc, t, loc_type='xy'):
    u_snapshot = self.getSnapshot(t, 'current_u_field')
    v_snapshot = self.getSnapshot(t, 'current_v_field')

    if loc_type == "xy":
      u_interp = RegularGridInterpolator((self.x_ticks, self.y_ticks), u_snapshot, fill_value=0.0, bounds_error=False)
      v_interp = RegularGridInterpolator((self.x_ticks, self.y_ticks), v_snapshot, fill_value=0.0, bounds_error=False)

      current_u_current = u_interp((loc.x, loc.y))
      current_v_current = v_interp((loc.x, loc.y))

    elif loc_type == "latlon":
      u_interp = RegularGridInterpolator((self.lon_ticks, self.lat_ticks), u_snapshot, fill_value=0.0, bounds_error=False)
      v_interp = RegularGridInterpolator((self.lon_ticks, self.lat_ticks), v_snapshot, fill_value=0.0, bounds_error=False)

      current_u_current = u_interp((loc.lon, loc.lat))
      current_v_current = v_interp((loc.lon, loc.lat))

    return LocDelta(d_xlon = float(current_u_current), d_ylat = float(current_v_current))


  def draw(self, ax, block=True, show=False, cbar_max=None, cbar_min=None, quiver_stride=7, snapshot_time=None, draw_currents=True, cmap='Greys'):

    if snapshot_time is None:
      ss_scalar_field = self.getSnapshot(self.t_ticks[0], 'scalar_field')
      ss_current_u_field = self.getSnapshot(self.t_ticks[0], 'current_u_field')
      ss_current_v_field = self.getSnapshot(self.t_ticks[0], 'current_v_field')
    else:
      ss_scalar_field = self.getSnapshot(snapshot_time, 'scalar_field')
      ss_current_u_field = self.getSnapshot(snapshot_time, 'current_u_field')
      ss_current_v_field = self.getSnapshot(snapshot_time, 'current_v_field')

    if cbar_min is None:
      cbar_min = np.min(ss_scalar_field)
    if cbar_max is None:
      cbar_max = np.max(ss_scalar_field)

    num_format  = '%.0f'
    formatter = tick.FormatStrFormatter(num_format)

    CS = plt.pcolor(self.x_ticks, self.y_ticks, ss_scalar_field.transpose(), cmap=cmap, vmin=cbar_min, vmax=cbar_max)

    if draw_currents:
      quiver = plt.quiver(self.x_ticks[::quiver_stride], self.y_ticks[::quiver_stride], ss_current_u_field.transpose()[::quiver_stride, ::quiver_stride], ss_current_v_field.transpose()[::quiver_stride, ::quiver_stride])
      quiver_key = plt.quiverkey(quiver, 0.95, 1.05, 0.2, "0.2 m/s", labelpos='E', coordinates='axes')

    ax.get_xaxis().set_major_formatter(formatter)
    ax.get_yaxis().set_major_formatter(formatter)
    plt.ylim([np.min(self.x_ticks), np.max(self.x_ticks)])
    plt.xlim([np.min(self.y_ticks), np.max(self.y_ticks)]) 
    cbar = plt.colorbar(CS, format='%.1f')
    plt.title("Ground Truth World")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)") 
    cbar.set_label(self.science_variable_type)
    ax.axis('scaled')

    if show:
      plt.show(block)

  def getRandomLocationXY(self):
    return Location(xlon=random.choice(self.x_ticks), ylat=random.choice(self.y_ticks))

  def getRandomLocationLatLon(self):
    return Location(xlon=random.choice(self.lon_ticks), ylat=random.choice(self.lat_ticks))



  @classmethod
  def roms(cls, datafile_path, xlen, ylen, center, feature='temperature', resolution=(0.1, 0.1)):

    # World bounds
    bounds = getBox(xlen=xlen, ylen=ylen, center=center)

    n_bound   = bounds[0]
    s_bound   = bounds[1]
    e_bound   = bounds[2]
    w_bound   = bounds[3]

    x_ticks   = np.arange(0.0, xlen+resolution[0], resolution[0])
    y_ticks   = np.arange(0.0, ylen+resolution[1], resolution[1])

    x_ticks = x_ticks - np.max(x_ticks)/2.
    y_ticks = y_ticks - np.max(y_ticks)/2.

    lon_ticks = np.linspace(w_bound, e_bound, len(x_ticks))
    lat_ticks = np.linspace(s_bound, n_bound, len(y_ticks))


    scalar_field, scalar_lat, scalar_lon, roms_t = getROMSData(datafile_path, feature)
    current_u, u_lat, u_lon, _ = getROMSData(datafile_path, 'u')
    current_v, v_lat, v_lon, _ = getROMSData(datafile_path, 'v')

    output_shape = (len(x_ticks), len(y_ticks), len(roms_t))

    scalar_field = reshapeROMS(scalar_field, scalar_lat, scalar_lon, bounds, output_shape)
    current_u = reshapeROMS(current_u, u_lat, u_lon, bounds, output_shape)
    current_v = reshapeROMS(current_v, v_lat, v_lon, bounds, output_shape)
    return cls(feature, scalar_field.data, current_u.data, current_v.data, x_ticks, y_ticks, roms_t, lon_ticks, lat_ticks, resolution[0], resolution[1], bounds)


  @classmethod
  def idealizedFront(cls, start_date, end_date, time_resolution, resolution, xlen, ylen):
    # script to create an undulated temperature front that propagates and changes orientation in time.   
    
    ##################################################
    ### Parameters
    ##################################################

    theta_0 = random.random()*360.0 # initial orientation of front (in degrees)
    dtheta_dt = -45 # rate at which the front is rotating (in degrees per day) 
    undulation_wavelength = 7 # wavelength of undulations on the front (in km)
    undulation_amplitude = 2 # undulation_amplitudelitude of the undulations;.
    wave_speed = 2.0 #2.0 # propagation speed of the undulations, in m/s. 
    temp_cold = 10 # is the cold side temperature
    temp_warm = 15 # is the warm side temperature;
    noise = 2.
    current_magnitude = 0.2 # Current Magnitude in m/s
    omega = (wave_speed / 1000)

    # World bounds
    bounds = getBox(
      xlen  = xlen,
      ylen  = ylen,
      center  = Location(0.0,0.0),
    )


    width   = 0.5*xlen
    height    = 0.5*ylen
    x_ticks   = np.arange(-width, width+resolution[0], resolution[0])
    y_ticks   = np.arange(-height, height+resolution[1], resolution[1])

    x_ticks = x_ticks - np.max(x_ticks)/2.
    y_ticks = y_ticks - np.max(y_ticks)/2.

    lon_ticks = np.linspace(bounds[3], bounds[2], len(x_ticks))
    lat_ticks = np.linspace(bounds[1], bounds[0], len(y_ticks)) 

    if isinstance(time_resolution, float) or isinstance(time_resolution, int):
      t_ticks = dateLinspace(start_date, end_date, time_resolution)
    elif isinstance(time_resolution, datetime.timedelta):
      t_ticks = dateRange(start_date, end_date, time_resolution)

    t_ticks = np.array([(x-start_date).total_seconds() for x in t_ticks])


    xx, yy = np.meshgrid(x_ticks, y_ticks)
 
    theta = theta_0
    noise_kernel = np.ones((5,5)) * (1 / 25.)


    res_scalar_field = np.empty((len(x_ticks), len(y_ticks), len(t_ticks)))
    res_current_u_field = np.empty((len(x_ticks), len(y_ticks), len(t_ticks)))
    res_current_v_field = np.empty((len(x_ticks), len(y_ticks), len(t_ticks)))

    for t_idx, t in enumerate(t_ticks):
      theta = math.radians(theta_0 + dtheta_dt * t/(24*3600))
      theta = theta % (math.pi * 2)
      cos_theta = math.cos(theta)
      sin_theta = math.sin(theta)
      



      if theta <= 1*math.pi / 4 or theta > 7*math.pi/4: #Mode 0
        zz_final = sin_theta * (cos_theta*xx + sin_theta*yy) + cos_theta * undulation_amplitude * np.sin((cos_theta*xx + sin_theta * yy + omega * t) * (2*math.pi / undulation_wavelength)) - yy
        zz_final = zz_final > 0
      elif theta > 1*math.pi / 4 and theta <= 3*math.pi/4: #Mode 1
        zz_final = cos_theta * (cos_theta*xx + sin_theta*yy) - sin_theta * undulation_amplitude * np.sin((cos_theta*xx + sin_theta * yy + omega * t) * (2*math.pi / undulation_wavelength)) - xx 
        zz_final = zz_final < 0
      elif theta > 3*math.pi / 4 and theta <= 5*math.pi/4: #Mode 2
        zz_final = sin_theta * (cos_theta*xx + sin_theta*yy) + cos_theta * undulation_amplitude * np.sin((cos_theta*xx + sin_theta * yy + omega * t) * (2*math.pi / undulation_wavelength)) - yy
        zz_final = zz_final < 0
      elif theta > 5*math.pi / 4 and theta <= 7*math.pi/4: #Mode 3
        zz_final = cos_theta * (cos_theta*xx + sin_theta*yy) - sin_theta * undulation_amplitude * np.sin((cos_theta*xx + sin_theta * yy + omega * t) * (2*math.pi / undulation_wavelength)) - xx 
        zz_final = zz_final > 0

      zz_final = zz_final * (temp_warm - temp_cold) + temp_cold

      t_noise = noise * (np.random.random(zz_final.shape) - 0.5)

      scalar_field = convolve2d(zz_final+t_noise, noise_kernel, boundary='symm', mode='same')

      #current_u_field = current_magnitude * cos_theta * np.ones(scalar_field.shape)
      #current_v_field = current_magnitude * sin_theta * np.ones(scalar_field.shape)

      current_u_field = yy
      current_v_field = -1*xx

      current_u_field = current_magnitude * current_u_field / np.max(np.sqrt(current_u_field**2 + current_v_field**2)) 
      current_v_field = current_magnitude * current_v_field / np.max(np.sqrt(current_u_field**2 + current_v_field**2)) 

      res_scalar_field[:,:,t_idx] = scalar_field.transpose()
      res_current_u_field[:,:,t_idx] = current_u_field.transpose()
      res_current_v_field[:,:,t_idx] = current_v_field.transpose()

    return cls('temperature', res_scalar_field, res_current_u_field, res_current_v_field, x_ticks, y_ticks, t_ticks, lon_ticks, lat_ticks, resolution[0], resolution[1], bounds)

  def random(cls, start_date, end_date, time_resolution, world_resolution, bounds=None, num_generators=50):

    if bounds is not None:
      x_ticks = np.linspace(bounds[3], bounds[2], world_resolution[0])
      y_ticks = np.linspace(bounds[1], bounds[0], world_resolution[1])
      
    else:
      x_ticks = np.arange(0,world_resolution[0])
      y_ticks = np.arange(0,world_resolution[1])
      bounds = [world_resolution[1], 0, world_resolution[0], 0]

    if isinstance(time_resolution, float) or isinstance(time_resolution, int):
      t_ticks = dateLinspace(start_date, end_date, time_resolution)
    elif isinstance(time_resolution, datetime.timedelta):
      t_ticks = dateRange(start_date, end_date, time_resolution)

    t_ticks = np.array([(x-start_date).total_seconds() for x in t_ticks])

    # pdb.set_trace()
    yy, xx, tt = np.meshgrid(y_ticks, x_ticks, t_ticks)
    #xx, yy, tt = np.mgrid[x_ticks[0]:x_ticks[-1]:world_resolution[0]*1j, y_ticks[0]:y_ticks[-1]:world_resolution[1]*1j, t_ticks[0]:t_ticks[-1]:time_resolution*1j]

    pos = np.empty(xx.shape + (3,))
    pos[:, :, :, 0] = xx
    pos[:, :, :, 1] = yy
    pos[:, :, :, 2] = tt

    sigma_x = .075*(bounds[0] - bounds[1])
    sigma_y = .075*(bounds[2] - bounds[3])
    sigma_t = .075*(t_ticks[-1] - t_ticks[0])

    cov = np.eye(3) * np.array([sigma_x, sigma_y, sigma_t])

    res = np.zeros((len(x_ticks), len(y_ticks), len(t_ticks)))


    generators = [[random.random()*(bounds[2] - bounds[3]) + bounds[3], random.random()*(bounds[0] - bounds[1]) + bounds[1], random.random()*(t_ticks[-1] - t_ticks[0]) + t_ticks[0]] for ii in range(num_generators)]

    for generator_idx, generator in enumerate(generators):
      x_o = generator[0]
      y_o = generator[1]
      t_o = generator[2]
      generator_res = (1/(2*np.pi*sigma_x*sigma_y*sigma_t) * np.exp(-((xx-x_o)**2/(2*sigma_x**2) + (yy-y_o)**2/(2*sigma_y**2) + (tt-t_o)**2/(2*sigma_t**2))))
      res = res + generator_res

    res = res - np.min(res)
    res = res / np.max(res)

    return cls('temperature', res, x_ticks, y_ticks, t_ticks, resolution[0], resolution[1], bounds)



def loadWorld(roms_file, world_center, world_width, world_height, world_resolution, science_variable='temperature', save_dir=None):
  
  if save_dir is not None:
    filename_pattern = re.compile('.*/(.*).nc')
    world_file_name = "%s_lat_%.5f_lon_%.5f_width_%.3f_height_%.3f.h5" % (
      re.findall(filename_pattern, roms_file)[0], 
      world_center.lat,
      world_center.lon,
      world_width,
      world_height
      )

    world_file_full_path = os.path.expandvars(save_dir + world_file_name)
    if os.path.isfile(world_file_full_path):
      print "%s Exists" % world_file_full_path
      wd = dd.io.load(world_file_full_path)
      print "Loaded World from File"

    else:
      print "%s Does Not Exist" % world_file_full_path
      wd = World.roms(
        datafile_path = roms_file,
        xlen          = world_width,
        ylen          = world_height,
        center        = world_center,
        feature       = science_variable,
        resolution    = world_resolution
        )

      dd.io.save(world_file_full_path, wd, compression='zlib')
      print "World saved to %s" % world_file_full_path


  else:
    wd = World.roms(
      datafile_path = roms_file,
      xlen          = world_width,
      ylen          = world_height,
      center        = world_center,
      feature       = science_variable,
      resolution    = world_resolution
      )
  return wd


def main():

  # n_bound = 29.0
  # s_bound = 28.0
  # e_bound = -94.0
  # w_bound = -95.0

  # bounds = [n_bound, s_bound, e_bound, w_bound]

  # d1 = datetime.datetime(2018, 1, 1)
  # d2 = datetime.datetime(2018, 1, 2)
  # bounds = getBox(xlen = 20, ylen = 20, center = Location(0.0,0.0))

  # wd = World.idealizedFront(
  #   start_date      = d1,
  #   end_date      = d2,
  #   time_resolution   = 24,
  #   resolution      = (0.100, 0.100),
  #   xlen        = 15.,
  #   ylen        = 20.,
  # )
  # wd = World.random(d1, d2, 24, (100, 110), bounds=bounds, num_generators = 50)
  datafile_path = os.path.dirname(os.path.realpath(__file__)) + "/../data/roms_data/"
  datafile_name = "txla_roms/txla_hindcast_jun_1_2015.nc"

  wd = World.roms(datafile_path + datafile_name, 20, 20, Location(xlon=-94.25, ylat=28.25), feature='salt', resolution=(0.1, 0.1))

  print "Generating Figures"

  for t_idx, t in enumerate(wd.t_ticks):
    fig = plt.figure()
    plt.clf()
    plt.title(str(datetime.datetime.fromtimestamp(wd.t_ticks[t_idx])))
    img = plt.pcolor(wd.lon_ticks, wd.lat_ticks, wd.scalar_field[:, :, t_idx].transpose(), vmin=np.min(wd.scalar_field), vmax=np.max(wd.scalar_field))
    cbar = plt.colorbar(img)
    quiver_stride = 10
    plt.xticks(rotation=45)
    plt.quiver(wd.lon_ticks[::quiver_stride], wd.lat_ticks[::quiver_stride], wd.current_u_field[:, :, t_idx].transpose()[::quiver_stride, ::quiver_stride], wd.current_v_field[:, :, t_idx].transpose()[::quiver_stride, ::quiver_stride])
    fig.canvas.draw()
    # plt.show(block=False)
    filename = "../results/plt/world-%03d" % t_idx
    plt.savefig(filename, bbox_inches='tight')
  plt.close('all')



if __name__ == '__main__':
  main()
