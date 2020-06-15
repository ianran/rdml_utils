import os, pdb, random, math, cmath, time, datetime, re, sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import deepdish as dd
import scipy.stats as stats

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import RegularGridInterpolator, interp2d, griddata, RectBivariateSpline
from scipy.signal import convolve2d
from scipy.stats import multivariate_normal


if sys.version_info[0] < 3:
    # python 2
    from location import Location, Observation, LocDelta
    from utils import dateLinspace, dateRange, getBox, getLatLon, gradient2d
    from roms import getROMSData, reshapeROMS
else:
    # python 3
    from rdml_utils.location import Location, Observation, LocDelta
    from rdml_utils.utils import dateLinspace, dateRange, getBox, getLatLon, gradient2d
    from rdml_utils.roms import getROMSData, reshapeROMS




class World(object):

  """docstring for World"""
  def __init__(self, sci_type, scalar_field, current_u_field, current_v_field, x_ticks, y_ticks, t_ticks, lon_ticks, lat_ticks, cell_x_size, cell_y_size, bounds):
    self.science_fields = {}
    if isinstance(sci_type, list) and isinstance(scalar_field, list):
      # Multiple Science Fields
      for science_field_type, science_field in zip(sci_type, scalar_field):
        assert isinstance(science_field_type, str)
        self.science_fields[science_field_type] = science_field

    elif isinstance(sci_type, str):
      # Single science
      self.science_fields[sci_type] = scalar_field

    else:
      raise TypeError("Possible mismatch between science types and scalar fields")

    # Handle Masked Arrays & Construct Obstacle Field
    for sci_key in self.science_fields.keys():
      science_field = self.science_fields[sci_key]


      if isinstance(science_field, np.ma.core.MaskedArray):
        self.science_fields[sci_key] = science_field.data  # Shape (X_ticks, y_ticks, t_ticks)
        self.obstacle_field = science_field.mask  # Shape (X_ticks, y_ticks, t_ticks) 0 = Not obstacle pixel, 1 = obstacle pixel
      elif isinstance(science_field, np.ndarray):
        self.obstacle_field = np.zeros(science_field.shape)
      else:
        pdb.set_trace()


    if isinstance(self.obstacle_field, np.bool_) and self.obstacle_field == False:
      self.obstacle_field = np.zeros(science_field.shape)

    if isinstance(current_u_field, np.ma.core.MaskedArray):
      self.current_u_field = current_u_field.data  # Shape (X_ticks, y_ticks, t_ticks)
    elif isinstance(current_u_field, np.ndarray):
      self.current_u_field = current_u_field  # Shape (X_ticks, y_ticks, t_ticks)

    if isinstance(current_v_field, np.ma.core.MaskedArray):
      self.current_v_field = current_v_field.data  # Shape (X_ticks, y_ticks, t_ticks)
    elif isinstance(current_v_field, np.ndarray):
      self.current_v_field = current_v_field  # Shape (X_ticks, y_ticks, t_ticks)

    self.x_ticks = x_ticks                          # km
    self.y_ticks = y_ticks                          # km
    self.t_ticks = t_ticks                          # UTC
    self.lon_ticks  = lon_ticks                     # Decimal Degrees
    self.lat_ticks = lat_ticks                      # Decimal Degrees
    self.cell_y_size = cell_y_size
    self.cell_x_size = cell_x_size

    self.bounds = bounds
    self.n_bound = bounds[0]
    self.s_bound = bounds[1]
    self.e_bound = bounds[2]
    self.w_bound = bounds[3]



  # def __str__(self):
  #   return "X-axis: " + str(self.x_ticks) + "\nY-axis: " + str(self.y_ticks) + "\nWorld:\n" + str(self.scalar_fields)

  def __repr__(self):
    return "World Class Object"

  def isObstacle(self, query_loc, loc_type='xy'):
    if not self.withinBounds(query_loc, loc_type):
      return True

    elif loc_type == 'xy':
      x_dists = [abs(query_loc.x - pt_x) for pt_x in self.x_ticks]
      y_dists = [abs(query_loc.y - pt_y) for pt_y in self.y_ticks]

      pt_x_idx = np.argmin(x_dists)
      pt_y_idx = np.argmin(y_dists)
      return self.obstacle_field[pt_x_idx, pt_y_idx, 0]

    elif loc_type == 'latlon':
      lon_dists = [abs(query_loc.lon - pt_lon) for pt_lon in self.lon_ticks]
      lat_dists = [abs(query_loc.lat - pt_lat) for pt_lat in self.lat_ticks]

      pt_lon_idx = np.argmin(lon_dists)
      pt_lat_idx = np.argmin(lat_dists)
      return self.obstacle_field[pt_lon_idx, pt_lat_idx, 0]


  def loc2cell(self, loc, loc_type='xy'):
    if loc_type == 'xy':
      x_distances = np.abs(self.x_ticks - loc.x)
      x_coord = np.argmin(x_distances)

      y_distances = np.abs(self.y_ticks - loc.y)
      y_coord = np.argmin(y_distances)

    elif loc_type == 'latlon':
      lat_distances = np.abs(self.lat_ticks - loc.lat)
      y_coord = np.argmin(lat_distances)

      lon_distances = np.abs(self.lon_ticks - loc.lon)
      x_coord = np.argmin(lon_distances)

    return Location(xlon=x_coord, ylat=y_coord)


  def cell2loc(self, cell, loc_type='xy'):
    if loc_type == 'xy':
      if cell.x >= len(self.x_ticks):
        x_coord = self.x_ticks[-1]
      elif cell.x < 0:
        x_coord = self.x_ticks[0]
      else:
        x_coord = self.x_ticks[int(cell.x)]

      if cell.y >= len(self.y_ticks):
        y_coord = self.y_ticks[-1]
      elif cell.y < 0:
        y_coord = self.y_ticks[0]
      else:
        y_coord = self.y_ticks[int(cell.y)]

    elif loc_type == 'latlon':
      if cell.x >= len(self.lon_ticks):
        x_coord = self.lon_ticks[-1]
      elif cell.x < 0:
        x_coord = self.lon_ticks[0]
      else:
        x_coord = self.lon_ticks[int(cell.x)]

      if cell.y >= len(self.lat_ticks):
        y_coord = self.lat_ticks[-1]
      elif cell.y < 0:
        y_coord = self.lat_ticks[0]
      else:
        y_coord = self.lat_ticks[int(cell.y)]

    return Location(xlon=x_coord, ylat=y_coord)


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


  def makeObservations(self, query_locs, query_times, query_type='salinity', loc_type='xy'):
    query_times = [min(time, self.t_ticks[-1]) for time in query_times]

    # check through science data types and see if query type is in there.
    if query_type != 'current' and query_type not in self.science_fields:
      raise ValueError('World cannot make Observations with science type: ' +str(query_type))
    else:



      if len(self.t_ticks) > 1:
        if loc_type == "xy":
          if query_type == 'current':
            u_interp = RegularGridInterpolator((self.x_ticks, self.y_ticks, self.t_ticks), self.current_u_field, fill_value=float('NaN'), bounds_error=False)
            v_interp = RegularGridInterpolator((self.x_ticks, self.y_ticks, self.t_ticks), self.current_v_field, fill_value=float('NaN'), bounds_error=False)

            u_obs = [Observation(query_loc, float(u_interp((query_loc.x, query_loc.y, query_time))), query_time) for query_loc, query_time in zip(query_locs, query_times) if self.withinBounds(query_loc, loc_type=loc_type)]
            v_obs = [Observation(query_loc, float(v_interp((query_loc.x, query_loc.y, query_time))), query_time) for query_loc, query_time in zip(query_locs, query_times) if self.withinBounds(query_loc, loc_type=loc_type)]

            return u_obs, v_obs
          else:
            science_field = self.science_fields[query_type]
            sci_interp = RegularGridInterpolator((self.x_ticks, self.y_ticks, self.t_ticks), science_field, fill_value=float('NaN'), bounds_error=False)
            return [Observation(query_loc, float(sci_interp((query_loc.x, query_loc.y, query_time))), query_time) for query_loc, query_time in zip(query_locs, query_times) if self.withinBounds(query_loc, loc_type=loc_type)]


        elif loc_type == "latlon":
          if query_type == 'current':
            u_interp = RegularGridInterpolator((self.lon_ticks, self.lat_ticks, self.t_ticks), self.current_u_field, fill_value=float('NaN'), bounds_error=False)
            v_interp = RegularGridInterpolator((self.lon_ticks, self.lat_ticks, self.t_ticks), self.current_v_field, fill_value=float('NaN'), bounds_error=False)

            u_obs = [Observation(query_loc, float(u_interp((query_loc.lon, query_loc.lat, query_time))), query_time) for query_loc, query_time in zip(query_locs, query_times) if self.withinBounds(query_loc, loc_type=loc_type)]
            v_obs = [Observation(query_loc, float(v_interp((query_loc.lon, query_loc.lat, query_time))), query_time) for query_loc, query_time in zip(query_locs, query_times) if self.withinBounds(query_loc, loc_type=loc_type)]

            return u_obs, v_obs
          else:
            science_field = self.science_fields[query_type]
            sci_interp = RegularGridInterpolator((self.lon_ticks, self.lat_ticks, self.t_ticks), science_field, fill_value=float('NaN'), bounds_error=False)
            return [Observation(query_loc, float(sci_interp((query_loc.lon, query_loc.lat, query_time))), query_time) for query_loc, query_time in zip(query_locs, query_times) if self.withinBounds(query_loc, loc_type=loc_type)]


      else:
        if loc_type == "xy":
          if query_type == 'current':
            u_interp = RegularGridInterpolator((self.x_ticks, self.y_ticks), self.current_u_field[:,:,0], fill_value=float('NaN'), bounds_error=False)
            v_interp = RegularGridInterpolator((self.x_ticks, self.y_ticks), self.current_v_field[:,:,0], fill_value=float('NaN'), bounds_error=False)

            u_obs = [Observation(query_loc, float(u_interp((query_loc.x, query_loc.y))), query_time) for query_loc, query_time in zip(query_locs, query_times) if self.withinBounds(query_loc, loc_type=loc_type)]
            v_obs = [Observation(query_loc, float(v_interp((query_loc.x, query_loc.y))), query_time) for query_loc, query_time in zip(query_locs, query_times) if self.withinBounds(query_loc, loc_type=loc_type)]

            return u_obs, v_obs
          else:
            science_field = self.science_fields[query_type][:,:,0]
            sci_interp = RegularGridInterpolator((self.x_ticks, self.y_ticks), science_field, fill_value=float('NaN'), bounds_error=False)
            return [Observation(query_loc, float(sci_interp((query_loc.x, query_loc.y))), query_time) for query_loc, query_time in zip(query_locs, query_times) if self.withinBounds(query_loc, loc_type=loc_type)]


        elif loc_type == "latlon":
          if query_type == 'current':
            u_interp = RegularGridInterpolator((self.lon_ticks, self.lat_ticks), self.current_u_field[:,:,0], fill_value=float('NaN'), bounds_error=False)
            v_interp = RegularGridInterpolator((self.lon_ticks, self.lat_ticks), self.current_v_field[:,:,0], fill_value=float('NaN'), bounds_error=False)

            u_obs = [Observation(query_loc, float(u_interp((query_loc.lon, query_loc.lat))), query_time) for query_loc, query_time in zip(query_locs, query_times) if self.withinBounds(query_loc, loc_type=loc_type)]
            v_obs = [Observation(query_loc, float(v_interp((query_loc.lon, query_loc.lat))), query_time) for query_loc, query_time in zip(query_locs, query_times) if self.withinBounds(query_loc, loc_type=loc_type)]

            return u_obs, v_obs
          else:
            science_field = self.science_fields[query_type][:,:,0]
            sci_interp = RegularGridInterpolator((self.lon_ticks, self.lat_ticks), science_field, fill_value=float('NaN'), bounds_error=False)
            return [Observation(query_loc, float(sci_interp((query_loc.lon, query_loc.lat))), query_time) for query_loc, query_time in zip(query_locs, query_times) if self.withinBounds(query_loc, loc_type=loc_type)]


  def getSnapshot(self, ss_time, snapshot_type='scalar_field'):
    time_dist = [abs(ss_time - x) for x in self.t_ticks]#
    snapshot_time_idx = time_dist.index(min(time_dist))


    if snapshot_type == 'obstacle_field':
      return self.obstacle_field[:,:,snapshot_time_idx]

    elif snapshot_type == 'current_u_field':
      return self.current_u_field[:,:,snapshot_time_idx]

    elif snapshot_type == 'current_v_field':
      return self.current_v_field[:,:,snapshot_time_idx]

    else:
      try:
        return self.science_fields[snapshot_type][:,:,snapshot_time_idx]
      except ValueError:
        raise ValueError("Unknown snapshot_type %s" % snapshot_type)


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


  def draw(self, ax, block=True, show=False, cbar_max=None, cbar_min=None, quiver_stride=None, snapshot_time=None, draw_currents=True, cmap='Greys', quiver_color='black', loc_type='xy', science_type='scalar_field'):


    if snapshot_time is None:
      ss_scalar_field = self.getSnapshot(self.t_ticks[0], science_type)
      ss_obstacle_field = self.getSnapshot(self.t_ticks[0], 'obstacle_field')
      ss_current_u_field = self.getSnapshot(self.t_ticks[0], 'current_u_field')
      ss_current_v_field = self.getSnapshot(self.t_ticks[0], 'current_v_field')
    else:
      ss_scalar_field = self.getSnapshot(snapshot_time, science_type)
      ss_obstacle_field = self.getSnapshot(snapshot_time, 'obstacle_field')
      ss_current_u_field = self.getSnapshot(snapshot_time, 'current_u_field')
      ss_current_v_field = self.getSnapshot(snapshot_time, 'current_v_field')


    masked_field = np.ma.MaskedArray(data=ss_scalar_field, mask=ss_obstacle_field)

    if cbar_min is None:
      cbar_min = np.min(masked_field)
    if cbar_max is None:
      cbar_max = np.max(masked_field)

    num_format  = '%.0f'
    formatter = tick.FormatStrFormatter(num_format)

    if loc_type == 'xy':
      CS = plt.pcolor(self.x_ticks, self.y_ticks, masked_field.transpose(), cmap=cmap, vmin=cbar_min, vmax=cbar_max)

      if draw_currents:
        if quiver_stride is None:
          quiver_stride = len(self.x_ticks) / 20
        quiver = plt.quiver(self.x_ticks[::quiver_stride], self.y_ticks[::quiver_stride], ss_current_u_field.transpose()[::quiver_stride, ::quiver_stride], ss_current_v_field.transpose()[::quiver_stride, ::quiver_stride], color=quiver_color)
        quiver_key = plt.quiverkey(quiver, 0.95, 1.05, 0.2, "0.2 m/s", labelpos='E', coordinates='axes')

      ax.get_xaxis().set_major_formatter(formatter)
      ax.get_yaxis().set_major_formatter(formatter)
      plt.ylim([np.min(self.x_ticks), np.max(self.x_ticks)])
      plt.xlim([np.min(self.y_ticks), np.max(self.y_ticks)])
      plt.title("Ground Truth World")
      plt.xlabel("X (km)")
      plt.ylabel("Y (km)")
      divider = make_axes_locatable(ax)
      cax = divider.append_axes("right", size="5%", pad=0.05)
      cbar = plt.colorbar(CS, format='%.1f', cax=cax)
      cbar.set_label(science_type)
      ax.axis('scaled')
    elif loc_type == 'latlon':
      CS = plt.pcolor(self.lon_ticks, self.lat_ticks, masked_field.transpose(), cmap=cmap, vmin=cbar_min, vmax=cbar_max)

      if draw_currents:
        if quiver_stride is None:
          quiver_stride = len(self.lon_ticks) / 20
        quiver = plt.quiver(self.lon_ticks[::quiver_stride], self.lat_ticks[::quiver_stride], ss_current_u_field.transpose()[::quiver_stride, ::quiver_stride], ss_current_v_field.transpose()[::quiver_stride, ::quiver_stride], color=quiver_color)
        quiver_key = plt.quiverkey(quiver, 0.95, 1.05, 0.2, "0.2 m/s", labelpos='E', coordinates='axes')

      ax.get_xaxis().set_major_formatter(formatter)
      ax.get_yaxis().set_major_formatter(formatter)
      plt.ylim([np.min(self.lon_ticks), np.max(self.lon_ticks)])
      plt.xlim([np.min(self.lat_ticks), np.max(self.lat_ticks)])
      plt.title("Ground Truth World")
      plt.xlabel("X (lon)")
      plt.ylabel("Y (lat)")
      divider = make_axes_locatable(ax)
      cax = divider.append_axes("right", size="5%", pad=0.05)
      cbar = plt.colorbar(CS, format='%.1f', cax=cax)
      cbar.set_label(science_type)
      ax.axis('scaled')
    else:
      print "Unrecognized loc type"

    if show:
      plt.show(block)

  def getRandomLocationXY(self):
    return Location(xlon=random.choice(self.x_ticks), ylat=random.choice(self.y_ticks))

  def getRandomLocationLatLon(self):
    return Location(xlon=random.choice(self.lon_ticks), ylat=random.choice(self.lat_ticks))

  @classmethod
  def donut(cls):
    x, y = np.mgrid[-1:1:.01, -1:1:.01]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    rv1 = stats.multivariate_normal([0, -0], [[.15, 0.0], [0.0, .15]])
    m1 = rv1.pdf(pos)
    m1 = m1 / np.max(m1)
    rv2 = stats.multivariate_normal([0, -0], [[.05, 0.0], [0.0, .05]])
    m2 = rv2.pdf(pos)
    m2 = m2 / np.max(m2)
    scalar_field = m1 - m2
    scalar_field = np.expand_dims(scalar_field / np.sum(scalar_field), 2)

    x_ticks = np.arange(0, scalar_field.shape[0])
    y_ticks = np.arange(0, scalar_field.shape[1])
    t_ticks = np.arange(0, scalar_field.shape[2])

    u_field = np.zeros(scalar_field.shape)
    v_field = np.zeros(scalar_field.shape)

    lat_ticks = y_ticks
    lon_ticks = x_ticks

    bounds = [np.max(y_ticks), np.min(y_ticks), np.max(x_ticks), np.min(y_ticks)]

    resolution = 1.0

    return cls(['temperature'], [scalar_field], u_field, v_field, x_ticks, y_ticks, t_ticks, lon_ticks, lat_ticks, resolution, resolution, bounds)

  @classmethod
  def tripleDonut(cls):
    x, y = np.mgrid[-1:1:.01, -1:1:.01]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y


    locs = [Location(-.25, -.25), Location(-.25, .25), Location(.25, 0)]
    scales = [0.5, 0.5, 0.5]

    scalar_field = np.zeros((pos.shape[0], pos.shape[1]))

    for loc, scale in zip(locs, scales):
      rv1 = stats.multivariate_normal([loc.x, loc.y], [[.15*scale, 0.0], [0.0, .15*scale]])
      m1 = rv1.pdf(pos)
      m1 = m1 / np.max(m1)
      rv2 = stats.multivariate_normal([loc.x, loc.y], [[.05*scale, 0.0], [0.0, .05*scale]])
      m2 = rv2.pdf(pos)
      m2 = m2 / np.max(m2)
      res = m1 - m2
      res = res / np.sum(res)
      scalar_field = np.maximum(scalar_field, res)

    scalar_field = np.expand_dims(scalar_field, 2)

    x_ticks = np.arange(0, scalar_field.shape[0])
    y_ticks = np.arange(0, scalar_field.shape[1])
    t_ticks = np.arange(0, scalar_field.shape[2])

    u_field = np.zeros(scalar_field.shape)
    v_field = np.zeros(scalar_field.shape)

    lat_ticks = y_ticks
    lon_ticks = x_ticks

    bounds = [np.max(y_ticks), np.min(y_ticks), np.max(x_ticks), np.min(y_ticks)]

    resolution = 1.0

    return cls(['temperature'], [scalar_field], u_field, v_field, x_ticks, y_ticks, t_ticks, lon_ticks, lat_ticks, resolution, resolution, bounds)

  @classmethod
  def randomDonut(cls, n_donuts):
    x, y = np.mgrid[-1:1:.01, -1:1:.01]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y


    scalar_field = np.zeros((pos.shape[0], pos.shape[1]))
    for ii in range(n_donuts):
      center_loc = Location(random.random() * 2. - 1., random.random() * 2. - 1.)
      outer_radius = random.random() / 2. + 0.3
      inner_radius = random.random() * outer_radius


      rv1 = stats.multivariate_normal([center_loc.x, center_loc.y], [[.15*outer_radius, 0.0], [0.0, .15*outer_radius]])
      m1 = rv1.pdf(pos)
      m1 = m1 / np.max(m1)
      rv2 = stats.multivariate_normal([center_loc.x, center_loc.y], [[.15*inner_radius, 0.0], [0.0, .15*inner_radius]])
      m2 = rv2.pdf(pos)
      m2 = m2 / np.max(m2)
      res = m1 - m2
      res = res / np.sum(res)
      scalar_field = np.maximum(scalar_field, res)

    scalar_field = np.expand_dims(scalar_field, 2)

    x_ticks = np.arange(0, scalar_field.shape[0])
    y_ticks = np.arange(0, scalar_field.shape[1])
    t_ticks = np.arange(0, scalar_field.shape[2])

    u_field = np.zeros(scalar_field.shape)
    v_field = np.zeros(scalar_field.shape)

    lat_ticks = y_ticks
    lon_ticks = x_ticks

    bounds = [np.max(y_ticks), np.min(y_ticks), np.max(x_ticks), np.min(y_ticks)]

    resolution = 1.0

    return cls(['temperature'], [scalar_field], u_field, v_field, x_ticks, y_ticks, t_ticks, lon_ticks, lat_ticks, resolution, resolution, bounds)


  @classmethod
  def randomHotspots(cls, world_center, world_width, world_height, world_resolution, science_variable):
    bounds = getBox(xlen=world_width, ylen=world_height, center=world_center)

    n_bound   = bounds[0]
    s_bound   = bounds[1]
    e_bound   = bounds[2]
    w_bound   = bounds[3]

    x_ticks   = np.arange(0.0, world_width+world_resolution[0], world_resolution[0])
    y_ticks   = np.arange(0.0, world_height+world_resolution[1], world_resolution[1])

    x_ticks = x_ticks - np.max(x_ticks)/2.
    y_ticks = y_ticks - np.max(y_ticks)/2.

    lon_ticks = np.linspace(w_bound, e_bound, len(x_ticks))
    lat_ticks = np.linspace(s_bound, n_bound, len(y_ticks))

    scalar_field = np.zeros((len(x_ticks), len(y_ticks)))

    # x, y = np.mgrid[-1:1:.01, -1:1:.01]
    xx, yy = np.meshgrid(x_ticks, y_ticks)
    pos = np.empty(xx.shape + (2,))
    pos[:, :, 0] = xx
    pos[:, :, 1] = yy


    for ii in range(100):
      center_loc = Location(xlon=min(x_ticks) + random.random()*(max(x_ticks) - min(x_ticks)),
                            ylat=min(y_ticks) + random.random()*(max(y_ticks) - min(y_ticks)))
      radius = random.random()  * len(x_ticks) / 2.

      rv1 = stats.multivariate_normal([center_loc.x, center_loc.y], [[radius, 0.0], [0.0, radius]])
      m1 = rv1.pdf(pos)
      m1 = m1 / np.max(m1)

      scalar_field = scalar_field + m1

    scalar_field = scalar_field / np.max(scalar_field)


    scalar_field = np.expand_dims(scalar_field, 2)

    t_ticks = np.arange(0, scalar_field.shape[2])

    u_field = np.zeros(scalar_field.shape)
    v_field = np.zeros(scalar_field.shape)

    scalar_field = np.ma.masked_greater(scalar_field, float('inf'))
    u_field = np.ma.masked_greater(u_field, float('inf'))
    v_field = np.ma.masked_greater(v_field, float('inf'))

    return cls(science_variable, scalar_field**2, u_field, v_field, x_ticks, y_ticks, t_ticks, lon_ticks, lat_ticks, world_resolution, world_resolution, bounds)


  @classmethod
  def gyre(cls, world_center, world_width, world_height, world_resolution, science_variable):


    bounds = getBox(xlen=world_width, ylen=world_height, center=world_center)

    n_bound   = bounds[0]
    s_bound   = bounds[1]
    e_bound   = bounds[2]
    w_bound   = bounds[3]

    x_ticks   = np.arange(0.0, world_width+world_resolution[0], world_resolution[0])
    y_ticks   = np.arange(0.0, world_height+world_resolution[1], world_resolution[1])

    x_ticks = x_ticks - np.max(x_ticks)/2.
    y_ticks = y_ticks - np.max(y_ticks)/2.

    lon_ticks = np.linspace(w_bound, e_bound, len(x_ticks))
    lat_ticks = np.linspace(s_bound, n_bound, len(y_ticks))


    generator_x = np.linspace(0, 2*math.pi, len(x_ticks))
    generator_y = np.linspace(math.pi, 3*math.pi, len(y_ticks))


    xx, yy = np.meshgrid(generator_x, generator_y)
    xx = xx + math.pi /2

    u_field = np.sin(yy) * np.sin(xx)
    v_field = np.sin(xx + math.pi/2) * np.sin(yy + math.pi/2)

    scalar_field = np.sqrt(u_field*u_field + v_field*v_field)
    scalar_field = scalar_field / np.max(scalar_field)

    scalar_field = np.expand_dims(scalar_field, 2)
    u_field = np.expand_dims(u_field, 2)
    v_field = np.expand_dims(v_field, 2)

    t_ticks = np.arange(0, scalar_field.shape[2])

    scalar_field = np.ma.masked_greater(scalar_field, float('inf'))
    u_field = np.ma.masked_greater(u_field, float('inf'))
    v_field = np.ma.masked_greater(v_field, float('inf'))

    return cls([science_variable], [scalar_field], u_field, v_field, x_ticks, y_ticks, t_ticks, lon_ticks, lat_ticks, world_resolution, world_resolution, bounds)

  # loadWorldYaml

  @classmethod
  def loadWorldYaml(cls, yaml_world, project_path):
    world_key = yaml_world['world_type']


    if 'roms' in world_key:
      roms_yaml = yaml_world[world_key]
      wd = loadWorld(
        roms_file = project_path + roms_yaml['roms_file'],
        world_center = Location(ylat=roms_yaml['center_latitude'], xlon=roms_yaml['center_longitude']),
        world_width = roms_yaml['width'],
        world_height = roms_yaml['height'],
        world_resolution = (roms_yaml['resolution'], roms_yaml['resolution']),
        science_variable = roms_yaml['science_variable'],
        save_dir = project_path + 'data/worlds/')

    elif 'gyre' == world_key:
      gyre_yaml = yaml_world[world_key]
      wd = World.gyre(
        world_center = Location(ylat=gyre_yaml['center_latitude'], xlon=gyre_yaml['center_longitude']),
        world_width = gyre_yaml['width'],
        world_height = gyre_yaml['height'],
        world_resolution = (gyre_yaml['resolution'], gyre_yaml['resolution']),
        science_variable = gyre_yaml['science_variable'])

    elif 'random' in world_key:
      random_yaml = yaml_world[world_key]
      wd = World.randomHotspots(
        world_center = Location(ylat=random_yaml['center_latitude'], xlon=random_yaml['center_longitude']),
        world_width = random_yaml['width'],
        world_height = random_yaml['height'],
        world_resolution = (random_yaml['resolution'], random_yaml['resolution']),
        science_variable = random_yaml['science_variable'])

    else:
      raise ValueError("Invalid world type %s" % world_key)


    if yaml_world['use_gradient']:
      if len(wd.scalar_field.shape) > 2:
        for t_idx in range(wd.scalar_field.shape[2]):
          wd.scalar_field[:,:,t_idx] = gradient2d(wd.scalar_field[:,:,t_idx])
          wd.current_u_field[:,:,t_idx] = gradient2d(wd.current_u_field[:,:,t_idx])
          wd.current_v_field[:,:,t_idx] = gradient2d(wd.current_v_field[:,:,t_idx])
      else:
        wd.scalar_field = gradient2d(wd.scalar_field)

    return wd


  @classmethod
  def roms(cls, datafile_path, xlen, ylen, center, feature=['temperature'], resolution=(0.1, 0.1)):
    if not isinstance(feature, list):
        feature = [feature]

    # World bounds
    bounds = getBox(xlen=xlen, ylen=ylen, center=center)

    n_bound   = bounds[0]
    s_bound   = bounds[1]
    e_bound   = bounds[2]
    w_bound   = bounds[3]

    x_ticks   = np.arange(0.0, xlen+(resolution[0]/2.), resolution[0])
    y_ticks   = np.arange(0.0, ylen+(resolution[1]/2.), resolution[1])

    x_ticks = x_ticks - np.max(x_ticks)/2.
    y_ticks = y_ticks - np.max(y_ticks)/2.

    lon_ticks = np.linspace(w_bound, e_bound, len(x_ticks))
    lat_ticks = np.linspace(s_bound, n_bound, len(y_ticks))

    scalar_fields = []



    current_u, u_lat, u_lon, roms_t = getROMSData(datafile_path, 'u')
    current_v, v_lat, v_lon, _ = getROMSData(datafile_path, 'v')

    output_shape = (len(x_ticks), len(y_ticks), len(roms_t))

    for i, feat in enumerate(feature):
        scalar_field, scalar_lat, scalar_lon, roms_t = getROMSData(datafile_path, feat)
        scalar_field = reshapeROMS(scalar_field, scalar_lat, scalar_lon, bounds, output_shape)
        scalar_fields.append(scalar_field)

    current_u = reshapeROMS(current_u, u_lat, u_lon, bounds, output_shape)
    current_v = reshapeROMS(current_v, v_lat, v_lon, bounds, output_shape)

    return cls(feature, scalar_fields, current_u, current_v, x_ticks, y_ticks, roms_t, lon_ticks, lat_ticks, resolution[0], resolution[1], bounds)


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

    return cls(['temperature'], [res_scalar_field], res_current_u_field, res_current_v_field, x_ticks, y_ticks, t_ticks, lon_ticks, lat_ticks, resolution[0], resolution[1], bounds)

  @classmethod
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

    return cls(['temperature'], [res], x_ticks, y_ticks, t_ticks, resolution[0], resolution[1], bounds)

  @classmethod
  def loadH5(cls, file_path):
    target_file = os.path.expandvars(file_path)

    if os.path.isfile(target_file):
      print( "%s Exists" % target_file )
      wd_dict = dd.io.load(target_file)

    sci_types = wd_dict['science_fields'].keys()
    sci_fields = [wd_dict['science_fields'][sci_key] for sci_key in sci_types]

    sci_fields[0] = np.ma.MaskedArray(sci_fields[0], mask=wd_dict['obstacle_field'])

    current_u_field = wd_dict['current_u_field']
    current_v_field = wd_dict['current_v_field']

    x_ticks = wd_dict['x_ticks']
    y_ticks = wd_dict['y_ticks']
    t_ticks = wd_dict['t_ticks']

    lat_ticks = wd_dict['lat_ticks']
    lon_ticks = wd_dict['lon_ticks']
    cell_x_size = wd_dict['cell_x_size']
    cell_y_size = wd_dict['cell_y_size']
    bounds = wd_dict['bounds']

    print("Loaded World from File")
    return cls(list(sci_types), sci_fields, current_u_field, current_v_field, x_ticks, y_ticks, t_ticks, lon_ticks, lat_ticks, cell_x_size, cell_y_size, bounds)

  def saveH5(self, file_path):
    dd.io.save(os.path.expandvars(file_path), self.__dict__, compression='zlib')

def loadWorld(roms_file, world_center, world_width, world_height, world_resolution, science_variable=['temperature'], save_dir=None):

  if save_dir is not None:
    filename_pattern = re.compile('.*/(.*).nc')
    world_file_name = "%s_lat_%.5f_lon_%.5f_width_%.3f_height_%.3f_resolution_%dm_%dm_%s.h5" % (
      re.findall(filename_pattern, roms_file)[0],
      world_center.lat,
      world_center.lon,
      world_width,
      world_height,
      int(world_resolution[0] * 1000),
      int(world_resolution[1] * 1000),
      science_variable
      )

    world_file_full_path = os.path.expandvars(save_dir + world_file_name)
    if os.path.isfile(world_file_full_path):
      print( "%s Exists" % world_file_full_path )
      wd = World.loadH5(world_file_full_path)#dd.io.load(world_file_full_path)
      print( "Loaded World from File" )

    else:
      print( "%s Does Not Exist" % world_file_full_path )
      wd = World.roms(
        datafile_path = roms_file,
        xlen          = world_width,
        ylen          = world_height,
        center        = world_center,
        feature       = science_variable,
        resolution    = world_resolution
        )

      # pdb.set_trace()

      # dd.io.save(world_file_full_path, wd, compression='zlib')
      wd.saveH5(world_file_full_path)
      print( "World saved to %s" % world_file_full_path )


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

  n_bound = 29.0
  s_bound = 28.0
  e_bound = -94.0
  w_bound = -95.0

  bounds = [n_bound, s_bound, e_bound, w_bound]

  d1 = datetime.datetime(2018, 1, 1)
  d2 = datetime.datetime(2018, 1, 2)
  bounds = getBox(xlen = 20, ylen = 20, center = Location(0.0,0.0))

  wd = World.idealizedFront(
    start_date      = d1,
    end_date      = d2,
    time_resolution   = 24,
    resolution      = (0.100, 0.100),
    xlen        = 15.,
    ylen        = 20.,
  )
  # wd = World.random(d1, d2, 24, (100, 110), bounds=bounds, num_generators = 50)
  #datafile_path = os.path.dirname(os.path.realpath(__file__)) + "/../data/roms_data/"
  #datafile_name = "txla_roms/txla_hindcast_jun_1_2015.nc"

  #wd = World.roms(datafile_path + datafile_name, 20, 20, Location(xlon=-94.25, ylat=28.25), feature=['salt'], resolution=(0.1, 0.1))



  print( "Generating Figures" )

  for t_idx, t in enumerate(wd.t_ticks):
    fig = plt.figure()
    plt.clf()
    plt.title(str(datetime.datetime.fromtimestamp(wd.t_ticks[t_idx])))
    img = plt.pcolor(wd.lon_ticks, wd.lat_ticks, wd.scalar_fields[0][:, :, t_idx].transpose(), vmin=np.min(wd.scalar_fields[0]), vmax=np.max(wd.scalar_fields[0]))
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
