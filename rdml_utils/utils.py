#!/usr/bin/env python
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
import numpy as np

from shapely.geometry import Point, Polygon, LineString
import math, operator, datetime, pdb, haversine, os, errno, sys
if sys.version_info[0] < 3:
    # python 2
    import Queue
    from location import Location, Observation, StampedLocation
else:
    # python 3
    import queue as Queue
    from rdml_utils.location import Location, Observation, StampedLocation

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
#from matplotlib.mlab import bivariate_normal




####################################
## Utility Classes
####################################


class EndOfSimException(Exception):
  pass


class QueueSet(object):
  """docstring for QueueSet"""
  def __init__(self):
    self.queue = Queue.Queue()
    self.queue_set = []

  def __len__(self):
    return len(self.queue_set)

  def __contains__(self, item):
    return query in self.queue_set

  def qsize(self):
    return self.queue.qsize()

  def push(self, item):
    if item not in self.queue_set:
      self.queue.put(item)
      self.queue_set.append(item)
      return True
    else:
      return False

  def pop(self):
    try:
      item = self.queue.get_nowait()
    except Queue.Empty:
      return None

    self.queue_set.remove(item)
    return item

  def hasElement(self, query):
    return query in self.queue_set

  def empty(self):
    return self.queue.empty()


class PriorityQueueSet(QueueSet):
  """docstring for PriorityQueueSet"""
  def __init__(self):
    QueueSet.__init__(self)
    self.queue = Queue.PriorityQueue()

  def push(self, item, priority):
    if item not in self.queue_set:
      self.queue.put((priority, item))
      self.queue_set.append(item)
      return True
    else:
      return False


  def pop(self):
    try:
      item = self.queue.get_nowait()[1]
    except Queue.Empty:
      return None
    self.queue_set.remove(item)
    return item

class State():

  def __init__(self, loc=Location(0,0), heading=0.0, goal=None):

    self.loc = loc
    self.heading = heading
    self.goal = goal

  @classmethod
  def fromTuple(cls, tup):
    return cls(loc=Location(xlon=tup[0], ylat=tup[1]), heading=tup[2], goal=tup[3])

  def __str__(self):
    return "Location: " + str(self.loc) + " Heading: %d" % self.heading + " Goal: %d" % self.goal

  def asTuple(self):
    return (self.loc.x, self.loc.y, self.heading, self.goal)

  def asTupleNoGoal(self):
    return (self.loc.x, self.loc.y, self.heading)

class genState():

  def __init__(self, d_cur, t_cur, d_next, t_next, g_index):
    self.d_cur = d_cur
    self.t_cur = t_cur
    self.d_next = d_next
    self.t_next = t_next
    self.g_index = g_index

  @classmethod
  def fromTuple(cls, tup):
    return cls(tup[0], tup[1], tup[2], tup[3], tup[4])

  @classmethod
  def fromLoc(cls, loc, goal_list, g_index, cur_h=0.0):
    if g_index < len(goal_list):
      d_cur = (loc - goal_list[g_index]).getMagnitude()
      t_cur = headingBetLocs(loc, goal_list[g_index]) - cur_h
      if g_index+1 < len(goal_list):
        d_next = (loc - goal_list[g_index+1]).getMagnitude()
        t_next = headingBetLocs(loc, goal_list[g_index+1]) - cur_h
      else:
        d_next = d_cur
        t_next = t_cur
      return cls(d_cur, t_cur, d_next, t_next, g_index)
    else:
      return None

  def asTuple(self):
    return (self.d_cur, self.t_cur, self.d_next, self.t_next, self.g_index)

  def asTupleNoGoal(self):
    return (self.d_cur, self.t_cur, self.d_next, self.t_next)

  def __str__(self):
    return str(self.asTuple())

####################################
## Utility Functions
####################################

def headingBetLocs(loc1, loc2):
  d = loc2 - loc1
  return math.atan2(d.d_ylat, d.d_xlon)

def getBox(xlen, ylen, center):
  # Given center point (coords), size in km, return bounds (coords)
  n_bound = getLatLon(center, ylen/2., 'north').lat
  s_bound = getLatLon(center, ylen/2., 'south').lat
  e_bound = getLatLon(center, xlen/2., 'east').lon
  w_bound = getLatLon(center, xlen/2., 'west').lon
  return [n_bound, s_bound, e_bound, w_bound]

def getLatLon(curr_pt, distance, bearing):
  # Gets the lat, lon coordinates of a point 'distance' km away at a heading of 'bearing' in degrees
  earth_r = 6378.1

  if bearing == "north":
    bearing = math.radians(0)
  elif bearing == 'south':
    bearing = math.radians(180)
  elif bearing == 'east':
    bearing = math.radians(90)
  elif bearing == 'west':
    bearing = math.radians(270)
  else:
    bearing = math.radians(bearing)

  lat1 = math.radians(curr_pt.lat)
  lon1 = math.radians(curr_pt.lon)

  lat2 = math.asin( math.sin(lat1)*math.cos(distance/earth_r) + math.cos(lat1)*math.sin(distance/earth_r)*math.cos(bearing))

  lon2 = lon1 + math.atan2(math.sin(bearing)*math.sin(distance/earth_r)*math.cos(lat1), math.cos(distance/earth_r)-math.sin(lat1)*math.sin(lat2))

  lat2 = math.degrees(lat2)
  lon2 = math.degrees(lon2)

  return Location(ylat=lat2,xlon=lon2)

def getTimestamp(dt):
  return (dt - datetime.datetime(1970, 1, 1)).total_seconds()

def dateRange(d1, d2, step):
  curr = d1
  while curr < d2:
    yield curr
    curr += step


def dateLinspace(d1, d2, n):
  datestep = (d2 - d1)/(n-1)
  for ii in range(n):
    yield d1 + ii * datestep


def ptInList(pt, l):
  return np.any(np.all(l == pt, axis=1))


def findMax(z, neighborhood_size=5):
  data_max = filters.maximum_filter(z, neighborhood_size)
  maxima = (z == data_max)
  labeled, num_objects = ndimage.label(maxima)
  slices = ndimage.find_objects(labeled)
  res = []
  # for dy, dx in slices: # This is the old version
  for dx, dy in slices:
    x_center = (dx.start + dx.stop - 1)/2
    y_center = (dy.start + dy.stop - 1)/2

    res.append(Location(ylat=y_center, xlon=x_center))

  return res


def findMin(z, neighborhood_size=5):
  data_min = filters.minimum_filter(z, neighborhood_size)
  minima = (z == data_min)
  labeled, num_objects = ndimage.label(minima)
  slices = ndimage.find_objects(labeled)
  res = []
  # for dy, dx in slices: # This is the old version
  for dx, dy in slices:
    x_center = (dx.start + dx.stop - 1)/2
    y_center = (dy.start + dy.stop - 1)/2

    res.append(Location(ylat=y_center, xlon=x_center))

  return res



def floodFill(mat, label, pixel):
  if mat[tuple(pixel)] == label:
    return
  elif mat[tuple(pixel)] == 1:
    mat[tuple(pixel)] = label

  for n in [x for x in getNeighbors(pixel, mat) if mat[tuple(x)]]:
    floodFill(mat, label, n)


def getNeighbors(loc, mat, step_size=1, connectivity=4):
  #Get List of Neighbors, Ignoring ones in obstacles or ones out of bounds
  if connectivity == 4:
    translations = [[1, 0], [-1, 0], [0, 1], [0, -1]]
  elif connectivity == 8:
    translations = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1]]

  neighbors = []
  for t in translations:
    query_loc = map(operator.add, loc, t)
    if isFree(query_loc, mat):
      neighbors.append(map(operator.add, loc, t))
  return map(tuple, neighbors)



def isFree(loc, mat):
  if loc[0] < 0 or loc[0] >= mat.shape[0]:
    return False
  elif loc[1] < 0 or loc[1] >= mat.shape[1]:
    return False
  elif mat[tuple(loc)] == 0:
    return False
  else:
    return True

def loc2cell(loc, bounds, mat_shape):
  #get the cell # of a lat/lon location
  # pdb.set_trace()

  # loc = [lon, lat]

  n_bound = bounds[0]
  s_bound = bounds[1]
  e_bound = bounds[2]
  w_bound = bounds[3]

  n_s_step = float(n_bound - s_bound)/mat_shape[1]
  e_w_step = float(e_bound - w_bound)/mat_shape[0]


  cell = [0,0]
  cell[1] = int((loc[0]-w_bound)/e_w_step)  #Longitude
  cell[0] = int((loc[1]-s_bound)/n_s_step)  #Latitude
  return cell

def cell2loc(cell, bounds, mat_shape):
  #get the cell # of a lat/lon location

  # returns loc = [lon, lat]


  n_bound = bounds[0]
  s_bound = bounds[1]
  e_bound = bounds[2]
  w_bound = bounds[3]

  n_s_step = float(n_bound - s_bound)/mat_shape[1]
  e_w_step = float(e_bound - w_bound)/mat_shape[0]

  loc = [0,0]
  loc[1] = cell[0]*n_s_step + s_bound #Latitude
  loc[0] = cell[1]*e_w_step + w_bound #Longitude
  return loc



def euclideanDist(p1, p2):
  if isinstance(p1, Location) and isinstance(p2, Location):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y-p2.y)**2)

  elif isinstance(p1, Location) and isinstance(p2, StampedLocation):
    return math.sqrt((p1.x - p2.loc.x)**2 + (p1.y-p2.loc.y)**2)

  elif isinstance(p1, StampedLocation) and isinstance(p2, Location):
    return math.sqrt((p1.loc.x - p2.x)**2 + (p1.loc.y-p2.y)**2)

  elif isinstance(p1, StampedLocation) and isinstance(p2, StampedLocation):
    return math.sqrt((p1.loc.x - p2.loc.x)**2 + (p1.loc.y-p2.loc.y)**2)

  else:
    #You know what this does
    dist = [(a - b)**2 for a, b in zip(p1, p2)]
    return math.sqrt(sum(dist))


def manhattanDist(p1, p2):
  if isinstance(p1, Location) and isinstance(p2, Location):
    return abs(p1.x - p2.x) + abs(p2.y - p2.y)
  else:
  #This too
    res = 0
    for pair in zip(p1, p2):
      res += abs(pair[1] - pair[0])

    return res

def haversineDist(p1, p2):
  if isinstance(p1, Location) and isinstance(p2, Location):
    return haversine.haversine((p1.lat, p1.lon), (p2.lat, p2.lon))
  else:
    return haversine.haversine(p1, p2)

def getFencedData(scalar_field, geofence, roms_lat, roms_lon):

  n_roms_bound = np.max(roms_lat)
  s_roms_bound = np.min(roms_lat)
  e_roms_bound = np.max(roms_lon)
  w_roms_bound = np.min(roms_lon)

  roms_bounds = [n_roms_bound, s_roms_bound, e_roms_bound, w_roms_bound]

  n_fence_bound = max([x.lat for x in geofence])
  s_fence_bound = min([x.lat for x in geofence])
  e_fence_bound = max([x.lon for x in geofence])
  w_fence_bound = min([x.lon for x in geofence])
  fence_bounds = [n_fence_bound, s_fence_bound, e_fence_bound, w_fence_bound]

  exterior_lat = [[x, idx] for idx, x in enumerate(roms_lat) if x > n_fence_bound or x < s_fence_bound]
  exterior_lon = [[x, idx] for idx, x in enumerate(roms_lon) if x > e_fence_bound or x < w_fence_bound]


  north_bound_pt, north_bound_key = min(exterior_lat, key=lambda x: abs(x[0]-n_fence_bound))
  south_bound_pt, south_bound_key = min(exterior_lat, key=lambda x: abs(x[0]-s_fence_bound))
  east_bound_pt, east_bound_key = min(exterior_lon, key=lambda x: abs(x[0]-e_fence_bound))
  west_bound_pt, west_bound_key = min(exterior_lon, key=lambda x: abs(x[0]-w_fence_bound))


  print( fence_bounds )
  print( north_bound_pt, north_bound_key )
  print( south_bound_pt, south_bound_key )
  print( east_bound_pt, east_bound_key )
  print( west_bound_pt, west_bound_key )


  fenced_lat = roms_lat[south_bound_key:north_bound_key+1]
  fenced_lon = roms_lon[west_bound_key:east_bound_key+1]

  fenced_scalar_field = scalar_field[south_bound_key:north_bound_key+1, west_bound_key:east_bound_key+1]

  return fence_bounds, fenced_lat, fenced_lon, fenced_scalar_field

def normalizeField(field):
  field_max = np.max(field)

  if field_max != 0.0:
    res = field / field_max
    return res
  elif field_max == 0.0 and np.min(field) == 0.0:
    return field



def castGaussian(loc, variance, xx, yy, magnitude=1.0):
  #Scoops out novely defined by variance and magnitude at location
  #res = bivariate_normal(xx, yy, 0.5*variance, 0.5*variance, loc.x, loc.y)

  # updated to modern numpy rather than archiac matlab like bivariate normal :p
  res = np.random.multivariate_normal([loc.x, loc.y],[[0.5*variance, 0],[0, 0.5*variance]], size=(xx,yy))

  return 1. - normalizeField(res - np.min(res)) * magnitude


def discountNovelty(novelty, dt, temporal_discount):
  return 1. - ((1. - novelty) * np.exp(-dt/temporal_discount))


def initializeNovelty(observations, xx, yy, variance, temporal_discount, obs_magnitude):
  novelty = np.ones(xx.shape)
  sorted_obs = sorted(observations, key=lambda x: x.time)
  prev_time = sorted_obs[0].time
  for obs in sorted_obs:
    dt = obs.time - prev_time
    novelty = discountNovelty(novelty, dt, temporal_discount)
    novelty = novelty * castGaussian(obs.loc, variance, xx, yy, obs_magnitude)
    prev_time = obs.time
  return novelty


def movingAverageDelta(l, n):
  if len(l) == 0 or len(l) == 1 or n == 1:
    return 0.0
  else:
    l_window = l[-n:]
    res = 0.0
    for p2, p1 in zip(l_window[:-1], l_window[1:]):
      res = res + (p1 - p2)
    return res/(len(l_window) - 1)


def arrayRound(arr, val):
  #rounds each element in array to nearest val
  return np.round(arr / val) * val



def timeSeriesGradient(input,window_width):

  # Padding
  half_window    = (int)(math.ceil(window_width/2))
  input_pad   = np.pad(input,(half_window,half_window),'edge')

  # Kernel
  kernel_domain = np.linspace(-np.pi,np.pi,num=window_width)
  kernel      = np.exp(-0.5*np.square(kernel_domain))*np.sin(kernel_domain)

  # Filtering
  filtered    = np.abs(np.convolve(input_pad,kernel,'valid'))

  if len(filtered) > len(input):
    filtered = filtered[:-1]

  return np.power(filtered,2.0)


def locationLinspace(l1, l2, n):
  xs = np.linspace(l1.x, l2.x, n)
  ys = np.linspace(l1.y, l2.y, n)

  return [Location(xlon=x, ylat=y) for x, y in zip(xs, ys)]

def locationArange(l1, l2, step_size):
  if l1 == l2:
    return [l1]
  elif step_size == 0:
    raise ValueError("Step Size should not be 0")
  else:
    unit_vec = (l2-l1).getUnit()*step_size

    if unit_vec.d_xlon != 0:
      xs = np.arange(l1.x, l2.x, unit_vec.d_xlon)
    if unit_vec.d_ylat != 0:
      ys = np.arange(l1.y, l2.y, unit_vec.d_ylat)

    if unit_vec.d_xlon == 0:
      xs = np.ones(ys.shape)*l1.x

    if unit_vec.d_ylat == 0:
      ys = np.ones(xs.shape)*l1.y

    return [Location(xlon=x, ylat=y) for x, y in zip(xs, ys)]

def positivizePath(plan):
  for robot_plan in plan:
    for step in robot_plan:
      if step[1] < 0:
        step[1] *= -1
        step[0] += math.pi

def polarVectorAdd(v1, v2):
  # v1, v2 are [r, theta] tuples
  r1 = v1[1]
  r2 = v2[1]
  theta_1 = v1[0]
  theta_2 = v2[0]

  r_res = math.sqrt(r1**2 + r2**2 + 2*r1*r2*math.cos(theta_2 - theta_1))
  theta_res = theta_1 + math.atan2(r2*math.sin(theta_2 - theta_1), r1 + r2*math.cos(theta_2 - theta_1))

  return [theta_res, r_res]


def getXYfromLatLon(latlon, origin):
  dx = haversine.haversine((latlon.lon, latlon.lat), (origin.lon, latlon.lat))
  dy = haversine.haversine((latlon.lon, latlon.lat), (latlon.lon, origin.lat))
  return Location(xlon=dx, ylat=dy)


def segmentInCircle(v1, v2, center, r):
  # Checks if a line segment defined by 'v1' and 'v2' ever goes within a circle centered at 'center' with radius 'r'
  # Code adapted from bobobobo's answer on stackoverflow: https://stackoverflow.com/a/1084899
  if euclideanDist(v1, center) <= r or euclideanDist(v2, center) <= r:
    return True
  else:
    d = v1 - v2
    f = center - v1

    a = d.dotProduct(d)
    b = 2 * (f.dotProduct(d))
    c = (f.dotProduct(f)) - (r**2)
    discriminant = b*b-4*a*c

    if discriminant >= 0:
      discriminant = math.sqrt(discriminant)
      t1 = (-b - discriminant) / (2 * a)
      t2 = (-b + discriminant) / (2 * a)

      if t1 >= 0 and t1 <= 1:
        return True
      elif t2 >= 0 and t2 <= 1:
        return True
  return False

def segmentInRectangle(v1, v2, c1, c2, c3, c4):
  # Checks if a line segment defined by v1 and v2 ever enters a rectangle defined by the 4 corners c1-c4.
  #  (note that these corners are assumed to be in order either clockwise or counterclockwise)
  #
  #   c1 ------- c2     c1 ------- c4
  #   |          |      |          |
  #   |          |  or  |          |
  #   |          |      |          |
  #   c4 ------- c3     c2 ------- c3


  # If either v1 or v2 lie within the rectangle, some of the segment is within the rectangle

  vertices = np.array([[pt.x, pt.y] for pt in [c1, c2, c3, c4]])

  hull = vertices[ConvexHull(vertices).vertices]
  fence = pth.Path(np.vstack([hull, hull[0]]))

  if fence.contains_point([v1.x, v1.y]) or fence.contains_point([v2.x, v2.y]):
    return True

  # If the segment intersects any of the edge segments, some of the segment is within the rectangle.
  for l1, l2 in zip([c1, c2, c3, c4], [c4, c1, c2, c3]):
    if segmentIntersect(v1, v2, l1, l2):
      return True

  return False


def segmentIntersect(l1, l2, v1, v2):
  # Check if two line segments defined by Location Objects {l1, l2} and {v1, v2} intersect.
  # Code Adapted from Victor Liu's answer on stackoverflow: https://stackoverflow.com/a/3842157

  d0 = np.linalg.det(np.array([ [l1.x - v1.x, l2.x - v1.x], [l1.y - v1.y, l2.y - v1.y] ]))
  d1 = np.linalg.det(np.array([ [l1.x - v2.x, l2.x - v2.x], [l1.y - v2.y, l2.y - v2.y] ]))
  d2 = np.linalg.det(np.array([ [v1.x - l1.x, v2.x - l1.x], [v1.y - l1.y, v2.y - l1.y] ]))
  d3 = np.linalg.det(np.array([ [v1.x - l2.x, v2.x - l2.x], [v1.y - l2.y, v2.y - l2.y] ]))

  if (d0 == 0) and (d1 == 0) and (d2 == 0) and (d3 == 0):
    return min(max(l1.x, l2.x), max(v1.x, v2.x)) >= max(min(v1.x, v2.x), min(l1.x, l2.x))
  elif (d0 * d1 <= 0) and (d2 * d3 <= 0):
    return True
  else:
    return False


def distanceToSegment(segment_locs, query_loc):
  assert len(segment_locs) >= 2, "Needed at least 2 points to form segment"

  segments = LineString([list(x) for x in segment_locs])
  pt = query_loc.shapelyPoint()
  return segments.distance(pt)

  # Adapted from: https://gist.github.com/nim65s/5e9902cd67f094ce65b0
  # Computes Distance from query_pt to the segment defined by segment_locs

  # segment_distances = []
  # for l1, l2 in zip(segment_locs[:-1], segment_locs[1:]):
  #   A = l1.npArray()
  #   B = l2.npArray()
  #   P = query_loc.npArray()
  #   if all(A == P) or all(B == P):
  #       return 0
  #   if np.arccos(np.dot((P - A) / np.linalg.norm(P - A), (B - A) / np.linalg.norm(B - A))) > math.pi / 2:
  #       return np.linalg.norm(P - A)
  #   if np.arccos(np.dot((P - B) / np.linalg.norm(P - B), (A - B) / np.linalg.norm(A - B))) > math.pi / 2:
  #       return np.linalg.norm(P - B)
  #   segment_distances.append(np.linalg.norm(np.cross(A-B, A-P))/np.linalg.norm(B-A))
  # return np.min(segment_distances)



def distanceToPoly(poly_locs, query_loc, units='km'):
  assert len(poly_locs) >= 3, "Needed at least 3 points to form Polygon"

  poly = Polygon([list(x) for x in poly_locs])
  pt = query_loc.shapelyPoint()

  dist_magnitude = poly.exterior.distance(pt)
  dist_sign_indicator = poly.distance(pt)

  if units=='km':
    if dist_magnitude == 0:
      return 0
    elif dist_sign_indicator == 0:
      return -dist_magnitude
    else:
      return dist_magnitude
  elif units=='latlon':
    scaling_factor = max((query_loc - getLatLon(query_loc, 1.0, 'north')).getMagnitude(), (getLatLon(query_loc, 1.0, 'west') - query_loc).getMagnitude())
    if dist_magnitude == 0:
      return 0
    elif dist_sign_indicator == 0:
      return -dist_magnitude / scaling_factor
    else:
      return dist_magnitude / scaling_factor





def calcLocation(reference, dx, dy):
  # Calculates the lat/lon coordinate of a point that is dx, dy km away from the reference lat/lon point
  bearing = math.atan2(dy, dx)
  distance = math.sqrt(dx^2 + dy^2)

  return getLatLon(reference, distance, bearing)


def timeOverlap(start1, duration1, start2, duration2):
  for t in [start1, start1+duration1]:
    if start2 <= t <= start2+duration2:
      return True

  for t in [start2, start2+duration2]:
    if start1 <= t <= start1+duration1:
      return True

  return False



def robotLocFromPlan(current_time, robot):
  # pdb.set_trace()
  if current_time >= robot.future_plan[-1].time:
    # If the robot is expected to have finished the plan, return the last waypoint
    print( "Robot %s is expected to have finished plan" % robot.name )
    return robot.future_plan[-1].loc, None
  else:
    # Otherwise interpolate along the path to find the robot's predicted location
    try:
      last_wpt, last_wpt_idx = next([x, idx] for idx, x in enumerate(robot.future_plan[::-1]) if x.time <= current_time)
      last_wpt_idx = len(robot.future_plan) - last_wpt_idx - 1

      current_wpt_idx = last_wpt_idx + 1
      current_wpt = robot.future_plan[current_wpt_idx]
      # unit_vec = (current_wpt.loc - last_wpt.loc).getUnit()
      # return last_wpt.loc + unit_vec * robot.vel * (current_time - last_wpt.time), current_wpt_idx
      vec = (current_wpt.loc - last_wpt.loc)
      return last_wpt.loc + vec * (current_time - last_wpt.time) / (current_wpt.time - last_wpt.time), current_wpt_idx
    except:
      print( "Dylan's addition" )
      return robot.future_plan[-1].loc, len(robot.future_plan)-1



# def robotLocFromPlan(current_time, robot):
#   # pdb.set_trace()
#   if current_time >= robot.future_plan[-1].time:
#     # If the robot is expected to have finished the plan, return the last waypoint
#     print "Robot %s is expected to have finished plan" % robot.name
#     return robot.future_plan[-1].loc, None
#   elif current_time >= robot.future_plan[0].time:
#     # Otherwise interpolate along the path to find the robot's predicted location
#     try:
#       last_wpt, last_wpt_idx = next([x, idx] for idx, x in enumerate(robot.future_plan[::-1]) if x.time <= current_time)
#       last_wpt_idx = len(robot.future_plan) - last_wpt_idx - 1

#       current_wpt_idx = last_wpt_idx + 1
#       current_wpt = robot.future_plan[current_wpt_idx]
#       # unit_vec = (current_wpt.loc - last_wpt.loc).getUnit()
#       # return last_wpt.loc + unit_vec * robot.vel * (current_time - last_wpt.time), current_wpt_idx

#       assert current_time >= last_wpt.time
#       assert current_time <= current_wpt.time
#       vec = (current_wpt.loc - last_wpt.loc)
#       return last_wpt.loc + vec * (current_time - last_wpt.time) / (current_wpt.time - last_wpt.time), current_wpt_idx
#     except:
#       print "Dylan's addition"
#       return robot.future_plan[-1].loc, len(robot.future_plan)-1

#   else:
#     print "[Warning] Robot %s is asked to predict location before start of plan." % robot.name
#     return robot.future_plan[0].loc, 0



def fancyStride(l, n):
  # Intended to provide a list of n approximately uniformly spaced items from list l  if n > len(l), return l
  stride_len = len(l)/float(n)
  if stride_len <= 1:
    return l
  else:
    idx_list = [min(round(stride_len*idx), len(l)-1) for idx in range(n)]
    return [l[int(idx)] for idx in idx_list]


def loadObservations(vehicle_data, planning_robots, min_depth, obs_type, obs_past_time=-float('inf'), obs_future_time=float('inf')):
  vehicle_obs = []

  for vehicle_type in vehicle_data:

    vehicle_type_data = vehicle_data[vehicle_type]
    for vehicle_name in vehicle_type_data:
      if vehicle_name in planning_robots:
        data = vehicle_type_data[vehicle_name]
        try:
          combined_data = zip(data['x_km'], data['y_km'], data['depth'], data[obs_type], data['datetime'])
          # print "[Info] Loaded %s data from vehicle %s-%s" % (obs_type, vehicle_type, vehicle_name)
        except KeyError:
          # print "[Warning] Missing '%s' data for vehicle '%s %s', skipping..." % (obs_type, vehicle_type, vehicle_name)
          continue

        new_vehicle_obs = [Observation(Location(xlon=x, ylat=y), data=data, time=obs_time) for x, y, depth, data, obs_time in combined_data if (depth < min_depth) and (obs_time >= obs_past_time) and (obs_time <= obs_future_time)]

        new_vehicle_obs = fancyStride(new_vehicle_obs, 1000)

        vehicle_obs += new_vehicle_obs

  return vehicle_obs


def gradient2d(field, ksize=5):
  sobelx = cv.Sobel(field, cv.CV_64F, 1, 0, ksize=ksize)
  sobely = cv.Sobel(field, cv.CV_64F, 0, 1, ksize=ksize)

  return np.sqrt(np.power(sobelx, 2) + np.power(sobely, 2))



def mkdirP(path):
  try:
    os.makedirs(path)
  except OSError as exc:  # Python >2.5
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise


def dateRange(d1, d2, step):
  curr = d1
  while curr < d2:
    yield curr
    curr += step



def samplePoints(matrix, n_pts):
  height, width = matrix.shape[:2]
  valid_pts = [pt for pt in itertools.product(range(width), range(height)) if matrix[pt[1], pt[0]] != 0]
  return random.sample(valid_pts, n_pts)


def hSigEqual(h1, h2):
  if len(h1) == len(h2):
    if all([x == y for (x,y) in zip(h1, h2)]):
      return True
  return False


def rayIntersection(line_segment, ray_origin):
  ray_direction = np.array([0, 1])
  ray_origin = ray_origin.npArray()
  point1 = line_segment[0].npArray()
  point2 = line_segment[1].npArray()


  v1 = ray_origin - point1
  v2 = point2 - point1
  v3 = np.array([-ray_direction[1], ray_direction[0]])

  if np.dot(v2, v3) == 0:
    return False

  t1 = np.cross(v2, v1) / np.dot(v2, v3)
  t2 = np.dot(v1, v3) / np.dot(v2, v3)

  return t1 >= 0.0 and t2 >= 0.0 and t2 <= 1.0


def findPtInContour(contour, height, width, res):
  while True:
    pt = [random.randint(0, width), random.randint(0, height)]
    if cv2.pointPolygonTest(contour, tuple(pt), False) != -1:
      for sample in res:
        if pt[0] == sample[0] or pt[1] == sample[1]:
          break
      else:
        pt[0] = pt[0] + .5
        return tuple(pt)

def state2Dis(state, tree):

  if state is not None:
    nn = tree.query(state.asTupleNoGoal())
    return tuple(tree.data[nn[1]])
  else:
    return None

def getLastNDim(data, ndim):
  if data.ndim <= ndim:
    return data
  slc = [0] * (data.ndim - ndim)
  slc += [slice(None)] * ndim
  return data[tuple(slc)]


def getFirstNDim(data, ndim):
  if data.ndim <= ndim:
    return data
  slc = [slice(None)] * ndim
  slc += [0] * (data.ndim - ndim)
  return data[tuple(slc)]

if __name__ == '__main__':
  foo = np.random.random((11, 22, 33, 44, 55))

  for ii in range(1,9):
    print (getFirstNDim(foo, ii).shape)
    print (getLastNDim(foo, ii).shape)

  pdb.set_trace()
