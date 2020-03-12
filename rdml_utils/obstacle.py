
from .location import Location, LocDelta
from .utils import euclideanDist, distanceToPoly, distanceToSegment

from scipy.spatial import ConvexHull
import matplotlib.path as pth
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import cascaded_union
import pdb, math, datetime

class Obstacle(object):
  """docstring for Obstacle"""
  def __init__(self, obs_id, points, exclusion_radius, obs_type = "polygon"):
    if isinstance(points, Location):
      self.points = [points]
    else:
      self.points = points
    self.obs_id = obs_id                      # Name of Obstacle
    self.exclusion_radius = exclusion_radius  # KM

    if len(self.points) == 1:
      self.obs_type = "point"
      # self.buffer = Point(self.points[0].x, self.points[0].y).buffer(self.exclusion_radius)
    elif len(self.points) == 2:
      self.obs_type = "line"
      # self.buffer = LineString([list(pt) for pt in self.points]).buffer(self.exclusion_radius)
    else:
      self.obs_type = obs_type
      # self.buffer = Polygon([list(pt) for pt in self.points]).buffer(self.exclusion_radius)


  def step(self, sim_period, world, step_time):
    pass

  @classmethod
  def fromDict(cls, obs_dict, obstacle_buffer=None):
    obs_locs = [Location(ylat=loc['latitude'], xlon=loc['longitude']) for loc in obs_dict["locations"]]
    if obstacle_buffer is None:
      obstacle_buffer = obs_dict["exclusion_radius"]
    return cls(obs_dict['obs_id'], obs_locs, obstacle_buffer, obs_dict['obstacle_type'])

  def __str__(self):
    return "Static Obstacle: %s\n\tType: %s\n\tLocation(s): %s\n\tRadius: %f" % (str(self.obs_id), self.obs_type, "".join("%s, " % loc for loc in self.points), self.exclusion_radius)

  def __eq__(self, other):
    if isinstance(other, Obstacle):
      return (self.points == other.points) and (self.obs_id == other.obs_id)
    else:
      return False

  def __ne__(self, other):
    return not self.__eq__(other)

  def pointInCollision(self, query_pt):
    if len(self.points) >= 3:
      if self.obs_type == "polygon":
        dist = distanceToPoly(self.points, query_pt)
      elif self.obs_type == "line":
        dist = distanceToSegment(self.points, query_pt)
    elif len(self.points) == 2:
      dist = distanceToSegment(self.points, query_pt)
    elif len(self.points) == 1:
      dist = euclideanDist(self.points[0], query_pt)
    else:
      dist = 0
      # We should never be here
      pdb.set_trace()

    return dist <= self.exclusion_radius

  def toDict(self):
    res = {}
    res['obs_id'] = self.obs_id
    res['locations'] = [{'longitude': float(loc.lon),
                         'latitude': float(loc.lat)
                        }
                        for loc in self.points]
    res['exclusion_radius'] = self.exclusion_radius
    res['obstacle_type'] = self.obs_type
    return res

  def getBuffer(self):
    if self.obs_type == "point":
      obs_geom = Point(self.points[0].x, self.points[0].y)
    elif self.obs_type == "line":
      obs_geom = LineString([list(pt) for pt in self.points])
    elif self.obs_type == "polygon":
      obs_geom = Polygon([list(pt) for pt in self.points])

    return obs_geom.buffer(self.exclusion_radius)



class DynamicObstacle(Obstacle):
  def __init__(self, obs_id, points, timestamp, exclusion_radius, heading, speed, lookahead_time, obs_type="polygon"):
    if isinstance(points, Location):
      Obstacle.__init__(self, obs_id, [points], exclusion_radius, obs_type)
    else:
      Obstacle.__init__(self, obs_id, points, exclusion_radius, obs_type)

    self.heading = heading                # (degrees) Using Nautical Heading (0 = North, 90 = East, 180 = South, 270 = West)
    self.speed = speed                    # (km/s) Vehicle's speed along its heading
    self.timestamp = timestamp            # Python Datetime Object (UTC)
    self.lookahead_time = lookahead_time  # (s) Number of Seconds forward we should watch out for the obstacle

  @classmethod
  def fromStatic(cls, static, speed, heading, timestamp, lookahead_time):
    return cls(static.obs_id, static.points, timestamp, static.exclusion_radius, heading, speed, lookahead_time, static.obs_type)

  def getBuffer(self):
    dynamic_vec = LocDelta.fromPolar(1.0, 90-math.degrees(self.heading), angle_units='deg') * self.speed * self.lookahead_time

    start_pts = self.points
    end_pts = [pt + dynamic_vec for pt in self.points]

    if self.obs_type == 'point':
      start_geom = Point(start_pts[0].x, start_pts[0].y).buffer(self.exclusion_radius)
      end_geom = Point(end_pts[0].x, end_pts[0].y).buffer(self.exclusion_radius)

    elif self.obs_type == 'line':
      start_geom = LineString([list(pt) for pt in start_pts]).buffer(self.exclusion_radius)
      end_geom = LineString([list(pt) for pt in end_pts]).buffer(self.exclusion_radius)

    elif self.obs_type == 'polygon':
      start_geom = Polygon([list(pt) for pt in start_pts]).buffer(self.exclusion_radius)
      end_geom = Polygon([list(pt) for pt in end_pts]).buffer(self.exclusion_radius)

    geometries = [start_geom, end_geom] + [LineString([list(p1), list(p2)]).buffer(self.exclusion_radius) for p1, p2 in zip(start_pts, end_pts)]

    return cascaded_union(geometries)


  @classmethod
  def fromDict(cls, obs_dict, lookahead_time=None, obstacle_buffer=None):
    obs_locs = [Location(ylat=loc['latitude'], xlon=loc['longitude']) for loc in obs_dict["locations"]]
    if obstacle_buffer is None:
      obstacle_buffer = obs_dict["exclusion_radius"]
    timestamp = datetime.datetime.strptime(obs_dict['timestamp'].split(".")[0], "%Y-%m-%d %H:%M:%S")
    if lookahead_time is None:
      lookahead_time = obs_dict['lookahead_time']

    return cls(obs_dict['obs_id'], obs_locs, timestamp, obstacle_buffer, obs_dict['heading'], obs_dict['speed'], lookahead_time, obs_dict['obstacle_type'])

  def pointInCollision(self, query_pt):
    print( "Query", query_pt )
    start_points = self.points

    projection = LocDelta.fromPolar(self.speed * self.lookahead_time, 90.-self.heading, angle_units='deg')

    projected_points = [pt + projection for pt in start_points]

    for pts_list in [start_points, projected_points]:
      for pt in pts_list:
        print( pt )
      if len(self.points) >= 3:
        if self.obs_type == "polygon":
          dist = distanceToPoly(pts_list, query_pt)
        elif self.obs_type == "line":
          dist = distanceToSegment(pts_list, query_pt)
      elif len(pts_list) == 2:
        dist = distanceToSegment(pts_list, query_pt)
      elif len(pts_list) == 1:
        dist = euclideanDist(pts_list[0], query_pt)
      else:
        dist = 0
        # We should never be here
        pdb.set_trace()
      print( dist )
      if dist <= self.exclusion_radius:
        return True

    for p1, p2 in zip(start_points, projected_points):
      if distanceToSegment([p1, p2], query_pt) <= self.exclusion_radius:
        return True

    return False


  def __str__(self):
    return "Obstacle: %s\n\tType: %s\n\tLocation: %s\n\tLast Updated %s UTC\n\tHeading: %f Speed: %f m/s\n\tRadius: %f" % (str(self.obs_id), self.obs_type, "".join("%s, " % loc for loc in self.points), str(self.timestamp), self.heading, self.speed, self.exclusion_radius)

  def step(self, sim_period, world, step_time, ignore_currents=False):
    self_direction = LocDelta(d_xlon=math.sin(self.heading), d_ylat=math.cos(self.heading))
    self_distance = self.speed*sim_period

    if ignore_currents:
      ocean_current_disturbance = LocDelta(0., 0.)
    else:
      ocean_current_disturbance = world.getUVcurrent(self.points[0], step_time) * sim_period / 1000.

    self.timestamp = step_time
    self.points = [pt + self_direction*self_distance + ocean_current_disturbance for pt in self.points]

  def toDict(self):
    res = {}
    res['obs_id'] = self.obs_id
    res['obstacle_type'] = self.obs_type
    res['locations'] = [{'longitude': float(loc.lon),
                         'latitude': float(loc.lat)
                        }
                        for loc in self.points]
    res['exclusion_radius'] = self.exclusion_radius
    res['timestamp'] = str(self.timestamp)
    res['heading'] = self.heading
    res['speed'] = self.speed
    res['lookahead_time'] = self.lookahead_time

    return res


def loadObstacles(static_obstacles_dict, dynamic_obstacles_dict, mission_cfg, ct=None):
  static_obstacles = []
  dynamic_obstacles = []
  bounding_box = [Location(ylat=mission_cfg['bounding_box'][ns], xlon=mission_cfg['bounding_box'][ew]) for ns, ew in zip(['north', 'north', 'south', 'south'], ['east', 'west', 'west', 'east'])]

  # Issue here is thast we are comparing km and deg - not good
  # Not sure what solution is as idk how we are converting between\
  if static_obstacles_dict is not None:
    for obstacle_name in static_obstacles_dict:
      obs_data = static_obstacles_dict[obstacle_name]
      obs = Obstacle.fromDict(obs_data)
      if any([distanceToPoly(bounding_box, pt, 'latlon') < 2*obs.exclusion_radius for pt in obs.points]):
        if ct is not None:
          obs.points = [ct.latlon2xy(pt) for pt in obs.points]
          if ct.vel != LocDelta(0,0):
            relative_obs_speed, relative_obs_theta = ct.vel.asPolar(degrees=True)
            # obs = DynamicObstacle.fromStatic(obs, relative_obs_speed, (90 - (relative_obs_theta + 180)) % 360, datetime.datetime.utcnow(), lookahead_time=mission_cfg['dynamic_obstacle_lookahead'])
            obs = DynamicObstacle.fromStatic(obs, relative_obs_speed, (relative_obs_theta + 180) % 360, datetime.datetime.utcnow(), lookahead_time=mission_cfg['dynamic_obstacle_lookahead'])


        static_obstacles.append(obs)


  if dynamic_obstacles_dict is not None:
    for obstacle_name in dynamic_obstacles_dict:
      obs_data = dynamic_obstacles_dict[obstacle_name]
      obs = DynamicObstacle.fromDict(obs_data, lookahead_time=mission_cfg['dynamic_obstacle_lookahead'])
      if any([distanceToPoly(bounding_box, pt, 'latlon') < obs.exclusion_radius for pt in obs.points]):
        if ct is not None:
          obs.points = [ct.latlon2xy(pt) for pt in obs.points]
          obs.heading = ct.headingglobal2local(obs.heading, units='degrees')
        dynamic_obstacles.append(obs)

  print( "Loaded %d static obstacles and %d dynamic obstacles" % (len(static_obstacles), len(dynamic_obstacles)) )
  return static_obstacles + dynamic_obstacles




if __name__ == '__main__':
  import os
  import oyaml as yaml
  import matplotlib.pyplot as plt

  with open(os.path.expandvars("$HOME/dev/sas_sim/data/obstacles_sim.yaml"),'rb') as f:
    obstacles_dict = yaml.load(f.read())

  with open(os.path.expandvars("$HOME/dev/sas_sim/cfg/mission.yaml"), 'rb') as f:
    yaml_mission = yaml.load(f.read())

  obs = loadObstacles(obstacles_dict, yaml_mission)


  for o in obs:
    x, y = o.getBuffer().exterior.xy
    plt.figure()
    plt.plot(x, y)

  plt.show()
