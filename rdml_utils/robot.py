import numpy as np
import os, pdb, math, time
from location import Location, LocDelta, StampedLocation
from utils import robotLocFromPlan, euclideanDist

class Robot(object):


  def __init__(self, robot_parameters, name, robot_location):
    # Parameters
    self.name = name
    self.location = robot_location
    self.robot_type = robot_parameters['type']
    self.vel = robot_parameters['vel']
    self.color = robot_parameters['color']
    self.icon  = robot_parameters['icon']
    self.sample_period = robot_parameters['sample_period']
    self.waypoint_tolerance = robot_parameters['waypoint_tolerance']
    self.time_last_sample = -float('inf')
    self.time_last_plan = -float('inf')
    self.replan_period = robot_parameters['replan_period']



    # List of location objects
    self.future_plan    = []

    # List of location objects
    self.past_path      = []

    # List of observation objects - science data (temp, salt, etc)
    self.science_observations   = []

    self.current_u_observations = []
    self.current_v_observations = []



  def step(self, sim_period, world, step_time, ignore_currents=False):
    if len(self.future_plan) == 0:
      self_direction = LocDelta(0, 0)
      self_distance = 0.

    else:
      while euclideanDist(self.future_plan[0], self.location) < self.vel * sim_period: #self.waypoint_tolerance:  # (self.future_plan[0]-self.location).getMagnitude() 
        self.incrementWaypoint()
        if len(self.future_plan) == 0:
          if isinstance(self.past_path[-1], StampedLocation):
            vec = self.past_path[-1].loc-self.location
          else:
            vec = self.past_path[-1]-self.location

          self_direction  = vec.getUnit()
          self_distance = vec.getMagnitude()
          break
      else:
        if isinstance(self.future_plan[0], StampedLocation):
          self_direction  = (self.future_plan[0].loc-self.location).getUnit()
        else:
          self_direction  = (self.future_plan[0]-self.location).getUnit()
          
        # self_direction  = (self.future_plan[0]-self.location).getUnit()
        self_distance = self.vel*sim_period

    if ignore_currents:
      ocean_current_disturbance = LocDelta(0., 0.)
    else:
      ocean_current_disturbance = world.getUVcurrent(self.location, step_time) * sim_period / 1000.

    self.location += self_direction*self_distance + ocean_current_disturbance


  def incrementWaypoint(self):
    next_wpt = self.future_plan.pop(0)
    self.past_path.append(next_wpt)


  def timeToSample(self, time):
    if self.sample_period < 0:
      return False
    else:
      if (time-self.time_last_sample) >= self.sample_period:
        # self.time_last_sample = time
        return True
      else:
        return False

  def __str__(self):
    # str_plan = ['<%.3f,%.3f>' % (p.lon,p.lat) for p in self.future_plan]
    # return '%s: %s' % (self.name,' '.join(str_plan))
    return '%s: %s' % (self.name,self.location)

def loadRobots(robots_list, robots_data, robots_cfg, robots_future_plans=None, ct=None, load_time=time.time()):
  res_robots = []
  for bot_name in robots_list:
    bot_type = robots_cfg[bot_name]['type']
    bot_data = robots_data[bot_type][bot_name]
    latlon_past_path = [Location(xlon=bot_lon, ylat=bot_lat) for bot_lon, bot_lat in zip(bot_data['longitude'], bot_data['latitude'])]
    new_bot = Robot(robots_cfg[bot_name], bot_name, ct.latlon2xy(latlon_past_path[-1]))
    if (robots_future_plans is not None) and (bot_name in robots_future_plans):
      new_bot.future_plan = [StampedLocation(ct.latlon2xy(Location(xlon=coord['lon'], ylat=coord['lat'])), coord['time']) for coord in robots_future_plans[bot_name]['future']]
      new_bot.time_last_plan = robots_future_plans[bot_name]['planning_time']
    else:
      print "[Warning] Failed to load plan for %s" % bot_name

    if load_time - bot_data['datetime'][-1] >= 600:
      plan_start_loc_idx = np.argmin([abs(x - new_bot.future_plan[0].time) for x in bot_data['datetime']]) 
      new_bot.future_plan[0] = StampedLocation(ct.latlon2xy(Location(xlon=bot_data['longitude'][plan_start_loc_idx], 
                                                                     ylat=bot_data['latitude'][plan_start_loc_idx])),
                                               new_bot.future_plan[0].time)

      new_bot.location = robotLocFromPlan(load_time, new_bot)[0]
      print "[Warning] Position estimate for %s-%s over 10 min old, interpolating along path instead" % (bot_type, bot_name)

    res_robots.append(new_bot)

  print "Loaded %d robots" % len(res_robots)
  return res_robots



def fixedLoadRobots(robots_list, robots_data, robots_cfg, robots_future_plans=None, ct=None, load_time=time.time()):
  res_robots = []
  for bot_name in robots_list:
    bot_type = robots_cfg[bot_name]['type']
    bot_data = robots_data[bot_type][bot_name]
    latlon_past_path = [Location(xlon=bot_lon, ylat=bot_lat) for bot_lon, bot_lat in zip(bot_data['longitude'], bot_data['latitude'])]
    new_bot = Robot(robots_cfg[bot_name], bot_name, ct.latlon2xy(latlon_past_path[-1]))
    if (robots_future_plans is not None) and (bot_name in robots_future_plans):
      new_bot.future_plan = [StampedLocation(ct.latlon2xy(Location(xlon=coord['lon'], ylat=coord['lat'])), coord['time']) for coord in robots_future_plans[bot_name]['future']]
      new_bot.time_last_plan = robots_future_plans[bot_name]['planning_time']
    else:
      print "[Warning] Failed to load plan for %s" % bot_name

    if load_time - bot_data['datetime'][-1] >= 600:
      plan_start_loc_idx = np.argmin([abs(x - new_bot.future_plan[0].time) for x in bot_data['datetime']]) 
      estimated_start_loc = StampedLocation(ct.latlon2xy(Location(xlon=bot_data['longitude'][plan_start_loc_idx], 
                                                                  ylat=bot_data['latitude'][plan_start_loc_idx])), 
                                            bot_data['datetime'][plan_start_loc_idx])

      new_bot.future_plan = [estimated_start_loc] + new_bot.future_plan[2:]

      for prev_pt_idx, pt in enumerate(new_bot.future_plan[1:]):
        pt.time = new_bot.future_plan[prev_pt_idx].time + (pt.loc - new_bot.future_plan[prev_pt_idx].loc).getMagnitude() / new_bot.vel

      new_bot.location = robotLocFromPlan(load_time, new_bot)[0]

      print "[Warning] Position estimate for %s-%s over 10 min old, interpolating along path instead" % (bot_type, bot_name)

    res_robots.append(new_bot)

  print "Loaded %d robots" % len(res_robots)
  return res_robots