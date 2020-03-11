
import pdb, numbers, math, haversine, time

import numpy as np
import matplotlib.pyplot as plt
from sas_utils import Location, LocDelta
import oyaml as yaml

class CoordTransformer(object):
  # Wrapper Class for handling the translation, rotation, and scaling to move
  #   a coordinate from the global (Latitude / Longitude) frame to the planning
  #   (X, Y kilometer) frame.
  def __init__(self, region_heading, region_center, region_velocity):
    # Region Heading = Heading in Degrees from the Vertical, Positive angle is CCW
    # Region Center = Center of Target Region in Decimal Degrees Latitude and Longitude
    # Region Velocity = Velocity of region in global frame in km / s

    self.rotate_theta = math.radians(region_heading)
    self.translate_x, self.translate_y = (Location(0, 0) - region_center).npArray()  # DX, DY in terms of Degrees Lon / Lat, respectively
    self.vel = region_velocity
    # Assuming the world is flat nearby the region center
    scale_x = haversine.haversine((region_center.lat, region_center.lon), (region_center.lat, region_center.lon + 1.))  # KM per Degree Longitude
    scale_y = haversine.haversine((region_center.lat, region_center.lon), (region_center.lat + 1., region_center.lon))  # KM per Degree Latitude


    self.rotate_transform = np.array([[math.cos(self.rotate_theta), -math.sin(self.rotate_theta), 0],
                                      [math.sin(self.rotate_theta),  math.cos(self.rotate_theta), 0],
                                      [0,                       0,                      1]])

    self.translate_transform = np.array([[1, 0, self.translate_x],
                                         [0, 1, self.translate_y],
                                         [0, 0, 1]])

    self.scaling_transform = np.array([[scale_x, 0,       0],
                                       [0,       scale_y, 0],
                                       [0,       0,       1]])

    self.reverse_rotate_transform = np.array([[math.cos(-self.rotate_theta), -math.sin(-self.rotate_theta), 0],
                                              [math.sin(-self.rotate_theta),  math.cos(-self.rotate_theta), 0],
                                              [0,                        0,                       1]])

    self.reverse_translate_transform = np.array([[1, 0, -self.translate_x],
                                                 [0, 1, -self.translate_y],
                                                 [0, 0,  1]])

    self.reverse_scaling_transform = np.array([[1./scale_x, 0,       0],
                                               [0,       1./scale_y, 0],
                                               [0,       0,          1]])

  def __str__(self):
    return "Location: %.3f, %.3f\nHeading: %.3f" % (self.translate_x, self.translate_y, math.degrees(self.rotate_theta))

  @classmethod
  def fromFile(cls, filename, transform_time=time.time()):
    with open(filename,'rb') as f:
      transforms = yaml.load(f.read())

    if isinstance(transforms, list):
      try:
        transform = next(x for x in transforms[::-1] if x['datetime'] < transform_time)
      except StopIteration:
        transform = transforms[-1] # Default to the last transformation
    elif isinstance(transforms, dict):
      transform = transforms

    return cls.fromDict(transform)

  @classmethod
  def fromDict(cls, trans_dict):
    region_center = Location(xlon=trans_dict['center']['longitude'], ylat=trans_dict['center']['latitude'])
    region_heading = trans_dict['heading']
    if 'velocity' in trans_dict:
      region_velocity = LocDelta(d_ylat=trans_dict['velocity']['latitudinal'], d_xlon=trans_dict['velocity']['longitudinal'])
    else:
      region_velocity = LocDelta(d_ylat=0.0, d_xlon=0.0)

    return cls(region_heading, region_center, region_velocity)

  def updateHeading(self, new_heading):
    self.rotate_theta = math.radians(region_heading)

    self.rotate_transform = np.array([[math.cos(self.rotate_theta), -math.sin(self.rotate_theta), 0],
                                      [math.sin(self.rotate_theta),  math.cos(self.rotate_theta), 0],
                                      [0,                       0,                      1]])

    self.reverse_rotate_transform = np.array([[math.cos(-self.rotate_theta), -math.sin(-self.rotate_theta), 0],
                                              [math.sin(-self.rotate_theta),  math.cos(-self.rotate_theta), 0],
                                              [0,                        0,                       1]])


  def updateCenter(self, new_center):
    self.translate_x, self.translate_y = (new_center - Location(0, 0)).npArray()  # DX, DY in terms of Degrees Lon / Lat, respectively

    # Assuming the world is flat nearby the region center
    scale_x = haversine.haversine((new_center.lat, new_center.lon), (new_center.lat, new_center.lon + 1.))  # KM per Degree Longitude
    scale_y = haversine.haversine((new_center.lat, new_center.lon), (new_center.lat + 1., new_center.lon))  # KM per Degree Latitude

    self.translate_transform = np.array([[1, 0, self.translate_x],
                                         [0, 1, self.translate_y],
                                         [0, 0, 1]])

    self.scaling_transform = np.array([[scale_x, 0,       0],
                                       [0,       scale_y, 1],
                                       [0,       0,       1]])

    self.reverse_translate_transform = np.array([[1, 0, -self.translate_x],
                                                 [0, 1, -self.translate_y],
                                                 [0, 0,  1]])

    self.reverse_scaling_transform = np.array([[1./scale_x, 0,       0],
                                               [0,       1./scale_y, 0],
                                               [0,       0,          1]])

  def latlon2xy(self, coord):
    np_coord = (np.hstack((coord.npArray(), 1))[np.newaxis]).transpose()
    translated_coord = np.matmul(self.translate_transform, np_coord)
    rotated_coord = np.matmul(self.rotate_transform, translated_coord)
    final_coord = np.matmul(self.scaling_transform, rotated_coord)

    return Location(xlon=float(final_coord[0]), ylat=float(final_coord[1]))

  def xy2latlon(self, coord):
    np_coord = (np.hstack((coord.npArray(), 1))[np.newaxis]).transpose()
    scaled_coord = np.matmul(self.reverse_scaling_transform, np_coord)
    rotated_coord = np.matmul(self.reverse_rotate_transform, scaled_coord)
    final_coord = np.matmul(self.reverse_translate_transform, rotated_coord)

    return Location(xlon=float(final_coord[0]), ylat=float(final_coord[1]))

  def headinglocal2global(self, local_heading, units='radians'):
    if units == 'radians':
      return local_heading + self.rotate_theta
    elif units == 'degrees':
      return local_heading + math.degrees(self.rotate_theta)

  def headingglobal2local(self, global_heading, units='radians'):
    if units == 'radians':
      return global_heading - self.rotate_theta
    elif units == 'degrees':
      return global_heading - math.degrees(self.rotate_theta)

  def uv2planningFrame(self, uv):
    adjusted_uv = uv - self.vel # When going to planning frame subtract frame velocity first
    np_uv = (np.hstack((adjusted_uv.npArray(), 1))[np.newaxis]).transpose()
    final_uv = np.matmul(self.rotate_transform, np_uv)

    return LocDelta(d_xlon=float(final_uv[0]), d_ylat=float(final_uv[1]))

  def uv2globalFrame(self, uv):
    np_uv = (np.hstack((uv.npArray(), 1))[np.newaxis]).transpose()
    rotated_uv = np.matmul(self.reverse_rotate_transform, np_uv)
    final_uv = rotated_uv + self.vel.npArray() # When going to global frame add frame velocity last
    return LocDelta(d_xlon=float(final_uv[0]), d_ylat=float(final_uv[1]))

if __name__ == '__main__':
  ref_coord = Location(xlon= -2.0, ylat = 2.0)
  frame_heading = 30.0
  box_size = 1.0
  latlon_coord = Location(xlon=1., ylat=1.)
  xy_coord = Location(xlon=50., ylat=50.)

  global_frame_current = LocDelta(d_xlon=.3, d_ylat=.3)
  planning_frame_current = LocDelta(d_xlon=30, d_ylat=30)

  global_frame_heading = math.atan2(global_frame_current.d_ylat, global_frame_current.d_xlon)
  planning_frame_heading = math.atan2(planning_frame_current.d_ylat, planning_frame_current.d_xlon)



  r = np.array([[math.cos(math.radians(frame_heading)), -math.sin(math.radians(frame_heading))],
                [math.sin(math.radians(frame_heading)),  math.cos(math.radians(frame_heading))]])

  c1 = np.matmul([1, 1], r) + np.array([ref_coord.x, ref_coord.y])
  c2 = np.matmul([1, -1], r) + np.array([ref_coord.x, ref_coord.y])
  c3 = np.matmul([-1, -1], r) + np.array([ref_coord.x, ref_coord.y])
  c4 = np.matmul([-1, 1], r) + np.array([ref_coord.x, ref_coord.y])

  ct = CoordTransformer(region_heading = frame_heading, region_center=ref_coord)

  transformed_latlon_coord = ct.latlon2xy(latlon_coord)
  transformed_xy_coord = ct.xy2latlon(xy_coord)

  transformed_global_frame_current = ct.uv2planningFrame(global_frame_current) * 100
  transformed_planning_frame_current = ct.uv2globalFrame(planning_frame_current) / 100


  try:
    assert ct.latlon2xy(ct.xy2latlon(xy_coord)) == xy_coord
  except AssertionError:
    pdb.set_trace()

  try:
    assert ct.xy2latlon(ct.latlon2xy(latlon_coord)) == latlon_coord
  except AssertionError:
    pdb.set_trace()

  print( "global_frame_heading: %.03f" % global_frame_heading )
  print( "planning_frame_heading: %.03f" % planning_frame_heading )

  transformed_global_frame_heading = ct.headingglobal2local(global_frame_heading)
  transformed_planning_frame_heading = ct.headinglocal2global(planning_frame_heading)

  print( "transformed_global_frame_heading: %.03f" % transformed_global_frame_heading )
  print( "transformed_planning_frame_heading: %.03f" % transformed_planning_frame_heading )

  plt.figure()
  plt.scatter(ref_coord.x, ref_coord.y, color='k')
  plt.scatter(latlon_coord.x, latlon_coord.y, color='g')
  plt.scatter(transformed_xy_coord.x, transformed_xy_coord.y, color='r')
  plt.plot([coord.x for coord in [latlon_coord, latlon_coord+global_frame_current]], [coord.y for coord in [latlon_coord, latlon_coord+global_frame_current]], c='g')
  plt.plot([coord.x for coord in [transformed_xy_coord, transformed_xy_coord+transformed_planning_frame_current]], [coord.y for coord in [transformed_xy_coord, transformed_xy_coord+transformed_planning_frame_current]], c='r')
  # plt.scatter([5, -5, -5, 5], [5, 5, -5, -5], c='b')
  plt.plot([-5, 5], [0, 0], color='k', linestyle=':', linewidth=1)
  plt.plot([0, 0], [-5, 5], color='k', linestyle=':', linewidth=1)
  plt.plot([ref_coord.x, ref_coord.x+math.cos(math.radians(90-frame_heading))*box_size*1.5], [ref_coord.y, ref_coord.y+math.sin(math.radians(90-frame_heading))*box_size*1.5], color='k')
  plt.plot([x[0] for x in [c1, c2, c3, c4, c1]], [x[1] for x in [c1, c2, c3, c4, c1]])
  plt.xlabel("Longitude (Degrees)")
  plt.ylabel("Latitude (Degrees)")
  plt.axis('equal')


  box_x_size = 111.17799069
  box_y_size = 111.19492664

  plt.figure()
  plt.scatter(0, 0, color='k')
  plt.scatter(xy_coord.x, xy_coord.y, color='r')
  plt.scatter(transformed_latlon_coord.x, transformed_latlon_coord.y, color='g')
  plt.plot([coord.x for coord in [xy_coord, xy_coord+planning_frame_current]], [coord.y for coord in [xy_coord, xy_coord+planning_frame_current]], c='r')
  plt.plot([coord.x for coord in [transformed_latlon_coord, transformed_latlon_coord+transformed_global_frame_current]], [coord.y for coord in [transformed_latlon_coord, transformed_latlon_coord+transformed_global_frame_current]], c='g')
  plt.plot([0, 0], [0, box_y_size*1.5], color='k')
  plt.plot([box_x_size, -box_x_size, -box_x_size, box_x_size, box_x_size], [box_y_size, box_y_size, -box_y_size, -box_y_size, box_y_size], color='k')
  plt.xlabel("X (km)")
  plt.ylabel("Y (km)")
  plt.axis('equal')


  plt.show(False)

  pdb.set_trace()
