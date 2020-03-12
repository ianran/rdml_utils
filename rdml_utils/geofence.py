import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.path as pth
from .location import Location, StampedLocation

import pdb

class Geofence(object):
  """docstring for Geofence"""
  def __init__(self, pts):
    self.pts = pts
    vertices = np.array([[pt.lon, pt.lat] for pt in self.pts])

    hull = vertices[ConvexHull(vertices).vertices]
    self.fence = pth.Path(np.vstack([hull, hull[0]]))


  def __str__(self):
    return "Geofence\n\t" + "\n\t".join(str(pt) for pt in self.pts)

  @classmethod
  def fromPts(cls, pts):
    return cls(pts)

  @classmethod
  def fromBounds(cls, bounds):
    if isinstance(bounds, dict):
      north_bound = bounds['north']
      south_bound = bounds['south']
      east_bound = bounds['east']
      west_bound = bounds['west']
    elif isinstance(bounds, list):
      north_bound = bounds[0]
      south_bound = bounds[1]
      east_bound = bounds[2]
      west_bound = bounds[3]
    else:
      print( "[GEOFENCE] Error: Could not parse vertices" )

    vertices = [Location(ylat=north_bound, xlon=east_bound), Location(ylat=north_bound, xlon=west_bound), Location(ylat=south_bound, xlon=west_bound), Location(ylat=south_bound, xlon=east_bound)]
    return cls(vertices)


  def isValidPoint(self, pt):
    if isinstance(pt, Location):
      lat = pt.lat
      lon = pt.lon
    elif isinstance(pt, StampedLocation):
      lon = pt.loc.lon
      lat = pt.loc.lat
    elif isinstance(pt, dict):
      lat = pt['lat']
      lon = pt['lon']
    elif isinstance(pt, list):
      lat = pt[1]
      lon = pt[0]
    else:
      print( "[GEOFENCE] Error: Could not parse point" )

    if self.fence is None:
      return True
    else:
      return self.fence.contains_point([lon, lat])

  def draw(self, ax, color):
    ax.plot([pt.x for pt in self.pts + [self.pts[0]]], [pt.y for pt in self.pts + [self.pts[0]]], c=color)
