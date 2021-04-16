import sys, pdb
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.path as pth

if sys.version_info[0] < 3:
    # python 2
    from location import Location, StampedLocation
else:
    # python 3
    from rdml_utils.location import Location, StampedLocation

class Geofence(object):
  """docstring for Geofence"""
  def __init__(self, pts, fence_type="interior"):
    self.pts = pts


    if fence_type == "interior":
      self.interior_fence = True
      # Interior fence contains valid points

      # F F F F F F
      # F 0 - - 0 F
      # F | T T | F
      # F 0 - - 0 F
      # F F F F F F
      

    elif fence_type == "exterior":
      self.interior_fence = False
      # exterior fence excludes valid points

      # T T T T T T
      # T 0 - - 0 T
      # T | F F | T
      # T 0 - - 0 T
      # T T T T T T


    vertices = np.array([[pt.lon, pt.lat] for pt in self.pts])

    hull = vertices[ConvexHull(vertices).vertices]
    self.fence = pth.Path(np.vstack([hull, hull[0]]))


  def __str__(self):
    return "Geofence\n\t" + "\n\t".join(str(pt) for pt in self.pts)

  @classmethod
  def fromPts(cls, pts, fence_type="interior"):
    return cls(pts, fence_type)

  @classmethod
  def fromBounds(cls, bounds, fence_type="interior"):
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
    return cls(vertices, fence_type)


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

    elif self.interior_fence:
      return self.fence.contains_point([lon, lat])

    elif not self.interior_fence:
      return not self.fence.contains_point([lon, lat])



  def draw(self, ax, color):
    ax.plot([pt.x for pt in self.pts + [self.pts[0]]], [pt.y for pt in self.pts + [self.pts[0]]], c=color)
