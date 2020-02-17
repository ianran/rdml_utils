
import pdb, time
from rdml_utils import Location
from utils import rayIntersection

class HSignature(object):
  def __init__(self, signature):
    self.signature = signature  # Tuple of signed nonzero integers
    self.reduce()

  def invert(self):
    self.signature = tuple(-1*item for item in self.signature)

  def checkLooping(self):
    # Return True if there is a loop in the H signature 
    return not len(set(self.signature)) == len(self.signature)

  def reduce(self):
    if len(self.signature) < 2:
      return

    for h1_idx in range(len(self.signature)-1):
      if self.signature[h1_idx] == -self.signature[h1_idx+1]:
        self.signature = self.signature[:h1_idx] + self.signature[h1_idx+2:]
        break
    else:
      return

    self.reduce()

  def contains(self, other):
    # True if Other is an extension of Self
    l_self = len(self.signature)
    l_other = len(other.signature)

    if l_self == l_other:
      return self == other
    
    elif l_self > l_other:
      return self.signature[:l_self] == other.signature
    
    else:
      return False


  def __len__(self):
    return len(self.signature)


  def __str__(self):
    return "<" + ",".join([str(x) if x > 0 else str(abs(x)) + "*" for x in self.signature]) + ">"


  def __eq__(self, other):
    return self.signature == other.signature

  def __ne__(self, other):
    return not self == other

  def __add__(self, other):
    return HSignature(self.signature + other.signature)

  def __hash__(self):
    s = str(self)
    return s.__hash__()





  @classmethod
  def fromPath(cls, path, representative_pts):
    h_sig = ()
    
    # Sorted representitive points from left to right
    sorted_named_rep_pts = list(enumerate(sorted(representative_pts, key=lambda pt: pt.x)))

    for p1, p2 in zip(path[:-1], path[1:]):
      if p1.x <= p2.x:  #if the segment goes from left to right it is a positive crossing
        segment = [p1, p2]
        for pt_idx, rep_pt in sorted_named_rep_pts:
          if rayIntersection(segment, rep_pt):
            h_sig += (pt_idx+1,)
      else: 
        segment = [p2, p1] 
        for pt_idx, rep_pt in sorted_named_rep_pts[::-1]:
          if p1.x > p2.x: # if the segment goes from right to left, it is a negative crossing
            if rayIntersection(segment, rep_pt):
              h_sig += (-(pt_idx+1),)

    return cls(h_sig)


if __name__ == '__main__':
  path = [Location(xlon=0., ylat=0.), Location(xlon=1., ylat=0.), Location(xlon=2., ylat=1.), Location(xlon=0.5, ylat=1.)]
  ref_pts = [Location(xlon=0.25, ylat=-1.), Location(xlon=0.75, ylat=-1.), Location(xlon=1.5, ylat=-1.)]

  foo = HSignature.fromPath(path, ref_pts)


