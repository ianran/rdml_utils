import pdb, copy, random, cv2, itertools

import matplotlib.pyplot as plt
import numpy as np


from matplotlib import collections as mc
from scipy.spatial import Delaunay
from .location import Location
from h_signature import HSignature
from utils import samplePoints, QueueSet, euclideanDist



class HomotopyAugmentedVertex:
  """docstring for ClassName"""
  def __init__(self, vertex_id, base_vertex_id, base_vertex_loc, h_sig):
    self.id = vertex_id
    self.base_id = base_vertex_id  # Needed for Computing Connectivity
    self.loc = base_vertex_loc
    self.h_sig = h_sig
    self.dist = 0
    self.neighbors = []

  def getKey(self):
    return (self.loc.asTuple(), self.h_sig.signature)

  def __str__(self):
    return "Vertex ID: %s\n\tBase ID: %s\n\tLocation: %s\n\tH-Signature: %s" % (self.id,
                                                                                self.base_id,
                                                                                self.loc,
                                                                                self.h_sig)

class HomotopyAugmentedEdge(object):
  def __init__(self, edge_id, v1, v2):
    self.v1 = v1  # HomotopyAugmentedVertex Object
    self.v2 = v2  # HomotopyAugmentedVertex Object
    self.len = euclideanDist(v1.loc, v2.loc)


  def __str__(self):
    return "Vertex1:\n\tID: %s\n\tLoc: %s\n\tH-sig: %s\nVertex1:\n\tID: %s\n\tLoc: %s\n\tH-sig: %s" % (self.v1.id,
                                                                                                       self.v1.loc,
                                                                                                       self.v1.h_sig,
                                                                                                       self.v2.id,
                                                                                                       self.v2.loc,
                                                                                                       self.v2.h_sig)


class HomotopyAugmentedGraph:
  """docstring for prm_graph"""
  def __init__(self, vertices, connectivity, rep_pts, base_vertex_id=0, budget=float('inf')):
    # Base graph is the graph that defines connectivity (not homotopy augmented)
    self.base_graph_vertices = vertices             # List of N Location objects
    self.base_graph_connectivity = connectivity     # 2-d NxN connectivity matrix

    # Root vertex from which the homotopy augmented graph is grown
    self.home_vertex = HomotopyAugmentedVertex(0, base_vertex_id, self.base_graph_vertices[base_vertex_id], HSignature(()))

    # Representative Points for each obstacle
    self.rep_pts = sorted(rep_pts, key=lambda pt: pt.x) # List of Location objects sorted left to right

    self.homotopy_vertices = {}  # Dictionary of HomotopyAugmentedVertex objects indexed by (location), (h-sig)
    self.homotopy_edges = {}

    # Initialize Homotopy Vertices with home vertex
    self.homotopy_vertices[self.home_vertex.loc.asTuple(), ()] = self.home_vertex
    self.num_vertices = 1
    self.num_edges = 0

    self.buildGraph(budget)


  @classmethod
  def fromImage(cls, map_img, budget=500):
    img_height, img_width = map_img.shape[:2]

    gray_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, labeled_img = cv2.connectedComponents(255-thresh, 4, cv2.CV_32S)

    rep_pts = [samplePoints(labeled_img == label, 1)[0] for label in np.unique(labeled_img) if np.sum((labeled_img == label) * thresh) == 0]

    sampled_pts = samplePoints(gray_img, 100)


    connectivity_mat = np.zeros((len(sampled_pts), len(sampled_pts)))

    tri = Delaunay(sampled_pts)

    for simplex in tri.simplices:
      for v1, v2 in itertools.combinations(simplex, 2):
        connectivity_mat[v1, v2] = 1
        connectivity_mat[v2, v1] = 1


    sampled_pts = [Location(xlon=pt[0], ylat=pt[1]) for pt in sampled_pts]
    rep_pts = [Location(xlon=pt[0], ylat=pt[1]) for pt in rep_pts]

    # plt.figure()
    # ax = plt.gca()
    # plt.imshow(map_img, origin='lower')

    # plt.scatter([pt[0] for pt in rep_pts], [pt[1] for pt in rep_pts], c='r')
    # plt.scatter([pt[0] for pt in sampled_pts], [pt[1] for pt in sampled_pts], c='b')

    # edges = [[sampled_pts[v1], sampled_pts[v2]] for v1 in range(len(sampled_pts)) for v2 in range(len(sampled_pts)) if connectivity_mat[v1, v2]]

    # lc = mc.LineCollection(edges, color='k', linewidth=1)
    # ax.add_collection(lc)

    # plt.show(False)
    # pdb.set_trace()

    return cls(sampled_pts, connectivity_mat, rep_pts, budget=budget)


  def buildGraph(self, budget = float('inf')):
    # print "Constructing Homotopy Augmented Graph From"
    # print self.home_vertex
    # print "Budget of %.3f" % budget

    open_set = QueueSet()
    open_set.push(self.home_vertex)

    while not open_set.empty():
      current_vertex = open_set.pop()
      neighbors = np.nonzero(self.base_graph_connectivity[current_vertex.base_id])[0]

      for neighbor in neighbors:
        #compute h_sig from current vertex to neighbors
        neighbor_vertex_loc = self.base_graph_vertices[neighbor]

        h_sig = HSignature.fromPath([current_vertex.loc, neighbor_vertex_loc], self.rep_pts)
        d_dist = euclideanDist(neighbor_vertex_loc, current_vertex.loc)

        if (current_vertex.dist + d_dist) <= budget:
          neighbor_h_sig = current_vertex.h_sig + h_sig

          if not neighbor_h_sig.checkLooping():
            new_vertex_key = (self.base_graph_vertices[neighbor].asTuple(), neighbor_h_sig.signature)
            if new_vertex_key not in self.homotopy_vertices:
              #Node is not in graph
              new_vertex = HomotopyAugmentedVertex(self.num_vertices+1, neighbor, neighbor_vertex_loc, neighbor_h_sig)
              new_vertex.dist = current_vertex.dist + d_dist
              self.homotopy_vertices[new_vertex.getKey()] = new_vertex
              open_set.push(new_vertex)

              self.addEdge(self.num_edges, new_vertex, current_vertex)
              self.num_vertices += 1
              try:
                assert len(self.homotopy_vertices) == self.num_vertices
              except:
                pdb.set_trace()
            else:
              graph_vertex = self.homotopy_vertices[new_vertex_key]
              self.addEdge(self.num_edges, graph_vertex, current_vertex)





  def addEdge(self, edge_id, v1, v2):
    if (v1.loc, v2.loc) in self.homotopy_edges or (v2.loc, v1.loc) in self.homotopy_edges:
      return

    self.homotopy_edges[v1.loc, v2.loc] = HomotopyAugmentedEdge(edge_id, v1, v2)
    self.num_edges = self.num_edges + 1

    v1.neighbors.append((v2.loc, v2.h_sig))
    v2.neighbors.append((v1.loc, v1.h_sig))

    d_dist =  euclideanDist(v1.loc, v2.loc)

    # Update vertex distances if this edge results in a shorter distance
    #if v1.dist < v2.dist:
    if v2.dist > (v1.dist + d_dist):
      v2.dist = v1.dist + d_dist


    # Not sure we need this since the edges grow outward (1/26/2020)
    # #elif v2.dist < v1.dist:
    # elif v1.dist > (v2.dist + d_dist):
    #   v1.dist = v2.dist + d_dist





#   def getPath(self, start_pt, goal_pt):

#     vertex_costs = {}
#     open_set = {}
#     closed_set = {}

#     for v in self.vertices:
#       vertex_costs[v] = float('NaN')

#     open_set[start_pt.loc, start_pt.h_sig] = start_pt
#     vertex_costs[start_pt.loc, start_pt.h_sig] = 0

#     while len(open_set) > 0:
#       current_point = getBestPoint(open_set, goal_pt.loc, vertex_costs)

#       if dist(current_point.loc, goal_pt.loc) == 0:
#         if hSigEqual(goal_pt.h_sig, current_point.h_sig):
#           #print "Path Found"
#           return buildPath(self, vertex_costs, current_point)

#       closed_set[current_point.loc, current_point.h_sig] = current_point
#       del open_set[current_point.loc, current_point.h_sig]

#       neighbors = self.getAdjacentPoints(current_point)

#       for neighbor in neighbors:
#         neighbor_loc = neighbor.loc
#         neighbor_h_sig = neighbor.h_sig

#         if (neighbor_loc, neighbor_h_sig) not in closed_set:
#           if (neighbor_loc, neighbor_h_sig) not in open_set:
#             open_set[neighbor_loc, neighbor_h_sig] = neighbor

#           vertex_costs[neighbor_loc, neighbor_h_sig] = nanMin(vertex_costs[current_point.loc, current_point.h_sig] + dist(current_point.loc, neighbor_loc), vertex_costs[neighbor_loc, neighbor_h_sig])
#     print "Astar Failed"
#     return 0



#   def getAdjacentPoints(self, point):
#     #print "Point:", point.loc, point.h_sig
#     res = []
#     for e in point.edges:
#       res.append(self.vertices[e])

#     return res

# def getBestPoint(open_set, goal_loc, vertex_costs):
#   best_score = float('Inf')

#   for p in open_set:
#     point = open_set[p]
#     score = dist(point.loc, goal_loc) + vertex_costs[p]
#     if score < best_score:
#       best_point = point
#       best_score = score

#   return best_point


# def buildPath(graph, vertex_costs, goal_pt):
#   path = [goal_pt]
#   curr_pt = goal_pt

#   while vertex_costs[curr_pt.loc, curr_pt.h_sig] != 0:
#     neighbors = graph.getAdjacentPoints(curr_pt)
#     best_neighbor_score = float('inf')
#     #print "========================================="
#     for neighbor in neighbors:
#       #print neighbor.loc, vertex_costs[neighbor.loc, neighbor.h_sig]
#       if vertex_costs[neighbor.loc, neighbor.h_sig] + dist(curr_pt.loc, neighbor.loc) < best_neighbor_score:
#         best_neighbor = neighbor
#         best_neighbor_score = vertex_costs[neighbor.loc, neighbor.h_sig] + dist(curr_pt.loc, neighbor.loc)

#     path.append(best_neighbor)
#     curr_pt = best_neighbor
#   return path[::-1]



# def getPathLength(path, graph):
#   if len(path) < 2:
#     return 0
#   else:
#     path_len = 0

#     for node_id, node in enumerate(path[1:]):
#       prev_node = path[node_id]
#       path_len = path_len + dist(prev_node.loc, node.loc)
#     return path_len

# def buildDistMat(inspection_locs, h_graph):
#   distances = {}

#   locs = {}
#   for ii, loc in enumerate(inspection_locs):

#     h_sigs = []
#     for vert in h_graph.vertices:
#       if all([x == y for (x,y) in zip(loc, vert[0])]):
#         h_sigs.append(vert[1])
#     locs[loc] = h_sigs

#   print "Num Locs", len(locs)

#   loc_ids = locs.keys()

#   #for loc in locs:
#   for ii in range(len(loc_ids)):
#     print "Loc1:", ii
#     loc = loc_ids[ii]
#     h_sigs = locs[loc]
#     for jj in range(ii, len(loc_ids)):
#       print "Loc2:", jj
#       for h_sig in h_sigs:
#         loc2 = loc_ids[jj]
#       #for loc2 in locs:
#         h_sigs2 = locs[loc2]
#         for h_sig2 in h_sigs2:
#           v1 = h_graph.vertices[loc, h_sig]
#           v2 = h_graph.vertices[loc2, h_sig2]
#           path = [x.prm_vertex for x in h_graph.getPath(v1, v2)]
#           path_len = getPathLength(path, h_graph.base_graph)
#           distances[loc, loc2, h_sig, h_sig2] = path_len
#           distances[loc2, loc, h_sig2, h_sig] = path_len

#   print len(distances)
#   return distances, locs




# def main():
#   num_inspection_pts = 2
#   map_img = randomWorld()
#   base_graph = prm_graph(map_img, 500)
#   inspection_pts = [base_graph.vertices[x] for x in random.sample(base_graph.vertices, num_inspection_pts)]

#   print "Base Graph Complete"
#   rep_pts = getRepPts(map_img)
#   h_graph = HomotopyAugmentedGraph(map_img, base_graph, inspection_pts[0], rep_pts)
#   h_graph.buildGraph(300)
#   print "Done"


#   print "H Graph Vertices\n=============================================="
#   for v in h_graph.vertices:
#     print v
#   print "-----------------------------------------------------------------"
#   planning_pts = [h_graph.vertices[x] for x in random.sample(h_graph.vertices, num_inspection_pts)]

#   print "Planning path between:"
#   print planning_pts[0].loc, planning_pts[0].h_sig
#   print planning_pts[1].loc, planning_pts[1].h_sig

#   h_path = h_graph.getPath(planning_pts[0], planning_pts[1])


#   prm_graph_img = base_graph.dispGraph()
#   for p1_id, p2 in enumerate(h_path[1:]):
#     p1 = h_path[p1_id]
#     print "point", p1.loc, p1.h_sig
#     cv2.line(prm_graph_img, tuple(p1.loc), tuple(p2.loc), (0, 125, 125), 3)

#   for pt_num, pt in enumerate(rep_pts):
#     cv2.line(prm_graph_img, pt, (pt[0], 0), (125, 125, 125), 1)
#     cv2.putText(prm_graph_img, str(pt_num), pt, cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
#     cv2.circle(prm_graph_img, pt, 3, (0, 0, 255), -1)

#   for pt in h_path:
#     cv2.circle(prm_graph_img, pt.loc, 2, (0, 0, 0), -1)

#   cv2.circle(prm_graph_img, h_path[0].loc, 4, (0, 255, 0), -1)
#   cv2.circle(prm_graph_img, h_path[-1].loc, 4, (0, 0, 255), -1)

#   cv2.imshow("PRM", prm_graph_img)
#   cv2.waitKey(0)








if __name__ == '__main__':
  main()
