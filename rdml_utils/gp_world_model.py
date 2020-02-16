
import pdb, GPy, itertools, numbers, time

import numpy as np
from scipy.interpolate import NearestNDInterpolator
from roms import getROMSData
from utils import getFencedData
from location import Location, Observation
import deepdish as dd
import matplotlib.pyplot as plt

class GPWorldModel(object):
  """docstring for GPWorldModel"""
  def __init__(self, gp_num_dimensions, gp_x, gp_y, gp_initial_variance, gp_length_scales, gp_mean, use_sparse_gp, num_inducing):
    kernel = GPy.kern.RBF(input_dim=gp_num_dimensions, variance=gp_initial_variance, lengthscale=gp_length_scales, ARD=True)

    assert gp_x.shape[1] == gp_num_dimensions

    if use_sparse_gp:
      self.model = GPy.models.SparseGPRegression(gp_x, gp_y, kernel, num_inducing=num_inducing)
    else:
      self.model = GPy.models.GPRegression(gp_x, gp_y, kernel)

    self.mean_estimate = gp_mean # This can either be a staic value or another GP

  def optimize(self, msgs=False):
    self.model.optimize('bfgs', messages=msgs)

  def queryGPPoint(self, point):
    query_point = np.reshape(np.array(point), (-1, len(point)))
    res = self.model.predict(query_point)[0] + self.getPriorPoint(point)
    return res

  def getPriorMean(self, x_ticks, y_ticks):
    shape = (len(x_ticks), len(y_ticks))
    if isinstance(self.mean_estimate, GPWorldModel):
      prior_mean, _ = self.mean_estimate.getGPModel(x_ticks, y_ticks)
      return prior_mean
    elif isinstance(self.mean_estimate, numbers.Number):
      return np.ones(shape) * self.mean_estimate

  def getPriorPoint(self, point):
    if isinstance(self.mean_estimate, GPWorldModel):
      return self.mean_estimate.queryGPPoint(point)[0][0]
    elif isinstance(self.mean_estimate, numbers.Number):
      return self.mean_estimate




class GPStaticWorldModel(GPWorldModel):
  """docstring for GPWorldModel"""
  def __init__(self, obs, gp_initial_variance, gp_length_scales, gp_mean, sparse=False, num_inducing=100):
    self.mean_estimate = gp_mean

    self.gp_x = np.array([[x.loc.lon, x.loc.lat] for x in obs])
    self.gp_y = np.array([x.data for x in obs])
    self.prior_y = np.array([self.getPriorPoint([x.loc.lon, x.loc.lat, x.time]) for x in obs])
    
    self.gp_y = np.reshape(self.gp_y - self.prior_y, (len(self.gp_y), -1))

    GPWorldModel.__init__(self, 2, self.gp_x, self.gp_y, gp_initial_variance, gp_length_scales, gp_mean, use_sparse_gp=sparse, num_inducing=num_inducing)

  def queryPoint(self, lon, lat, time=0.0):
    #assumes Lon, Lat, Time
    query_point = [lon, lat]
    return self.queryGPPoint(query_point)

  def getGPModel(self, x_ticks, y_ticks):
    # Query the GP Model to get the Mean and Variance Estimates
    shape = (len(x_ticks), len(y_ticks))

    query = np.array(list(itertools.product(x_ticks, y_ticks)))

    zz = self.model.predict(query)

    prediction_mean = np.array(zz[0]).reshape(shape)
    prediction_variance = np.array(zz[1]).reshape(shape)
    
    prior_prediction_mean = self.getPriorMean(x_ticks, y_ticks)
    estimate = prediction_mean + prior_prediction_mean 

    return estimate, prediction_variance




class GPTimeVaryingWorldModel(GPWorldModel):
  """docstring for GPWorldModel"""
  def __init__(self, obs, gp_initial_variance, gp_length_scales, gp_mean, gp_time_scale_factor=1.0, sparse=False, num_inducing=100):
    self.mean_estimate = gp_mean
    self.gp_time_scale_factor = gp_time_scale_factor

    self.gp_x = np.array([[x.loc.lon, x.loc.lat, x.time/gp_time_scale_factor] for x in obs])
    self.gp_y = np.array([x.data for x in obs])
    self.prior_y = np.array([self.getPriorPoint([x.loc.lon, x.loc.lat, x.time]) for x in obs])
    self.gp_y = np.reshape(self.gp_y - self.prior_y, (len(self.gp_y), -1))


    GPWorldModel.__init__(self, 3, self.gp_x, self.gp_y, gp_initial_variance, gp_length_scales, gp_mean, use_sparse_gp=sparse, num_inducing=num_inducing)

  
  def queryPoint(self, lon, lat, time=0.0):
    #assumes Lon, Lat, Time
    query_point = [lon, lat, Time]
    return self.queryGPPoint(query_point)

  def getGPModel(self, query_time, x_ticks, y_ticks):
    # Query the GP Model to get the Mean and Variance Estimates 
    shape = (len(x_ticks), len(y_ticks))

    query = np.array(list(itertools.product(x_ticks, y_ticks)))
    query = np.hstack([query, np.ones((len(query), 1))*(query_time/self.gp_time_scale_factor)])

    zz = self.model.predict(np.array(query))

    prediction_mean = np.array(zz[0]).reshape(shape)
    prediction_variance = np.array(zz[1]).reshape(shape)
    prior_prediction_mean = self.getPriorMean(x_ticks, y_ticks)
    
    estimate = prediction_mean + prior_prediction_mean 

    return estimate, prediction_variance




class GPComboTimeVaryingWorldModel(GPWorldModel):
  """docstring for GPComboWorldModel"""
  def __init__(self, obs, gp_initial_variance, gp_length_scales, x_ticks, y_ticks, gp_time_scale_factor=1.0, nn_omit_time=-float('inf'), sparse=False, num_inducing=100):
    self.gp_time_scale_factor = gp_time_scale_factor
    world_est_time = time.time()

    filtered_obs = [x for x in obs if x.time >= nn_omit_time] # Filter out all observations that occurred before the omit time cutoff

    pts       = np.array([[o.loc.lon, o.loc.lat] for o in filtered_obs])
    data      = np.array([o.data for o in filtered_obs])
   
    nearestInterp = NearestNDInterpolator(pts, data)
   
    psuedo_obs_x = np.linspace(x_ticks[0], x_ticks[-1], 10) 
    psuedo_obs_y = np.linspace(y_ticks[0], y_ticks[-1], 10) 

    q_pts = np.array([coord for coord in itertools.product(psuedo_obs_x, psuedo_obs_y)])
    
    pseudo_obs = nearestInterp(q_pts)
    pseudo_obs = [Observation(Location(xlon=pt[0], ylat=pt[1]), data=o, time=0.0) for pt, o in zip(q_pts, pseudo_obs)]

    prior_mean_estimate = np.mean([x.data for x in pseudo_obs])

    prior_gp = GPStaticWorldModel(pseudo_obs, gp_initial_variance, gp_length_scales[:2], prior_mean_estimate)

    self.mean_estimate = prior_gp

    self.gp_x = np.array([[x.loc.lon, x.loc.lat, x.time/self.gp_time_scale_factor] for x in obs])
    self.gp_y = np.array([x.data for x in obs])

    self.prior_y = np.array([self.getPriorPoint([x.loc.lon, x.loc.lat, x.time]) for x in obs])
    self.gp_y = np.reshape(self.gp_y - self.prior_y, (len(self.gp_y), -1))

    GPWorldModel.__init__(self, 3, self.gp_x, self.gp_y, gp_initial_variance, gp_length_scales, prior_gp, use_sparse_gp=sparse, num_inducing=num_inducing)

  def queryPoint(self, lon, lat, time=0.0):
    #assumes Lon, Lat, Time
    query_point = [lon, lat, Time]
    return self.queryGPPoint(query_point)

  def getGPModel(self, query_time, x_ticks, y_ticks):
    # Query the GP Model to get the Mean and Variance Estimates 
    shape = (len(x_ticks), len(y_ticks))

    query = np.array(list(itertools.product(x_ticks, y_ticks)))
    query = np.hstack([query, np.ones((len(query), 1))*(query_time/self.gp_time_scale_factor)])

    zz = self.model.predict(np.array(query))

    prediction_mean = np.array(zz[0]).reshape(shape)
    prediction_variance = np.array(zz[1]).reshape(shape)
    prior_prediction_mean = self.getPriorMean(x_ticks, y_ticks)
    
    estimate = prediction_mean + prior_prediction_mean 

    return estimate, prediction_variance