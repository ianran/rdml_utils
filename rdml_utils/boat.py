# Class to make an autonmous boat

from utils import State
from location import Location, LocDelta

import random as random
import math as math

class Boat():

	def __init__(self, x = 0.0, y = 0.0, theta = 0.0, vel = 1.0, max_turn = math.pi / 4, boat = None):
		if boat is not None:

			self.loc = boat.loc
			self.theta = boat.theta
			self.vel = boat.vel
			self.max_turn = boat.max_turn

		else:

			self.loc = Location(xlon = x, ylat = y)
			self.theta = theta
			self.vel = vel
			self.max_turn = max_turn

	def step(self, x_vel, y_vel, theta_vel, move_noise, turn_noise, scale=1):
		self.loc += (LocDelta(d_ylat = y_vel, d_xlon = x_vel) + LocDelta(d_ylat = random.gauss(0,move_noise), d_xlon = random.gauss(0,move_noise))) * scale
		self.theta += (theta_vel + random.gauss(0,turn_noise)) * scale
		self.theta = (self.theta + (math.pi * 2)) % (math.pi * 2)

	def calVels(self, u):
		x_vel = self.vel * math.cos(self.theta)
		y_vel = self.vel * math.sin(self.theta)
		if abs(u) < self.max_turn:
			theta_vel = u
		else:
			theta_vel = self.max_turn * (u / abs(u))

		return x_vel, y_vel, theta_vel

	def calControl(self, goal, radius):
		dist = goal - self.loc
		
		if (dist.getMagnitude() > radius):
			des_head = math.atan2(dist.d_ylat, dist.d_xlon)
			con = des_head - self.theta
			
			if abs(con) > math.pi:
				if con / abs(con) < 0:
					con = 2 * math.pi + con
				else:
					con = con - 2 * math.pi

			return con

		return None

	def getParamsAsString(self):
		return ['Boat Location: '+str(self.loc), 'Boat Theta: '+str(self.theta), 'Boat Velocity: '+str(self.vel), 'Boat Max Turn Speed: '+str(self.max_turn)]