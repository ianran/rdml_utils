# A class to hold a policy for realizable path planning

from scipy import spatial
import os, sys
import numpy as np
import pdb
import pickle

if sys.version_info[0] < 3:
    # python 2
    from utils import genState, state2Dis
    from location import Location, LocDelta
else:
    # python 3
    from rdml_utils.utils import genState, state2Dis
    from rdml_utils.location import Location, LocDelta


class Policy(object):

    # Policy file name and the parameters for the kd_tree
    def __init__(self, policy_file_name, allow_d, d_dis, allow_t, t_dis):

        self.kd_tree = self.buildKDTree(allow_d, allow_t, d_dis, t_dis)
        self.policy = self.loadPolicy(policy_file_name)

    # Function to build a kd_tree from given range of distances and heading and thier discritization levels
    def buildKDTree(self, d, t, d_level, t_level):
        d_c, t_c, d_n, t_n = np.mgrid[d[0]:d[1]:d_level, t[0]:t[1]:t_level, d[0]:d[1]:d_level, t[0]:t[1]:t_level]
        return spatial.KDTree(zip(d_c.ravel(), t_c.ravel(), d_n.ravel(), t_n.ravel()))

    # Function to load a policy from a given pickle file
    def loadPolicy(self, policy_file_name):
        if os.path.exists(policy_file_name):
            policy = pickle.load(open(policy_file_name,'r'))
            return policy
        else:
            return None

    # Given a boat, goals, and current goal index return the desired action
    def getAction(self, boat, goal_list, goal_index):
        # pdb.set_trace()
        dis_state = state2Dis(genState.fromLoc(boat.loc, goal_list, goal_index, cur_h=boat.theta), self.kd_tree)
        s = genState.fromTuple(dis_state + (goal_index,))

        if s.asTupleNoGoal() not in self.policy.keys():
            s_t = min(self.policy.keys(), key=lambda x: abs(s.d_cur - x[0]) + abs(s.t_cur - x[1]) + abs(s.d_next - x[2]) + abs(s.t_next  - x[3]))
        else:
            s_t = s.asTupleNoGoal()

        acts = [(self.policy[s_t].acts[i], x) for i, x in enumerate(self.policy[s_t].act_cost)]
        act = min(acts,key=lambda item:item[1])

        return act[0]

class PolicySim(object):

    def __init__(self, max_iter_num, radius, move_noise, turn_noise, scale):

        self.max_iter_num = max_iter_num
        self.radius = radius
        self.move_noise = move_noise
        self.turn_noise = turn_noise
        self.scale = scale


    def runSim(self, policy, boat, goals):

        g_index = 0
        step_num = 0
        end = False

        path = [boat.loc]

        while(step_num < self.max_iter_num and not end):

            action = policy.getAction(boat, goals, g_index)
            g_index += action

            if g_index < 0:
                g_index = 0

            if g_index < len(goals):

                u = boat.calControl(goals[g_index], self.radius)

                if u is not None:
                    x_vel, y_vel, theta_vel = boat.calVels(u)
                else:
                    g_index += 1
                    x_vel, y_vel, theta_vel = boat.calVels(0.0)
                    if g_index >= len(goals):
                        end = True

                boat.step(x_vel, y_vel, theta_vel, self.move_noise, self.turn_noise, scale=self.scale)
                path.append(boat.loc)

                step_num += 1
                if g_index == (len(goals)-1):
                    end = not (self.radius < (boat.loc - goals[g_index]).getMagnitude())

            else:
                end = True

        return path
