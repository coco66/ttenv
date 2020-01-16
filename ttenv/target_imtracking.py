"""Target Tracking Environments for Reinforcement Learning. OpenAI gym format
[Vairables]
d: radial coordinate of a belief target in the learner frame
alpha : angular coordinate of a belief target in the learner frame
ddot : radial velocity of a belief target in the learner frame
alphadot : angular velocity of a belief target in the learner frame
Sigma : Covariance of a belief target
o_d : linear distance to the closet obstacle point
o_alpha : angular distance to the closet obstacle point
[Environment Descriptions]
TargetTrackingEnv5 : Local Image-based Double Integrator Target model with KF belief tracker
    RL state: [local_map_image, [d, alpha, ddot, alphadot, logdet(Sigma), observed] * nb_targets, [o_d, o_alpha]]
    Target : Double Integrator model, [x,y,xdot,ydot]
    Belief Target : KF, Double Integrator model
"""
import gym
from gym import spaces, logger

import numpy as np
from numpy import linalg as LA
import os, copy

from ttenv.maps import map_utils
from ttenv.agent_models import *
from ttenv.policies import *
from ttenv.belief_tracker import KFbelief
import ttenv.util as util
from ttenv.target_tracking import TargetTrackingEnv1

class TargetTrackingEnv5(TargetTrackingEnv1):
    def __init__(self, num_targets=1, map_name='empty', is_training=True,
                                        known_noise=True, im_size=50, **kwargs):
        TargetTrackingEnv1.__init__(self, num_targets=num_targets,
            map_name=map_name, is_training=is_training, known_noise=known_noise, **kwargs)
        self.id = 'TargetTracking-v5'
        self.im_size = im_size
        self.local_mapmin_g = None
        self.observation_space = spaces.Box(np.concatenate((
            np.zeros(im_size*im_size,), self.limit['state'][0])),
            np.concatenate((np.ones(im_size*im_size,), self.limit['state'][1])),
            dtype=np.float32)

    def reset(self, init_random = True):
        self.state = []
        init_pose = self.get_init_pose(init_random=init_random)
        self.agent.reset(init_pose['agent'])
        for i in range(self.num_targets):
            self.belief_targets[i].reset(
                        init_state=np.concatenate((init_pose['belief_targets'][i][:2], np.zeros(2))),
                        init_cov=self.target_init_cov)
            self.targets[i].reset(np.concatenate((init_pose['targets'][i][:2], self.target_init_vel)))
            r, alpha, _ = util.xyg2polarb(self.belief_targets[i].state[:2],
                                 self.agent.state[:2], self.agent.state[2])
            logdetcov = np.log(LA.det(self.belief_targets[i].cov))
            self.state.extend([r, alpha, 0.0, 0.0, logdetcov, 0.0])

        self.state.extend([self.sensor_r, np.pi])
        self.state = np.array(self.state)
        self.local_map, self.local_mapmin_g = map_utils.local_map(self.MAP, self.im_size, self.agent.state)
        return np.concatenate((self.local_map.flatten(), self.state))

    def step(self, action):
        action_vw = self.action_map[action]
        _ = self.agent.update(action_vw, [t.state[:2] for t in self.targets])

        observed = []
        for i in range(self.num_targets):
            self.targets[i].update(self.agent.state[:2])
            # Observe
            obs = self.observation(self.targets[i])
            observed.append(obs[0])
            self.belief_targets[i].predict() # Belief state at t+1
            if obs[0]: # if observed, update the target belief.
                self.belief_targets[i].update(obs[1], self.agent.state)

        obstacles_pt = map_utils.get_cloest_obstacle(self.MAP, self.agent.state)
        reward, done, mean_nlogdetcov = self.get_reward(obstacles_pt, observed, self.is_training)
        self.state = []
        if obstacles_pt is None:
            obstacles_pt = (self.sensor_r, np.pi)
        for i in range(self.num_targets):
            r_b, alpha_b, _ = util.xyg2polarb(self.belief_targets[i].state[:2],
                                 self.agent.state[:2], self.agent.state[2])
            r_dot_b, alpha_dot_b = util.xyg2polarb_dot_2(
                                    self.belief_targets[i].state[:2],
                                    self.belief_targets[i].state[2:],
                                    self.agent.state[:2], self.agent.state[-1],
                                    action_vw[0], action_vw[1])
            self.state.extend([r_b, alpha_b, r_dot_b, alpha_dot_b,
                np.log(LA.det(self.belief_targets[i].cov)), float(observed[i])])
        self.state.extend([obstacles_pt[0], obstacles_pt[1]])
        self.state = np.array(self.state)
        self.local_map, self.local_mapmin_g = map_utils.local_map(self.MAP, self.im_size, self.agent.state)
        return np.concatenate((self.local_map.flatten(), self.state)), reward, done, {'mean_nlogdetcov': mean_nlogdetcov}
