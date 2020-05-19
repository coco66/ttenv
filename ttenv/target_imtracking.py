"""Target Tracking Environments for Reinforcement Learning with an Image Input. OpenAI gym format
[Environment Descriptions]
TargetTrackingEnv4 : Local Image-based Double Integrator Target model with KF belief tracker
    RL state: [local_map_image, [d, alpha, ddot, alphadot, logdet(Sigma), observed] * nb_targets, [o_d, o_alpha]]
    Target : Double Integrator model, [x,y,xdot,ydot]
    Belief Target : KF, Double Integrator model

TargetTrackingEnv5 : Local map & Local visit frequency maps of outside the front range - left, right, front, back are given.
    Therefore, the image input fed to the convolutional neural network has five depth.
    This intend to use a smaller image size.
"""
from gym import spaces, logger

import numpy as np
from numpy import linalg as LA

from ttenv.agent_models import *
from ttenv.policies import *
from ttenv.belief_tracker import KFbelief
import ttenv.util as util
from ttenv.target_tracking import TargetTrackingEnv1

class TargetTrackingEnv4(TargetTrackingEnv1):
    def __init__(self, num_targets=1, map_name='empty', is_training=True,
                                        known_noise=True, im_size=28, **kwargs):
        self.im_size = im_size
        TargetTrackingEnv1.__init__(self, num_targets=num_targets,
            map_name=map_name, is_training=is_training, known_noise=known_noise, **kwargs)
        self.id = 'TargetTracking-v4'
        self.local_mapmin_g = None

    def set_limits(self, target_speed_limit=None):
        super().set_limits(target_speed_limit)
        self.observation_space = spaces.Box(
            np.concatenate((-np.ones(self.im_size*self.im_size,), self.limit['state'][0])),
            np.concatenate((np.ones(self.im_size*self.im_size,), self.limit['state'][1])),
            dtype=np.float32)

    def reset(self, **kwargs):
        _ = super().reset(**kwargs)

        # Get the local maps.
        map_state = self.map_state_func()
        return np.concatenate((map_state, self.state))

    def step(self, action):
        _, reward, done, info = super().step(action)

        # Get the local maps.
        map_state = self.map_state_func()
        return np.concatenate((map_state, self.state)), reward, done, info

    def map_state_func(self):
        self.local_map, self.local_mapmin_g, _ = self.MAP.local_map(
                                                    self.im_size, self.agent.state)
        # normalize the maps
        self.local_map = [(self.local_map - 0.5) * 2]
        self.local_mapmin_g = [self.local_mapmin_g]
        return self.local_map[0].flatten()

class TargetTrackingEnv5(TargetTrackingEnv4):
    def __init__(self, num_targets=1, map_name='empty', is_training=True,
                                        known_noise=True, im_size=28, **kwargs):
        TargetTrackingEnv4.__init__(self, num_targets=num_targets,
            map_name=map_name, is_training=is_training, known_noise=known_noise, im_size=im_size, **kwargs)
        self.id = 'TargetTracking-v5'

    def reset(self, **kwargs):
        self.MAP.reset_visit_freq_map()
        return super().reset(**kwargs)

    def set_limits(self, target_speed_limit=None):
        super().set_limits(target_speed_limit)
        self.observation_space = spaces.Box(
            np.concatenate((-np.ones(5*self.im_size*self.im_size,), self.limit['state'][0])),
            np.concatenate((np.ones(5*self.im_size*self.im_size,), self.limit['state'][1])),
            dtype=np.float32)

    def map_state_func(self):
        # Update the visit frequency map.
        b_speed = np.mean([np.sqrt(np.sum(self.belief_targets[i].state[2:]**2)) for i in range(self.num_targets)])
        decay_factor = np.exp(self.sampling_period*b_speed/self.sensor_r*np.log(0.7))
        self.MAP.update_visit_freq_map(self.agent.state, decay_factor)

        self.local_map, self.local_mapmin_g, _ = self.MAP.local_map(
                                                self.im_size, self.agent.state)
        _, local_mapmin_gs, local_visit_maps = self.MAP.local_visit_map_surroundings(
                                                self.im_size, self.agent.state)
        # normalize the maps
        self.local_map = [(self.local_map - 0.5) * 2]
        for i in range(4):
            self.local_map.append(local_visit_maps[i] - 1.0)

        self.local_mapmin_g = [self.local_mapmin_g]
        self.local_mapmin_g.extend(local_mapmin_gs)

        return np.array(self.local_map).T.flatten()
