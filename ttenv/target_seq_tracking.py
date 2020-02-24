import gym
from gym import spaces, logger

import numpy as np
from numpy import linalg as LA
import os, copy

from ttenv.agent_models import *
from ttenv.policies import *
from ttenv.belief_tracker import KFbelief
import ttenv.util as util
from ttenv.target_tracking import TargetTrackingEnv1
from ttenv.target_imtracking import TargetTrackingEnv5, TargetTrackingEnv8

class TargetTrackingEnv1_SEQ(TargetTrackingEnv1):
    def __init__(self, num_targets=1, map_name='empty', is_training=True,
                known_noise=True, target_path_dir=None, **kwargs):
        TargetTrackingEnv1.__init__(self, num_targets=num_targets,
            map_name=map_name, is_training=is_training, known_noise=known_noise, **kwargs)
        self.id = 'TargetTracking-v1_SEQ'
        self.history_length = 5
        rel_vel_limit = self.limit['state'][1][2]
        state_limit_l = [0.0, -np.pi, -rel_vel_limit, -10*np.pi]
        state_limit_l.extend(self.history_length * [-50.0])
        state_limit_l.append(0.0)
        state_limit_h = [600.0, np.pi, rel_vel_limit, 10*np.pi]
        state_limit_h.extend(self.history_length * [50.0])
        state_limit_h.append(2.0)
        self.limit['state'] = [np.concatenate((state_limit_l*num_targets, [0.0, -np.pi])),
                               np.concatenate((state_limit_h*num_targets, [self.sensor_r, np.pi]))]
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)

    def reset(self, **kwargs):
        state = super().reset(**kwargs)
        self.logdetcov_history =[Storage(max_capacity = self.history_length,
            init_value = np.log(LA.det(self.belief_targets[i].cov))) for i in range(self.num_targets)]
        return self.add_history_to_state(state)

    def step(self, action):
        state, reward, done, info = super().step(action)
        new_state = self.add_history_to_state(state)
        return new_state, reward, done, info

class TargetTrackingEnv5_SEQ(TargetTrackingEnv5):
    def __init__(self, num_targets=1, map_name='empty', is_training=True,
                                        known_noise=True, im_size=28, **kwargs):
        TargetTrackingEnv5.__init__(self, num_targets=num_targets,
            map_name=map_name, is_training=is_training, known_noise=known_noise, im_size=im_size, **kwargs)
        self.id = 'TargetTracking-v5_SEQ'
        self.history_length = 5
        state_limit_l = self.limit['state'][0][:4]
        state_limit_l.extend(self.history_length * [self.limit['state'][0][5]])
        state_limit_l.append(0.0)
        state_limit_h = self.limit['state'][1][:4]
        state_limit_h.extend(self.history_length * [self.limit['state'][1][5]])
        state_limit_h.append(2.0)
        self.limit['state'] = [np.concatenate((state_limit_l*num_targets, self.limit['state'][0][-2:])),
                               np.concatenate((state_limit_h*num_targets, self.limit['state'][1][-2:]))]
        self.observation_space = spaces.Box(np.concatenate((
            np.zeros(im_size*im_size,), self.limit['state'][0])),
            np.concatenate((np.ones(im_size*im_size,), self.limit['state'][1])),
            dtype=np.float32)

    def reset(self, **kwargs):
        state = super().reset(**kwargs)
        im_state = state[:-len(self.state)]
        self.logdetcov_history =[Storage(max_capacity = self.history_length,
            init_value = np.log(LA.det(self.belief_targets[i].cov))) for i in range(self.num_targets)]
        new_state = self.add_history_to_state(self.state)
        return np.concatenate((im_state, new_state))

    def step(self, action):
        state, reward, done, info = super().step(action)
        im_state = state[:-len(self.state)]
        new_state = self.add_history_to_state(self.state)
        return np.concatenate((im_state, new_state)), reward, done, info

class TargetTrackingEnv8_SEQ(TargetTrackingEnv8):
    def __init__(self, num_targets=1, map_name='empty', is_training=True,
                                        known_noise=True, im_size=28, **kwargs):
        TargetTrackingEnv8.__init__(self, num_targets=num_targets,
            map_name=map_name, is_training=is_training, known_noise=known_noise, im_size=im_size, **kwargs)
        self.id = 'TargetTracking-v8_SEQ'
        self.history_length = 5
        state_limit_l = self.limit['state'][0][:4]
        state_limit_l.extend(self.history_length * [self.limit['state'][0][5]])
        state_limit_l.append(0.0)
        state_limit_h = self.limit['state'][1][:4]
        state_limit_h.extend(self.history_length * [self.limit['state'][1][5]])
        state_limit_h.append(2.0)
        self.limit['state'] = [np.concatenate((state_limit_l*num_targets, self.limit['state'][0][-2:])),
                               np.concatenate((state_limit_h*num_targets, self.limit['state'][1][-2:]))]
        self.observation_space = spaces.Box(np.concatenate((
            -np.ones(5*im_size*im_size,), self.limit['state'][0])),
            np.concatenate((np.ones(5*im_size*im_size,), self.limit['state'][1])),
            dtype=np.float32)

    def reset(self, **kwargs):
        state = super().reset(**kwargs)
        im_state = state[:-len(self.state)]
        self.logdetcov_history =[Storage(max_capacity = self.history_length,
            init_value = np.log(LA.det(self.belief_targets[i].cov))) for i in range(self.num_targets)]
        new_state = self.add_history_to_state(self.state)
        return np.concatenate((im_state, new_state))

    def step(self, action):
        state, reward, done, info = super().step(action)
        im_state = state[:-len(self.state)]
        new_state = self.add_history_to_state(self.state)
        return np.concatenate((im_state, new_state)), reward, done, info

class Storage():
    def __init__(self, max_capacity, init_value):
        self.storage = max_capacity * [init_value]
        self.max_capacity = max_capacity

    def add(self, val):
        self.storage.pop(0)
        self.storage.append(val)

    def get_values(self):
        return self.storage
