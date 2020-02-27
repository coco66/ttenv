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

NUM_TARGET_DEP_VARS = 6
NUM_TARGET_INDEP_VARS = 2
LOGDETCOV_IDX = 4
HISTORY_LENGTH = 3

def get_state_limit(state_limit, num_targets):
    state_limit_l = np.concatenate((state_limit[0][:LOGDETCOV_IDX],
                            HISTORY_LENGTH * [state_limit[0][LOGDETCOV_IDX]],
                            [state_limit[0][LOGDETCOV_IDX+1]]))
    state_limit_h = np.concatenate((state_limit[1][:LOGDETCOV_IDX],
                            HISTORY_LENGTH * [state_limit[1][LOGDETCOV_IDX]],
                            [state_limit[1][LOGDETCOV_IDX+1]]))
    new_state_limit = [np.concatenate((state_limit_l*num_targets, state_limit[0][-2:])),
                           np.concatenate((state_limit_h*num_targets, state_limit[1][-2:]))]
    return new_state_limit

class TargetTrackingEnv1_SEQ(TargetTrackingEnv1):
    def __init__(self, num_targets=1, map_name='empty', is_training=True,
                known_noise=True, target_path_dir=None, **kwargs):
        TargetTrackingEnv1.__init__(self, num_targets=num_targets,
            map_name=map_name, is_training=is_training, known_noise=known_noise, **kwargs)
        self.id = 'TargetTracking-v1_SEQ'
        self.history_length = HISTORY_LENGTH
        self.limit['state'] = get_state_limit(self.limit['state'], num_targets)
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)

    def reset(self, **kwargs):
        state = super().reset(**kwargs)
        self.logdetcov_history =[Storage(max_capacity = self.history_length,
            init_value = np.log(LA.det(self.belief_targets[i].cov))) for i in range(self.num_targets)]
        return self.add_history_to_state(state, NUM_TARGET_DEP_VARS, NUM_TARGET_INDEP_VARS, LOGDETCOV_IDX)

    def step(self, action):
        state, reward, done, info = super().step(action)
        new_state = self.add_history_to_state(state, NUM_TARGET_DEP_VARS, NUM_TARGET_INDEP_VARS, LOGDETCOV_IDX)
        return new_state, reward, done, info

class TargetTrackingEnv5_SEQ(TargetTrackingEnv5):
    def __init__(self, num_targets=1, map_name='empty', is_training=True,
                                        known_noise=True, im_size=28, **kwargs):
        TargetTrackingEnv5.__init__(self, num_targets=num_targets,
            map_name=map_name, is_training=is_training, known_noise=known_noise, im_size=im_size, **kwargs)
        self.id = 'TargetTracking-v5_SEQ'
        self.history_length = HISTORY_LENGTH
        self.limit['state'] = get_state_limit(self.limit['state'], num_targets)
        self.observation_space = spaces.Box(np.concatenate((
            np.zeros(im_size*im_size,), self.limit['state'][0])),
            np.concatenate((np.ones(im_size*im_size,), self.limit['state'][1])),
            dtype=np.float32)

    def reset(self, **kwargs):
        state = super().reset(**kwargs)
        im_state = state[:-len(self.state)]
        self.logdetcov_history =[Storage(max_capacity = self.history_length,
            init_value = np.log(LA.det(self.belief_targets[i].cov))) for i in range(self.num_targets)]
        new_state = self.add_history_to_state(self.state, NUM_TARGET_DEP_VARS, NUM_TARGET_INDEP_VARS, LOGDETCOV_IDX)
        return np.concatenate((im_state, new_state))

    def step(self, action):
        state, reward, done, info = super().step(action)
        im_state = state[:-len(self.state)]
        new_state = self.add_history_to_state(self.state, NUM_TARGET_DEP_VARS, NUM_TARGET_INDEP_VARS, LOGDETCOV_IDX)
        return np.concatenate((im_state, new_state)), reward, done, info

class TargetTrackingEnv8_SEQ(TargetTrackingEnv8):
    def __init__(self, num_targets=1, map_name='empty', is_training=True,
                                        known_noise=True, im_size=28, **kwargs):
        TargetTrackingEnv8.__init__(self, num_targets=num_targets,
            map_name=map_name, is_training=is_training, known_noise=known_noise, im_size=im_size, **kwargs)
        self.id = 'TargetTracking-v8_SEQ'
        self.history_length = HISTORY_LENGTH
        self.limit['state'] = get_state_limit(self.limit['state'], num_targets)
        self.observation_space = spaces.Box(np.concatenate((
            -np.ones(5*im_size*im_size,), self.limit['state'][0])),
            np.concatenate((np.ones(5*im_size*im_size,), self.limit['state'][1])),
            dtype=np.float32)

    def reset(self, **kwargs):
        state = super().reset(**kwargs)
        im_state = state[:-len(self.state)]
        self.logdetcov_history =[Storage(max_capacity = self.history_length,
            init_value = np.log(LA.det(self.belief_targets[i].cov))) for i in range(self.num_targets)]
        new_state = self.add_history_to_state(self.state, NUM_TARGET_DEP_VARS, NUM_TARGET_INDEP_VARS, LOGDETCOV_IDX)
        return np.concatenate((im_state, new_state))

    def step(self, action):
        state, reward, done, info = super().step(action)
        im_state = state[:-len(self.state)]
        new_state = self.add_history_to_state(self.state, NUM_TARGET_DEP_VARS, NUM_TARGET_INDEP_VARS, LOGDETCOV_IDX)
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
