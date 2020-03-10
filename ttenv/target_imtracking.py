"""Target Tracking Environments for Reinforcement Learning with an Image Input. OpenAI gym format
[Environment Descriptions]
TargetTrackingEnv5 : Local Image-based Double Integrator Target model with KF belief tracker
    RL state: [local_map_image, [d, alpha, ddot, alphadot, logdet(Sigma), observed] * nb_targets, [o_d, o_alpha]]
    Target : Double Integrator model, [x,y,xdot,ydot]
    Belief Target : KF, Double Integrator model

TargetTrackingEnv6 : Local visit frequency map is given to the agent as well as all the inputs of V5.
    The covered area is same as the area presented in the local map.

TargetTrackingEnv7 : Local map & Local visit frequency maps of outside the front range - left, right, front, back are given.
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

class TargetTrackingEnv5(TargetTrackingEnv1):
    def __init__(self, num_targets=1, map_name='empty', is_training=True,
                                        known_noise=True, im_size=28, **kwargs):
        self.im_size = im_size
        TargetTrackingEnv1.__init__(self, num_targets=num_targets,
            map_name=map_name, is_training=is_training, known_noise=known_noise, **kwargs)
        self.id = 'TargetTracking-v5'
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

class TargetTrackingEnv6(TargetTrackingEnv5):
    def __init__(self, num_targets=1, map_name='empty', is_training=True,
                                        known_noise=True, im_size=50, **kwargs):
        TargetTrackingEnv5.__init__(self, num_targets=num_targets,
            map_name=map_name, is_training=is_training, known_noise=known_noise,
            im_size=im_size, **kwargs)
        self.id = 'TargetTracking-v6'

    def reset(self, **kwargs):
        self.MAP.reset_visit_freq_map()
        return super().reset(**kwargs)

    def map_state_func(self):
        # Update the visit frequency map.
        b_speed = np.mean([np.sqrt(np.sum(self.belief_targets[i].state[2:]**2)) for i in range(self.num_targets)])
        decay_factor = np.exp(self.sampling_period*b_speed/self.sensor_r*np.log(0.7))
        self.MAP.update_visit_freq_map(self.agent.state, decay_factor)

        _, self.local_mapmin_g, local_visit_map = self.MAP.local_visit_map(
                                                    self.im_size, self.agent.state)
        # normalize the maps
        self.local_map = [local_visit_map - 1.0]
        self.local_mapmin_g = [self.local_mapmin_g]
        return self.local_map[0].flatten()

class TargetTrackingEnv7(TargetTrackingEnv5):
    def __init__(self, num_targets=1, map_name='empty', is_training=True,
                                        known_noise=True, im_size=28, **kwargs):
        TargetTrackingEnv5.__init__(self, num_targets=num_targets,
            map_name=map_name, is_training=is_training, known_noise=known_noise,
            im_size=im_size, **kwargs)
        self.id = 'TargetTracking-v7'

    def set_limits(self, target_speed_limit=None):
        self.num_target_dep_vars = 7
        self.num_target_indep_vars = 2

        if target_speed_limit is None:
            self.target_speed_limit = np.random.choice([1.0, 3.0])
        else:
            self.target_speed_limit = target_speed_limit
        rel_speed_limit = self.target_speed_limit + METADATA['action_v'][0] # Maximum relative speed

        self.limit = {} # 0: low, 1:highs
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin,[-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [np.concatenate((self.MAP.mapmin,[-self.target_speed_limit, -self.target_speed_limit])),
                                np.concatenate((self.MAP.mapmax, [self.target_speed_limit, self.target_speed_limit]))]
        self.limit['state'] = [np.concatenate(([0.0, -np.pi, -rel_speed_limit, -10*np.pi, -50.0, 0.0, 0.0]*self.num_targets, [0.0, -np.pi])),
                               np.concatenate(([600.0, np.pi, rel_speed_limit, 10*np.pi,  50.0, 2.0, 2.0]*self.num_targets, [self.sensor_r, np.pi]))]
        self.observation_space = spaces.Box(
                                    np.concatenate((-np.ones(5*self.im_size*self.im_size,), self.limit['state'][0])),
                                    np.concatenate((np.ones(5*self.im_size*self.im_size,), self.limit['state'][1])),
                                    dtype=np.float32)

        assert(len(self.limit['state'][0]) == (self.num_target_dep_vars * self.num_targets + self.num_target_indep_vars))

    def reset(self, **kwargs):
        self.MAP.reset_visit_freq_map()
        return super().reset(**kwargs)

    def state_func(self, action_vw, observed):
        # Find the closest obstacle coordinate.
        obstacles_pt = self.MAP.get_closest_obstacle(self.agent.state)
        if obstacles_pt is None:
            obstacles_pt = (self.sensor_r, np.pi)

        self.state = []
        for i in range(self.num_targets):
            r_b, alpha_b = util.relative_distance_polar(self.belief_targets[i].state[:2],
                                                xy_base=self.agent.state[:2],
                                                theta_base=self.agent.state[2])
            r_dot_b, alpha_dot_b = util.relative_velocity_polar(
                                    self.belief_targets[i].state[:2],
                                    self.belief_targets[i].state[2:],
                                    self.agent.state[:2], self.agent.state[2],
                                    action_vw[0], action_vw[1])
            is_belief_blocked = self.MAP.is_blocked(self.agent.state[:2], self.belief_targets[i].state[:2])
            self.state.extend([r_b, alpha_b, r_dot_b, alpha_dot_b,
                                np.log(LA.det(self.belief_targets[i].cov)),
                                float(observed[i]), float(is_belief_blocked)])
        self.state.extend([obstacles_pt[0], obstacles_pt[1]])
        self.state = np.array(self.state)

        # Update the visit frequency map.
        b_speed = np.mean([np.sqrt(np.sum(self.belief_targets[i].state[2:]**2)) for i in range(self.num_targets)])
        decay_factor = np.exp(self.sampling_period*b_speed/self.sensor_r*np.log(0.7))
        self.MAP.update_visit_freq_map(self.agent.state, decay_factor, observed=bool(np.mean(observed)))

    def map_state_func(self):
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

class TargetTrackingEnv8(TargetTrackingEnv5):
    def __init__(self, num_targets=1, map_name='empty', is_training=True,
                                        known_noise=True, im_size=28, **kwargs):
        TargetTrackingEnv5.__init__(self, num_targets=num_targets,
            map_name=map_name, is_training=is_training, known_noise=known_noise, im_size=im_size, **kwargs)
        self.id = 'TargetTracking-v8'

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

class TargetTrackingEnv9(TargetTrackingEnv7):
    def __init__(self, num_targets=1, map_name='empty', is_training=True,
                                        known_noise=True, im_size=28, **kwargs):
        TargetTrackingEnv7.__init__(self, num_targets=num_targets,
            map_name=map_name, is_training=is_training, known_noise=known_noise,
            im_size=im_size, **kwargs)
        self.id = 'TargetTracking-v9'
        self.set_limits(METADATA['target_speed_limit'])

    def set_limits(self, target_speed_limit=None):
        self.num_target_dep_vars = 7
        self.num_target_indep_vars = 3

        if target_speed_limit is None:
            self.target_speed_limit = np.random.choice([1.0, 3.0])
        else:
            self.target_speed_limit = target_speed_limit
        rel_speed_limit = self.target_speed_limit + METADATA['action_v'][0] # Maximum relative speed

        self.limit = {} # 0: low, 1:highs
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin,[-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [np.concatenate((self.MAP.mapmin,[-self.target_speed_limit, -self.target_speed_limit])),
                                np.concatenate((self.MAP.mapmax, [self.target_speed_limit, self.target_speed_limit]))]
        self.limit['state'] = [np.concatenate(([0.0, -np.pi, -rel_speed_limit, -10*np.pi, -50.0, 0.0, 0.0]*self.num_targets, [0.0, -np.pi, 0.0])),
                               np.concatenate(([600.0, np.pi, rel_speed_limit, 10*np.pi,  50.0, 2.0, 2.0]*self.num_targets, [self.sensor_r, np.pi, self.sensor_r]))]
        self.observation_space = spaces.Box(
                                    np.concatenate((-np.ones(5*self.im_size*self.im_size,), self.limit['state'][0])),
                                    np.concatenate((np.ones(5*self.im_size*self.im_size,), self.limit['state'][1])),
                                    dtype=np.float32)
        assert(len(self.limit['state'][0]) == (self.num_target_dep_vars * self.num_targets + self.num_target_indep_vars))

    def state_func(self, action_vw, observed):
        # Find the closest obstacle coordinate.
        obstacles_pt, front_obstacle_r = self.MAP.get_closest_obstacle_v2(self.agent.state)
        if obstacles_pt is None:
            obstacles_pt = (self.sensor_r, np.pi)
        if front_obstacle_r is None:
            front_obstacle_r = self.sensor_r

        self.state = []
        for i in range(self.num_targets):
            r_b, alpha_b = util.relative_distance_polar(self.belief_targets[i].state[:2],
                                                xy_base=self.agent.state[:2],
                                                theta_base=self.agent.state[2])
            r_dot_b, alpha_dot_b = util.relative_velocity_polar(
                                    self.belief_targets[i].state[:2],
                                    self.belief_targets[i].state[2:],
                                    self.agent.state[:2], self.agent.state[2],
                                    action_vw[0], action_vw[1])
            is_belief_blocked = self.MAP.is_blocked(self.agent.state[:2], self.belief_targets[i].state[:2])
            self.state.extend([r_b, alpha_b, r_dot_b, alpha_dot_b,
                                np.log(LA.det(self.belief_targets[i].cov)),
                                float(observed[i]), float(is_belief_blocked)])
        self.state.extend([obstacles_pt[0], obstacles_pt[1], front_obstacle_r])
        self.state = np.array(self.state)

        # Update the visit frequency map.
        b_speed = np.mean([np.sqrt(np.sum(self.belief_targets[i].state[2:]**2)) for i in range(self.num_targets)])
        decay_factor = np.exp(self.sampling_period*b_speed/self.sensor_r*np.log(0.7))
        self.MAP.update_visit_freq_map(self.agent.state, decay_factor, observed=bool(np.mean(observed)))

class TargetTrackingEnv10(TargetTrackingEnv5):
    def __init__(self, num_targets=1, map_name='empty', is_training=True,
                                        known_noise=True, im_size=50, **kwargs):
        TargetTrackingEnv5.__init__(self, num_targets=num_targets,
            map_name=map_name, is_training=is_training, known_noise=known_noise,
            im_size=im_size, **kwargs)
        self.id = 'TargetTracking-v10'

    def reset(self, **kwargs):
        self.MAP.reset_visit_freq_map()
        return super().reset(**kwargs)

    def set_limits(self, target_speed_limit=None):
        super().set_limits(target_speed_limit)
        self.observation_space = spaces.Box(
            np.concatenate((-np.ones(2*self.im_size*self.im_size,), self.limit['state'][0])),
            np.concatenate((np.ones(2*self.im_size*self.im_size,), self.limit['state'][1])),
            dtype=np.float32)

    def map_state_func(self):
        # Update the visit frequency map.
        b_speed = np.mean([np.sqrt(np.sum(self.belief_targets[i].state[2:]**2)) for i in range(self.num_targets)])
        decay_factor = np.exp(self.sampling_period*b_speed/self.sensor_r*np.log(0.7))
        self.MAP.update_visit_freq_map_full(self.agent.state, decay_factor)

        self.local_map, self.local_mapmin_g, local_visit_map = self.MAP.local_map_seperate(
                                                self.im_size, self.agent.state)
        # normalize the maps
        self.local_map = [(self.local_map - 0.5) * 2]
        self.local_map.append((local_visit_map - 0.5) * 2)

        self.local_mapmin_g = [self.local_mapmin_g, self.local_mapmin_g]

        return np.array(self.local_map).T.flatten()
