import gym
from gym import spaces, logger

import numpy as np
from numpy import linalg as LA
import os

from ttenv.maps import map_utils
from ttenv.metadata import METADATA
import ttenv.util as util

class TargetTrackingBase(gym.Env):
    def __init__(self, num_targets=1, map_name='empty',
                    is_training=True, known_noise=True, **kwargs):
        gym.Env.__init__(self)
        self.seed()
        self.state = None
        self.action_space = spaces.Discrete(len(METADATA['action_v']) * \
                                                    len(METADATA['action_w']))
        self.action_map = {}
        for (i,v) in enumerate(METADATA['action_v']):
            for (j,w) in enumerate(METADATA['action_w']):
                self.action_map[len(METADATA['action_w'])*i+j] = (v,w)
        assert(len(self.action_map.keys())==self.action_space.n)

        # Robot
        self.sampling_period = 0.5 # sec
        self.sensor_r_sd = METADATA['sensor_r_sd']
        self.sensor_b_sd = METADATA['sensor_b_sd']
        self.sensor_r = METADATA['sensor_r']
        self.fov = METADATA['fov']
        map_dir_path = '/'.join(map_utils.__file__.split('/')[:-1])
        self.MAP = map_utils.GridMap(
            map_path=os.path.join(map_dir_path, map_name),
            margin2wall = METADATA['margin2wall'])

        # Targets
        self.num_targets = num_targets
        self.viewer = None
        self.is_training = is_training
        self.target_init_vel = np.array(METADATA['target_init_vel'])
        self.target_speed_limit = METADATA['target_speed_limit']
        self.const_q = METADATA['const_q']
        self.const_q_true = METADATA['const_q_true']

        # initialization
        self.agent_init_pos =  np.array([self.MAP.origin[0], self.MAP.origin[1], 0.0])
        self.target_init_pos = np.array(self.MAP.origin)
        self.target_init_cov = METADATA['target_init_cov']
        self.reset_num = 0

    def reset(self, **kwargs):
        self.state = []
        self.num_collisions = 0
        init_pose = self.get_init_pose(**kwargs)
        self.reset_num += 1
        return init_pose

    def step(self, action):
        # The agent performs an action (t -> t+1)
        action_vw = self.action_map[action]
        is_col = self.agent.update(action_vw, [t.state[:2] for t in self.targets])
        self.num_collisions += int(is_col)

        # The targets move (t -> t+1) and are observed by the agent.
        observed = self.update_target_and_belief()
        reward, done, mean_nlogdetcov = self.get_reward(self.is_training,
                                                                is_col=is_col)
        obstacles_pt = self.MAP.get_closest_obstacle(self.agent.state)
        if obstacles_pt is None:
            obstacles_pt = (self.sensor_r, np.pi)

        self.update_state(observed, obstacles_pt, action_vw)
        return self.state, reward, done, {'mean_nlogdetcov': mean_nlogdetcov}

    def update_target_and_belief(self):
        observed = []
        for i in range(self.num_targets):
            # Update a target
            self.targets[i].update(self.agent.state[:2])
            # Observe
            observation = self.observation(self.targets[i])
            observed.append(observation[0])
            # If observed, update a belief.
            if observation[0]:
                self.belief_targets[i].update(observation[1], self.agent.state)
            # Predict the target for the next step.
            self.belief_targets[i].predict()
        return observed

    def get_init_pose(self, init_pose_list=[], target_path=[], **kwargs):
        """Generates initial positions for the agent, targets, and target beliefs.
        Parameters
        ---------
        init_pose_list : a list of dictionaries with pre-defined initial positions.
        lin_dist_range : a tuple of the minimum and maximum distance of a target
                        and a belief target from the agent.
        ang_dist_range_target : a tuple of the minimum and maximum angular
                            distance (counter clockwise) of a target from the
                            agent. -pi <= x <= pi
        ang_dist_range_belief : a tuple of the minimum and maximum angular
                            distance (counter clockwise) of a belief from the
                            agent. -pi <= x <= pi
        blocked : True if there is an obstacle between a target and the agent.
        """
        if init_pose_list != []:
            if target_path != []:
                self.set_target_path(target_path[self.reset_num])
            return init_pose_list[self.reset_num]
        else:
            return self.get_init_pose_random(**kwargs)

    def get_init_pose_random(self,
                            lin_dist_range_a2b=METADATA['lin_dist_range_a2b'],
                            ang_dist_range_a2b=METADATA['ang_dist_range_a2b'],
                            lin_dist_range_b2t=METADATA['lin_dist_range_b2t'],
                            ang_dist_range_b2t=METADATA['ang_dist_range_b2t'],
                            blocked=None,
                            **kwargs):
        if blocked is None and self.MAP.map is not None:
            if np.random.rand() < 0.5:
                blocked = True
            else:
                blocked = False

        is_agent_valid = False
        while(not is_agent_valid):
            init_pose = {}
            if self.MAP.map is None:
                blocked = False
                a_init = self.agent_init_pos[:2]
                is_agent_valid = True
            else:
                while(not is_agent_valid):
                    a_init = np.random.random((2,)) * (self.MAP.mapmax-self.MAP.mapmin) + self.MAP.mapmin
                    is_agent_valid = not(self.MAP.is_collision(a_init))

            init_pose['agent'] = [a_init[0], a_init[1], np.random.random() * 2 * np.pi - np.pi]
            init_pose['targets'], init_pose['belief_targets'] = [], []
            for i in range(self.num_targets):
                count, is_belief_valid = 0, False
                while(not is_belief_valid):
                    is_belief_valid, init_pose_belief = self.gen_rand_pose(
                        init_pose['agent'][:2], init_pose['agent'][2],
                        lin_dist_range_a2b[0], lin_dist_range_a2b[1],
                        ang_dist_range_a2b[0], ang_dist_range_a2b[1])
                    is_blocked = self.MAP.is_blocked(init_pose['agent'][:2], init_pose_belief[:2])
                    if is_belief_valid:
                        is_belief_valid = (blocked == is_blocked)
                    count += 1
                    if count > 100:
                        is_agent_valid = False
                        break
                init_pose['belief_targets'].append(init_pose_belief)

                count, is_target_valid, init_pose_target = 0, False, np.zeros((2,))
                while((not is_target_valid) and is_belief_valid):
                    is_target_valid, init_pose_target = self.gen_rand_pose(
                        init_pose['belief_targets'][i][:2],
                        init_pose['belief_targets'][i][2],
                        lin_dist_range_b2t[0], lin_dist_range_b2t[1],
                        ang_dist_range_b2t[0], ang_dist_range_b2t[1])
                    is_blocked = self.MAP.is_blocked(init_pose['agent'][:2], init_pose_target[:2])
                    if is_target_valid:
                        is_target_valid = (blocked == is_blocked)
                    count += 1
                    if count > 100:
                        is_agent_valid = False
                        break
                init_pose['targets'].append(init_pose_target)
        return init_pose

    def gen_rand_pose(self, frame_xy, frame_theta, min_lin_dist, max_lin_dist,
            min_ang_dist, max_ang_dist, additional_frame=None):
        """Genertes random position and yaw.
        Parameters
        --------
        frame_xy, frame_theta : xy and theta coordinate of the frame you want to compute a distance from.
        min_lin_dist : the minimum linear distance from o_xy to a sample point.
        max_lin_dist : the maximum linear distance from o_xy to a sample point.
        min_ang_dist : the minimum angular distance (counter clockwise direction) from c_theta to a sample point.
        max_ang_dist : the maximum angular distance (counter clockwise direction) from c_theta to a sample point.
        """
        if max_ang_dist < min_ang_dist:
            max_ang_dist += 2*np.pi
        rand_ang = util.wrap_around(np.random.rand() * \
                        (max_ang_dist - min_ang_dist) + min_ang_dist)

        rand_r = np.random.rand() * (max_lin_dist - min_lin_dist) + min_lin_dist
        rand_xy = np.array([rand_r*np.cos(rand_ang), rand_r*np.sin(rand_ang)])
        rand_xy_global = util.transform_2d_inv(rand_xy, frame_theta, np.array(frame_xy))
        if additional_frame:
            rand_xy_global = util.transform_2d_inv(rand_xy_global, additional_frame[2], np.array(additional_frame[:2]))
        is_valid = not(self.MAP.is_collision(rand_xy_global))
        return is_valid, [rand_xy_global[0], rand_xy_global[1], rand_ang + frame_theta]

    def add_history_to_state(self, state, num_target_dep_vars, num_target_indep_vars, logdetcov_idx):
        """
        Replacing the current logetcov value to a sequence of the recent few
        logdetcov values for each target.
        It uses fixed values for :
            1) the number of target dependent variables
            2) current logdetcov index at each target dependent vector
            3) the number of target independent variables
        """
        new_state = []
        for i in range(self.num_targets):
            self.logdetcov_history[i].add(state[num_target_dep_vars*i+logdetcov_idx])
            new_state = np.concatenate((new_state, state[num_target_dep_vars*i: num_target_dep_vars*i+logdetcov_idx]))
            new_state = np.concatenate((new_state, self.logdetcov_history[i].get_values()))
            new_state = np.concatenate((new_state, state[num_target_dep_vars*i+logdetcov_idx+1:num_target_dep_vars*(i+1)]))
        new_state = np.concatenate((new_state, state[-num_target_indep_vars:]))
        return new_state

    def set_target_path(self, target_path):
        targets = [Agent2DFixedPath(dim=self.target_dim, sampling_period=self.sampling_period,
                                limit=self.limit['target'],
                                collision_func=lambda x: self.MAP.is_collision(x),
                                path=target_path[i]) for i in range(self.num_targets)]
        self.targets = targets

    def observation(self, target):
        r, alpha = util.relative_distance_polar(target.state[:2],
                                            xy_base=self.agent.state[:2],
                                            theta_base=self.agent.state[2])
        observed = (r <= self.sensor_r) \
                    & (abs(alpha) <= self.fov/2/180*np.pi) \
                    & (not(self.MAP.is_blocked(self.agent.state, target.state)))
        z = None
        if observed:
            z = np.array([r, alpha])
            z += np.random.multivariate_normal(np.zeros(2,), self.observation_noise(z))
        return observed, z

    def observation_noise(self, z):
        obs_noise_cov = np.array([[self.sensor_r_sd * self.sensor_r_sd, 0.0],
                                [0.0, self.sensor_b_sd * self.sensor_b_sd]])
        return obs_noise_cov

    def get_reward(self, is_training=True, **kwargs):
        return reward_fun_1(self.belief_targets, is_training=is_training, **kwargs)

def reward_fun_0(belief_targets, obstacles_pt, observed, is_training=True,
        c_mean=0.1, c_std=0.1, c_observed=0.1, c_penalty=1.0):

    # Penalty when it is closed to an obstacle.
    if obstacles_pt is None:
        penalty = 0.0
    else:
        penalty =  METADATA['margin2wall']**2 * \
                        1./max(METADATA['margin2wall']**2, obstacles_pt[0]**2)

    detcov = [LA.det(b_target.cov) for b_target in belief_targets]
    r_detcov_mean = - np.mean(np.log(detcov))
    r_detcov_std = - np.std(np.log(detcov))
    r_observed = np.mean(observed)
    # reward = - c_penalty * penalty + c_mean * r_detcov_mean + \
    #              c_std * r_detcov_std + c_observed * r_observed
    if sum(observed) == 0:
        reward = - c_penalty * penalty + c_mean * r_detcov_mean + \
                     c_std * r_detcov_std
    else:
        reward = - c_penalty * penalty + c_mean * r_detcov_mean + \
                     c_std * r_detcov_std
        reward = max(0.0, reward) + c_observed * r_observed

    mean_nlogdetcov = None
    if not(is_training):
        logdetcov = [np.log(LA.det(b_target.cov)) for b_target in belief_targets]
        mean_nlogdetcov = -np.mean(logdetcov)
    return reward, False, mean_nlogdetcov

def reward_fun(belief_targets, obstacles_pt, is_training=True, c_mean=0.1):

    detcov = [LA.det(b_target.cov) for b_target in belief_targets]
    r_detcov_mean = - np.mean(np.log(detcov))
    reward = c_mean * r_detcov_mean

    mean_nlogdetcov = None
    if not(is_training):
        logdetcov = [np.log(LA.det(b_target.cov)) for b_target in belief_targets]
        mean_nlogdetcov = -np.mean(logdetcov)
    return reward, False, mean_nlogdetcov

def reward_fun_1(belief_targets, is_col, is_training=True, c_mean=0.1, c_penalty=1.0):
    detcov = [LA.det(b_target.cov) for b_target in belief_targets]
    r_detcov_mean = - np.mean(np.log(detcov))
    reward = c_mean * r_detcov_mean
    if is_col :
        reward = min(0.0, reward) - c_penalty * 1.0

    mean_nlogdetcov = None
    if not(is_training):
        logdetcov = [np.log(LA.det(b_target.cov)) for b_target in belief_targets]
        mean_nlogdetcov = -np.mean(logdetcov)
    return reward, False, mean_nlogdetcov
