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

TargetTrackingEnv0 : Static Target model + noise - No Velocity Estimate
    RL state: [d, alpha, logdet(Sigma), observed] * nb_targets , [o_d, o_alpha]
    Target: Static [x,y] + noise
    Belief Target: KF, Estimate only x and y

TargetTrackingEnv1 : Double Integrator Target model with KF belief tracker
    RL state: [d, alpha, ddot, alphadot, logdet(Sigma), observed] * nb_targets, [o_d, o_alpha]
    Target : Double Integrator model, [x,y,xdot,ydot]
    Belief Target : KF, Double Integrator model

TargetTrackingEnv2 : Predefined target paths with KF belief tracker
    RL state: [d, alpha, ddot, alphadot, logdet(Sigma), observed] * nb_targets, [o_d, o_alpha]
    Target : Pre-defined target paths - input files required
    Belief Target : KF, Double Integrator model

TargetTrackingEnv3 : SE2 Target model with UKF belief tracker
    RL state: [d, alpha, logdet(Sigma), observed] * nb_targets, [o_d, o_alpha]
    Target : SE2 model [x,y,theta] + a control policy u=[v,w]
    Belief Target : UKF for SE2 model [x,y,theta]

TargetTrackingEnv4 : SE2 Target model with UKF belief tracker [x,y,theta,v,w]
    RL state: [d, alpha, ddot, alphadot, logdet(Sigma), observed] * nb_targets, [o_d, o_alpha]
    Target : SE2 model [x,y,theta] + a control policy u=[v,w]
    Belief Target : UKF for SE2Vel model [x,y,theta,v,w]
"""
import gym
from gym import spaces, logger
from gym.utils import seeding

import numpy as np
from numpy import linalg as LA
import os, copy

from ttenv.maps import map_utils
from ttenv.agent_models import *
from ttenv.policies import *
from ttenv.belief_tracker import KFbelief, UKFbelief
from ttenv.metadata import METADATA
import ttenv.util as util

class TargetTrackingEnv0(gym.Env):
    def __init__(self, num_targets=1, map_name='empty',
                    is_training=True, known_noise=True, **kwargs):
        gym.Env.__init__(self)
        self.seed()
        self.id = 'TargetTracking-v0'
        self.state = None
        self.action_space = spaces.Discrete(len(METADATA['action_v']) * \
                                                    len(METADATA['action_w']))
        self.action_map = {}
        for (i,v) in enumerate(METADATA['action_v']):
            for (j,w) in enumerate(METADATA['action_w']):
                self.action_map[len(METADATA['action_w'])*i+j] = (v,w)
        assert(len(self.action_map.keys())==self.action_space.n)

        self.target_dim = 2
        self.num_targets = num_targets
        self.viewer = None
        self.is_training = is_training

        self.sampling_period = 0.5 # sec
        self.sensor_r_sd = METADATA['sensor_r_sd']
        self.sensor_b_sd = METADATA['sensor_b_sd']
        self.sensor_r = METADATA['sensor_r']
        self.fov = METADATA['fov']
        map_dir_path = '/'.join(map_utils.__file__.split('/')[:-1])
        self.MAP = map_utils.GridMap(
            map_path=os.path.join(map_dir_path, map_name),
            r_max = self.sensor_r, fov = self.fov/180.0*np.pi,
            margin2wall = METADATA['margin2wall'])
        # LIMITS
        self.limit = {} # 0: low, 1:high
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin,[-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [self.MAP.mapmin, self.MAP.mapmax]
        self.limit['state'] = [np.concatenate(([0.0, -np.pi, -50.0, 0.0]*num_targets, [0.0, -np.pi ])),
                               np.concatenate(([600.0, np.pi, 50.0, 2.0]*num_targets, [self.sensor_r, np.pi]))]
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)

        self.agent_init_pos =  np.array([self.MAP.origin[0], self.MAP.origin[1], 0.0])
        self.target_init_pos = np.array(self.MAP.origin)
        self.target_init_cov = METADATA['target_init_cov']
        self.target_noise_cov = METADATA['const_q'] * self.sampling_period**3 / 3 * np.eye(self.target_dim)
        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = METADATA['const_q_true'] * np.eye(2)
        self.targetA = np.eye(self.target_dim)
        # Build a robot
        self.agent = AgentSE2(dim=3, sampling_period=self.sampling_period, limit=self.limit['agent'],
                            collision_func=lambda x: map_utils.is_collision(self.MAP, x))
        # Build a target
        self.targets = [AgentDoubleInt2D(dim=self.target_dim, sampling_period=self.sampling_period,
                            limit=self.limit['target'],
                            collision_func=lambda x: map_utils.is_collision(self.MAP, x),
                            A=self.targetA, W=self.target_true_noise_sd) for _ in range(num_targets)]
        self.belief_targets = [KFbelief(dim=self.target_dim, limit=self.limit['target'], A=self.targetA,
                            W=self.target_noise_cov, obs_noise_func=self.observation_noise,
                            collision_func=lambda x: map_utils.is_collision(self.MAP, x))
                                for _ in range(num_targets)]
        self.reset_num = 0

    def get_init_pose(self, init_random=True, **kwargs):
        if init_random:
            return self.get_init_pose_random(
                                init_distance_min=METADATA['init_distance_min'],
                                init_distance_max=METADATA['init_distance_max'],
                                **kwargs)
        else:
            return self.get_init_pose_eval(**kwargs)

    def gen_rand_pose(self, o_xy, c_theta, ang_range, min_dist, max_dist, opposite=False):
        """Genertes random position and yaw.
        Parameters
        --------
        o_xy : xy position of a point in the global frame which we compute a distance from.
        c_theta : angular position of a point in the global frame which we compute an angular distance from.
        ang_range : a range for sampling an angle around c_theta.
        min_dist : the minimum distance from o_xy to a sample point.
        max_dist : the maximum distance from o_xy to a sample point.
        opposite : True if you sample a point in the opposite side (np.pi) of the c_theta direction.
        """
        rand_ang = (np.random.rand() - 0.5) * ang_range
        if ang_range == 2*np.pi:
            t_rand_ang = rand_ang
        if opposite:
            t_rand_ang = util.wrap_around(rand_ang + c_theta + np.pi)
        else:
            t_rand_ang = util.wrap_around(rand_ang + c_theta)

        t_r = np.random.rand() * (max_dist - min_dist) + min_dist
        t_init = np.array([t_r*np.cos(t_rand_ang), t_r*np.sin(t_rand_ang)]) + o_xy
        is_target_valid = not(map_utils.is_collision(self.MAP, t_init))
        return is_target_valid, [t_init[0], t_init[1], t_rand_ang]

    def get_init_pose_random(self, init_distance_min=0.0, init_distance_max=10.0, **kwargs):
        is_agent_valid = False
        while(not is_agent_valid):
            init_pose = {}
            if self.MAP.map is None:
                a_init = self.agent_init_pos[:2]
                is_agent_valid = True
            else:
                is_agent_valid = False
                while(not is_agent_valid):
                    a_init = np.random.random((2,)) * (self.MAP.mapmax-self.MAP.mapmin) + self.MAP.mapmin
                    is_agent_valid = not(map_utils.is_collision(self.MAP, a_init))

            init_pose['agent'] = [a_init[0], a_init[1], np.random.random() * 2 * np.pi - np.pi]
            init_pose['targets'], init_pose['belief_targets'] = [], []
            for i in range(self.num_targets):
                count, is_target_valid = 0, False
                while(not is_target_valid):
                    is_target_valid, init_pose_target = self.gen_rand_pose(
                            init_pose['agent'][:2], init_pose['agent'][2],
                            2*np.pi, init_distance_min, init_distance_max)
                    count += 1
                    if count > 100:
                        is_agent_valid = False
                        break
                init_pose['targets'].append(init_pose_target)

                count, is_belief_valid, init_pose_belief = 0, False, np.zeros((2,))
                while(not is_belief_valid):
                    is_belief_valid, init_pose_belief = self.gen_rand_pose(
                            init_pose['agent'][:2], init_pose['agent'][2],
                            2*np.pi, init_distance_min, init_distance_max)
                    count += 1
                    if count > 100:
                        is_agent_valid = False
                        break
                init_pose['belief_targets'].append(init_pose_belief)
        return init_pose

    def get_init_pose_eval(self, init_pose_list=[], init_distance_min=5.0,
                    init_distance_max=10.0, target_direction=True,
                    belief_direction=True, blocked=False, **kwargs):
        """Generates initial positions for the agent, targets, and target beliefs.
        Parameters
        ---------
        init_pose_list : a list of dictionaries with pre-defined initial positions.
        init_distance_min : the minimum distance between a target and the agent.
        init_distance_max : the maximum distance between a target and the agent.
        target_direction : True if a target and the agent are on the same side (within pi)
        belief_direction : True if a target and its corresponding belief are on the same side (within pi)
        """
        if init_pose_list:
            self.reset_num += 1
            return init_pose_list[self.reset_num-1]
        else:
            is_agent_valid = False
            while(not is_agent_valid):
                init_pose = {}
                if self.MAP.map is None:
                    if blocked:
                        raise ValueError('Unable to find a blocked initial condition. There is no obstacle in this map.')
                    a_init = self.agent_init_pos[:2]
                    is_agent_valid = True
                else:
                    while(not is_agent_valid):
                        a_init = np.random.random((2,)) * (self.MAP.mapmax-self.MAP.mapmin) + self.MAP.mapmin
                        is_agent_valid = not(map_utils.is_collision(self.MAP, a_init))

                init_pose['agent'] = [a_init[0], a_init[1], np.random.random() * 2 * np.pi - np.pi]
                init_pose['targets'], init_pose['belief_targets'] = [], []
                for i in range(self.num_targets):
                    count, is_target_valid = 0, False
                    while(not is_target_valid):
                        is_target_valid, init_pose_target = self.gen_rand_pose(
                            init_pose['agent'][:2], init_pose['agent'][2],
                            np.pi, init_distance_min,
                            init_distance_max, not(target_direction))
                        if blocked and is_target_valid:
                            is_target_valid = map_utils.is_blocked(self.MAP, init_pose['agent'][:2], init_pose_target[:2])
                        count += 1
                        if count > 100:
                            is_agent_valid = False
                            break
                    init_pose['targets'].append(init_pose_target)

                    count, is_belief_valid, init_pose_belief = 0, False, np.zeros((2,))
                    while((not is_belief_valid) and is_target_valid):
                        is_belief_valid, init_pose_belief = self.gen_rand_pose(
                            init_pose['agent'][:2], init_pose['targets'][i][2],
                            0.5*np.pi, init_distance_min,
                            init_distance_max, not(belief_direction))
                        count += 1
                        if count > 100:
                            is_agent_valid = False
                            break
                    init_pose['belief_targets'].append(init_pose_belief)
        return init_pose

    def reset(self, init_random=True, **kwargs):
        self.state = []
        init_pose = self.get_init_pose(init_random=init_random, **kwargs)
        self.agent.reset(init_pose['agent'])
        for i in range(self.num_targets):
            self.belief_targets[i].reset(
                        init_state=init_pose['belief_targets'][i][:self.target_dim],
                        init_cov=self.target_init_cov)
            self.targets[i].reset(init_pose['targets'][i][:self.target_dim])
            r, alpha, _ = util.xyg2polarb(self.belief_targets[i].state[:2],
                                 self.agent.state[:2], self.agent.state[2])
            logdetcov = np.log(LA.det(self.belief_targets[i].cov))
            self.state.extend([r, alpha, logdetcov, 0.0])

        self.state.extend([self.sensor_r, np.pi])
        self.state = np.array(self.state)
        return self.state

    def observation(self, target):
        r, alpha, _ = util.xyg2polarb(target.state[:2],
                             self.agent.state[:2], self.agent.state[2])
        observed = (r <= self.sensor_r) \
                    & (abs(alpha) <= self.fov/2/180*np.pi) \
                    & (not(map_utils.is_blocked(self.MAP, self.agent.state, target.state)))
        z = None
        if observed:
            z = np.array([r, alpha])
            z += np.random.multivariate_normal(np.zeros(2,), self.observation_noise(z))
        return observed, z

    def observation_noise(self, z):
        obs_noise_cov = np.array([[self.sensor_r_sd * self.sensor_r_sd, 0.0],
                                [0.0, self.sensor_b_sd * self.sensor_b_sd]])
        return obs_noise_cov

    def get_reward(self, obstacles_pt, observed, is_training=True):
        return reward_fun(self.belief_targets, obstacles_pt, observed, is_training)

    def step(self, action):
        action_vw = self.action_map[action]
        _ = self.agent.update(action_vw, [t.state[:2] for t in self.targets])
        obstacles_pt = map_utils.get_cloest_obstacle(self.MAP, self.agent.state)
        observed = []
        for i in range(self.num_targets):
            self.targets[i].update(self.agent.state[:2])
            # Observe
            obs = self.observation(self.targets[i])
            observed.append(obs[0])
            self.belief_targets[i].predict() # Belief state at t+1
            if obs[0]: # if observed, update the target belief.
                self.belief_targets[i].update(obs[1], self.agent.state)

        reward, done, mean_nlogdetcov = self.get_reward(obstacles_pt, observed, self.is_training)
        self.state = []
        if obstacles_pt is None:
            obstacles_pt = (self.sensor_r, np.pi)
        for i in range(self.num_targets):
            r_b, alpha_b, _ = util.xyg2polarb(self.belief_targets[i].state[:2],
                                 self.agent.state[:2], self.agent.state[2])
            self.state.extend([r_b, alpha_b,
                                    np.log(LA.det(self.belief_targets[i].cov)), float(observed[i])])
        self.state.extend([obstacles_pt[0], obstacles_pt[1]])
        self.state = np.array(self.state)
        return self.state, reward, done, {'mean_nlogdetcov': mean_nlogdetcov}

class TargetTrackingEnv1(TargetTrackingEnv0):
    def __init__(self, num_targets=1, map_name='empty', is_training=True, known_noise=True, **kwargs):
        TargetTrackingEnv0.__init__(self, num_targets=num_targets, map_name=map_name,
            is_training=is_training, known_noise=known_noise, **kwargs)
        self.id = 'TargetTracking-v1'
        self.target_dim = 4
        self.target_init_vel = METADATA['target_init_vel'] * np.ones((2,))

        # LIMIT
        self.limit = {} # 0: low, 1:highs
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin,[-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [np.concatenate((self.MAP.mapmin,[-METADATA['target_vel_limit'], -METADATA['target_vel_limit']])),
                                np.concatenate((self.MAP.mapmax, [METADATA['target_vel_limit'], METADATA['target_vel_limit']]))]
        rel_vel_limit = METADATA['target_vel_limit'] + METADATA['action_v'][0] # Maximum relative speed
        self.limit['state'] = [np.concatenate(([0.0, -np.pi, -rel_vel_limit, -10*np.pi, -50.0, 0.0]*num_targets, [0.0, -np.pi])),
                               np.concatenate(([600.0, np.pi, rel_vel_limit, 10*np.pi,  50.0, 2.0]*num_targets, [self.sensor_r, np.pi]))]
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)
        self.targetA = np.concatenate((np.concatenate((np.eye(2), self.sampling_period*np.eye(2)), axis=1),
                                        [[0,0,1,0],[0,0,0,1]]))
        self.target_noise_cov = METADATA['const_q'] * np.concatenate((
                            np.concatenate((self.sampling_period**3/3*np.eye(2), self.sampling_period**2/2*np.eye(2)), axis=1),
                        np.concatenate((self.sampling_period**2/2*np.eye(2), self.sampling_period*np.eye(2)),axis=1) ))
        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = METADATA['const_q_true'] * np.concatenate((
                        np.concatenate((self.sampling_period**2/2*np.eye(2), self.sampling_period/2*np.eye(2)), axis=1),
                        np.concatenate((self.sampling_period/2*np.eye(2), self.sampling_period*np.eye(2)),axis=1) ))
        # Build a robot
        self.agent = AgentSE2(3, self.sampling_period, self.limit['agent'],
                            lambda x: map_utils.is_collision(self.MAP, x))
        # Build a target
        self.targets = [AgentDoubleInt2D(self.target_dim, self.sampling_period, self.limit['target'],
                            lambda x: map_utils.is_collision(self.MAP, x),
                            W=self.target_true_noise_sd, A=self.targetA) for _ in range(num_targets)]
        self.belief_targets = [KFbelief(dim=self.target_dim, limit=self.limit['target'], A=self.targetA,
                            W=self.target_noise_cov, obs_noise_func=self.observation_noise,
                            collision_func=lambda x: map_utils.is_collision(self.MAP, x))
                            for _ in range(num_targets)]

    def reset(self, init_random=True, **kwargs):
        self.state = []
        init_pose = self.get_init_pose(init_random=init_random, **kwargs)
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
        return self.state

    def step(self, action):
        # The agent performs an action (t -> t+1)
        action_vw = self.action_map[action]
        _ = self.agent.update(action_vw, [t.state[:2] for t in self.targets])

        # The targets move (t -> t+1) and are observed by the agent.
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
        reward, done, mean_nlogdetcov = self.get_reward(obstacles_pt, observed,
                                                            self.is_training)
        self.state = []
        if obstacles_pt is None:
            obstacles_pt = (self.sensor_r, np.pi)
        for i in range(self.num_targets):
            r_b, alpha_b, _ = util.xyg2polarb(self.belief_targets[i].state[:2],
                                    self.agent.state[:2], self.agent.state[-1])
            r_dot_b, alpha_dot_b = util.xyg2polarb_dot_2(
                                    self.belief_targets[i].state[:2],
                                    self.belief_targets[i].state[2:],
                                    self.agent.state[:2], self.agent.state[-1],
                                    action_vw[0], action_vw[1])
            self.state.extend([r_b, alpha_b, r_dot_b, alpha_dot_b,
                                    np.log(LA.det(self.belief_targets[i].cov)),
                                    float(observed[i])])
        self.state.extend([obstacles_pt[0], obstacles_pt[1]])
        self.state = np.array(self.state)
        return self.state, reward, done, {'mean_nlogdetcov': mean_nlogdetcov}

class TargetTrackingEnv2(TargetTrackingEnv1):
    def __init__(self, num_targets=1, map_name='empty', is_training=True,
                known_noise=True, target_path_dir=None, **kwargs):
        """
        A predefined path for each target must be provided under the target_path_dir.
        Each path_i file for i=target_num is a T by 4 matrix where T is the
        number of time steps in a trajectory (or per episode). Each row consists
        of (x, y, xdot, ydot).
        """
        if target_path_dir is None:
            raise ValueError('No path directory for targets is provided.')
        TargetTrackingEnv1.__init__(self, num_targets=num_targets,
            map_name=map_name, is_training=is_training, known_noise=known_noise, **kwargs)
        self.id = 'TargetTracking-v2'
        self.targets = [Agent2DFixedPath(dim=self.target_dim, sampling_period=self.sampling_period,
                                limit=self.limit['target'],
                                collision_func=lambda x: map_utils.is_collision(self.MAP, x),
                                path=np.load(os.path.join(target_path_dir, "path_%d.npy"%(i+1)))) for i in range(self.num_targets)]
    def reset(self, init_random=True, **kwargs):
        self.state = []
        if init_random:
            if self.MAP.map is None:
                a_init = self.agent_init_pos[:2]
                self.agent.reset(self.agent_init_pos)
            else:
                isvalid = False
                while(not isvalid):
                    a_init = np.random.random((2,)) * (self.MAP.mapmax-self.MAP.mapmin) + self.MAP.mapmin
                    isvalid = not(map_utils.is_collision(self.MAP, a_init))
                self.agent.reset([a_init[0], a_init[1], np.random.random()*2*np.pi-np.pi])
            for i in range(self.num_targets):
                t_init = np.load("path_sh_%d.npy"%(i+1))[0][:2]
                self.belief_targets[i].reset(init_state=np.concatenate((t_init + METADATA['init_distance_belief'] * (np.random.rand(2)-0.5), np.zeros(2))), init_cov=self.target_init_cov)
                self.targets[i].reset(np.concatenate((t_init, self.target_init_vel)))
                r, alpha, _ = util.xyg2polarb(self.belief_targets[i].state[:2],
                                     self.agent.state[:2], self.agent.state[2])
                logdetcov = np.log(LA.det(self.belief_targets[i].cov))
                self.state.extend([r, alpha, 0.0, 0.0, logdetcov, 0.0])
        self.state.extend([self.sensor_r, np.pi])
        self.state = np.array(self.state)
        return self.state


class TargetTrackingEnv3(TargetTrackingEnv0):
    def __init__(self, num_targets=1, map_name='empty', is_training=True, known_noise=True, **kwargs):
        TargetTrackingEnv0.__init__(self, num_targets=num_targets,
            map_name=map_name, is_training=is_training, known_noise=known_noise, **kwargs)
        self.id = 'TargetTracking-v3'
        self.target_dim = 3

        # LIMIT
        self.limit = {} # 0: low, 1:highs
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin,[-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [np.concatenate((self.MAP.mapmin, [-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['state'] = [np.concatenate(([0.0, -np.pi, -50.0, 0.0]*num_targets, [0.0, -np.pi ])),
                               np.concatenate(([600.0, np.pi, 50.0, 2.0]*num_targets, [self.sensor_r, np.pi]))]
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)
        self.target_noise_cov = METADATA['const_q'] * self.sampling_period * np.eye(self.target_dim)
        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = METADATA['const_q_true'] * \
                                self.sampling_period * np.eye(self.target_dim)
        # Build a robot
        self.agent = AgentSE2(3, self.sampling_period, self.limit['agent'],
                            lambda x: map_utils.is_collision(self.MAP, x))
        # Build a target
        self.targets = [AgentSE2(self.target_dim, self.sampling_period, self.limit['target'],
                            lambda x: map_utils.is_collision(self.MAP, x),
                            policy=SinePolicy(0.1, 0.5, 5.0, self.sampling_period)) for _ in range(num_targets)]
        # SinePolicy(0.5, 0.5, 2.0, self.sampling_period)
        # CirclePolicy(self.sampling_period, self.MAP.origin, 3.0)
        # RandomPolicy()

        self.belief_targets = [UKFbelief(dim=self.target_dim, limit=self.limit['target'], fx=SE2Dynamics,
                            W=self.target_noise_cov, obs_noise_func=self.observation_noise,
                            collision_func=lambda x: map_utils.is_collision(self.MAP, x))
                            for _ in range(num_targets)]

    def step(self, action):
        action_vw = self.action_map[action]
        boundary_penalty = self.agent.update(action_vw, [t.state[:2] for t in self.targets])
        obstacles_pt = map_utils.get_cloest_obstacle(self.MAP, self.agent.state)
        observed = []
        for i in range(self.num_targets):
            self.targets[i].update()

            # Observe
            obs = self.observation(self.targets[i])
            observed.append(obs[0])
            # Update the belief of the agent on the target using UKF
            self.belief_targets[i].update(obs[0], obs[1], self.agent.state,
                                        np.array([np.random.random(),
                                        np.pi*np.random.random()-0.5*np.pi]))

        reward, done, mean_nlogdetcov = self.get_reward(obstacles_pt, observed, self.is_training)
        self.state = []
        if obstacles_pt is None:
            obstacles_pt = (self.sensor_r, np.pi)
        for i in range(self.num_targets):
            r_b, alpha_b, _ = util.xyg2polarb(self.belief_targets[i].state[:2],
                                 self.agent.state[:2], self.agent.state[2])
            self.state.extend([r_b, alpha_b,
                                np.log(LA.det(self.belief_targets[i].cov)), float(observed[i])])
        self.state.extend([obstacles_pt[0], obstacles_pt[1]])
        self.state = np.array(self.state)
        return self.state, reward, done, {'mean_nlogdetcov': mean_nlogdetcov}

class TargetTrackingEnv4(TargetTrackingEnv0):
    def __init__(self, num_targets=1, map_name='empty', is_training=True, known_noise=True, **kwargs):
        TargetTrackingEnv0.__init__(self, num_targets=num_targets,
            map_name=map_name, is_training=is_training, known_noise=known_noise, **kwargs)
        self.id = 'TargetTracking-v4'
        self.target_dim = 5
        self.target_init_vel = METADATA['target_init_vel'] * np.ones((2,))

        # LIMIT
        self.limit = {} # 0: low, 1:highs
        rel_vel_limit = METADATA['target_vel_limit'] + METADATA['action_v'][0] # Maximum relative speed
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin,[-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [np.concatenate((self.MAP.mapmin, [-np.pi, -METADATA['target_vel_limit'], -np.pi])),
                                            np.concatenate((self.MAP.mapmax, [np.pi, METADATA['target_vel_limit'], np.pi]))]
        self.limit['state'] = [np.concatenate(([0.0, -np.pi, -rel_vel_limit, -10*np.pi, -50.0, 0.0]*num_targets, [0.0, -np.pi ])),
                               np.concatenate(([600.0, np.pi, rel_vel_limit, 10*np.pi, 50.0, 2.0]*num_targets, [self.sensor_r, np.pi]))]
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)
        self.target_noise_cov = np.zeros((self.target_dim, self.target_dim))
        for i in range(3):
            self.target_noise_cov[i,i] = METADATA['const_q'] * self.sampling_period**3/3
        self.target_noise_cov[3:, 3:] = METADATA['const_q'] * \
                    np.array([[self.sampling_period, self.sampling_period**2/2],
                             [self.sampling_period**2/2, self.sampling_period]])
        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = METADATA['const_q_true'] * \
                                  self.sampling_period * np.eye(self.target_dim)
        # Build a robot
        self.agent = AgentSE2(3, self.sampling_period, self.limit['agent'],
                            lambda x: map_utils.is_collision(self.MAP, x))
        # Build a target
        self.targets = [AgentSE2(self.target_dim, self.sampling_period, self.limit['target'],
                            lambda x: map_utils.is_collision(self.MAP, x),
                            policy=ConstantPolicy(self.target_noise_cov[3:, 3:])) for _ in range(num_targets)]
        # SinePolicy(0.5, 0.5, 2.0, self.sampling_period)
        # CirclePolicy(self.sampling_period, self.MAP.origin, 3.0)
        # RandomPolicy()

        self.belief_targets = [UKFbelief(dim=self.target_dim, limit=self.limit['target'], fx=SE2DynamicsVel,
                            W=self.target_noise_cov, obs_noise_func=self.observation_noise,
                            collision_func=lambda x: map_utils.is_collision(self.MAP, x))
                            for _ in range(num_targets)]

    def reset(self,  init_random=True, **kwargs):
        self.state = []
        init_pose = self.get_init_pose(init_random=init_random, **kwargs)
        self.agent.reset(init_pose['agent'])
        for i in range(self.num_targets):
            self.belief_targets[i].reset(
                        init_state=np.concatenate((init_pose['belief_targets'][i], np.zeros(2))),
                        init_cov=self.target_init_cov)
            t_init = np.concatenate((init_pose['targets'][i], [self.target_init_vel[0], 0.0]))
            self.targets[i].reset(t_init)
            self.targets[i].policy.reset(t_init)
            r, alpha, _ = util.xyg2polarb(self.belief_targets[i].state[:2],
                                 self.agent.state[:2], self.agent.state[2])
            logdetcov = np.log(LA.det(self.belief_targets[i].cov))
            self.state.extend([r, alpha, 0.0, 0.0, logdetcov, 0.0])
        self.state.extend([self.sensor_r, np.pi])
        self.state = np.array(self.state)
        return self.state

    def step(self, action):
        action_vw = self.action_map[action]
        boundary_penalty = self.agent.update(action_vw, [t.state[:2] for t in self.targets])
        obstacles_pt = map_utils.get_cloest_obstacle(self.MAP, self.agent.state)
        observed = []
        for i in range(self.num_targets):
            self.targets[i].update()
            # Observe
            obs = self.observation(self.targets[i])
            observed.append(obs[0])
            # Update the belief of the agent on the target using UKF
            self.belief_targets[i].update(obs[0], obs[1], self.agent.state,
             np.array([np.random.random(), np.pi*np.random.random()-0.5*np.pi]))

        reward, done, mean_nlogdetcov = self.get_reward(obstacles_pt, observed, self.is_training)
        self.state = []
        if obstacles_pt is None:
            obstacles_pt = (self.sensor_r, np.pi)
        for i in range(self.num_targets):
            r_b, alpha_b, _ = util.xyg2polarb(self.belief_targets[i].state[:2],
                                 self.agent.state[:2], self.agent.state[2])
            r_dot_b, alpha_dot_b = util.xyg2polarb_dot(
                                    self.belief_targets[i].state[:3],
                                    self.belief_targets[i].state[3:],
                                    self.agent.state, action_vw)
            self.state.extend([r_b, alpha_b, r_dot_b, alpha_dot_b,
                                    np.log(LA.det(self.belief_targets[i].cov)), float(observed[i])])
        self.state.extend([obstacles_pt[0], obstacles_pt[1]])
        self.state = np.array(self.state)
        return self.state, reward, done, {'mean_nlogdetcov': mean_nlogdetcov}

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

def reward_fun(belief_targets, obstacles_pt, observed, is_training=True,
        c_mean=0.01):

    detcov = [LA.det(b_target.cov) for b_target in belief_targets]
    r_detcov_mean = - np.mean(np.log(detcov))
    reward = c_mean * r_detcov_mean

    mean_nlogdetcov = None
    if not(is_training):
        logdetcov = [np.log(LA.det(b_target.cov)) for b_target in belief_targets]
        mean_nlogdetcov = -np.mean(logdetcov)
    return reward, False, mean_nlogdetcov
