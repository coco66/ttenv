import numpy as np
from numpy import linalg as LA
import os

from ttenv.maps import map_utils
import ttenv.util as util

from ttenv.agent_models import Agent, AgentDoubleInt2D_Nonlinear
from ttenv.metadata import METADATA
from ttenv.target_tracking import TargetTrackingEnv1
from ttenv.target_tracking import TargetTrackingBase
from ttenv.belief_tracker import KFbelief

import ttenv.infoplanner_python as infoplanner
from ttenv.infoplanner_python.infoplanner_binding import Configure, Policy


class BeliefWrapper(object):
    def __init__(self, num_targets=1, dim=4):
        self.num_targets = num_targets
        self.dim = dim
        self.state = None
        self.cov = None

    def update(self, state, cov):
        self.state = np.reshape(state, (self.num_targets, self.dim))
        self.cov = [cov[n*self.dim: (n+1)*self.dim,n*self.dim: (n+1)*self.dim ] for n in range(self.num_targets)]

class TargetWrapper(object):
    def __init__(self, num_targets=1, dim=4):
        self.state = None
        self.num_targets = num_targets
        self.dim = dim

    def reset(self, target):
        self.target = target
        self.state = np.reshape(self.target.getTargetState(), (self.num_targets, self.dim))

    def update(self):
        self.target.forwardSimulate(1)
        self.state = np.reshape(self.target.getTargetState(), (self.num_targets, self.dim))

class FeedTargetWrapper(TargetWrapper):
    def __init__(self, num_targets=1, dim=4):
        super().__init__(num_targets, dim)

    def update(self, target_state):
        self.target.setTargetState(target_state)
        self.state = np.reshape(target_state, (self.num_targets, self.dim))

class TargetTrackingInfoPlanner2(TargetTrackingEnv1):
    """
    Target tracking envrionment using InfoPlanner algorithm for the agent model.
    Target and belief models use ttenv functions.    
    """
    def __init__(self, num_targets=1, map_name='empty', is_training=True, known_noise=True):
        TargetTrackingEnv1.__init__(self, num_targets=num_targets,
            map_name=map_name, is_training=is_training, known_noise=known_noise)
        self.id = 'TargetTracking-info2'
        self.info_targets = FeedTargetWrapper(num_targets)

    def reset(self, **kwargs):
        self.has_discovered = [0] * self.num_targets
        self.state = []
        self.num_collisions = 0

        # Always set the limits first.
        if 'target_speed_limit' in kwargs:
            self.set_limits(target_speed_limit=kwargs['target_speed_limit'])

        if not('const_q' in kwargs):
            kwargs['const_q'] = METADATA['const_q']
        self.build_models(**kwargs)

        # Reset the agent, targets, and beliefs with sampled initial positions.
        init_pose = self.get_init_pose(**kwargs)

        a_init_igl = infoplanner.IGL.SE3Pose(init_pose['agent'], np.array([0, 0, 0, 1]))
        t_init_b_sets, t_init_sets = [], []
        for i in range(self.num_targets):
            t_init_b_sets.append(init_pose['belief_targets'][i][:2])
            t_init_sets.append(init_pose['targets'][i][:2])

        belief_target = self.cfg.setup_integrator_belief(n_targets=self.num_targets,
                                                    q=self.const_q,
                                                    init_pos=t_init_b_sets,
                                                    cov_pos=self.target_init_cov,
                                                    cov_vel=self.target_init_cov,
                                                    init_vel=self.target_init_vel)
        info_targets = self.cfg.setup_integrator_targets(n_targets=self.num_targets,
                                                init_pos=t_init_sets,
                                                init_vel=self.target_init_vel,
                                                q=self.const_q,
                                                max_vel=self.target_speed_limit)  # Integrator Ground truth Model
        self.info_targets.reset(info_targets)

        # Build a robot
        self.agent.reset(a_init_igl, belief_target)

        # Reset targets and beliefs.
        for i in range(self.num_targets):
            self.belief_targets[i].reset(
                        init_state=np.concatenate((init_pose['belief_targets'][i][:2], np.zeros(2))),
                        init_cov=self.target_init_cov)
            self.targets[i].reset(np.concatenate((init_pose['targets'][i][:2], self.target_init_vel)))

        # The targets are observed by the agent (z_0) and the beliefs are updated (b_0).
        observed = self.observe_and_update_belief()

        # Predict the target for the next step, b_1|0.
        for i in range(self.num_targets):
            self.belief_targets[i].predict()

        # Compute the RL state.
        self.state_func([0.0, 0.0], observed)

        return self.state

    def step(self, action):
        # The agent performs an action (t -> t+1)
        target_states = np.array([self.targets[i].state for i in range(self.num_targets)])
        is_col = self.agent.update(action, target_states) # No collision detection for now.
        action_vw = self.action_map[action]
        self.num_collisions += is_col

        # The targets move (t -> t+1)
        target_state = []
        for i in range(self.num_targets):
            if self.has_discovered[i]:
                self.targets[i].update(self.agent.state[:2])
            target_state.extend(self.targets[i].state)
        self.info_targets.update(target_state)

        # The targets are observed by the agent (z_t+1) and the beliefs are updated.
        observed = self.observe_and_update_belief()

        # Compute a reward from b_t+1|t+1 or b_t+1|t.
        reward, done, mean_nlogdetcov, std_nlogdetcov = self.get_reward(self.is_training, is_col=False)

        # Predict the target for the next step, b_t+2|t+1
        for i in range(self.num_targets):
            self.belief_targets[i].predict()

        # Compute the RL state.
        self.state_func(action_vw, observed)

        return self.state, reward, done, {'mean_nlogdetcov': mean_nlogdetcov, 'std_nlogdetcov': std_nlogdetcov}

    def observe_and_update_belief(self):
        observed = super().observe_and_update_belief()

        # Update the belief in the info agent.
        b_mean = [self.belief_targets[i].state for i in range(self.num_targets)]
        b_cov = [self.belief_targets[i].cov for i in range(self.num_targets)]
        self.agent.update_belief_state(b_mean, b_cov)
        return observed

    def build_models(self, const_q=None, known_noise=True, **kwargs):
        self.MAP.generate_map(**kwargs)
        map_dir_path = '/'.join(map_utils.__file__.split('/')[:-1])
        # Setup Ground Truth Target Simulation
        map_nd = infoplanner.IGL.map_nd(self.MAP.mapmin, self.MAP.mapmax, self.MAP.mapres)
        if self.MAP.map is None:
            cmap_data = list(map(str, [0] * map_nd.size()[0] * map_nd.size()[1]))
        else:
            cmap_data = list(map(str, np.squeeze(self.MAP.map.astype(np.int8).reshape(-1, 1)).tolist()))
        se2_env = infoplanner.IGL.SE2Environment(map_nd, cmap_data, os.path.join(map_dir_path,'mprim_SE2_RL.yaml'))

        self.cfg = Configure(map_nd, cmap_data)
        sensor = infoplanner.IGL.RangeBearingSensor(self.sensor_r, self.fov, self.sensor_r_sd, self.sensor_b_sd, map_nd, cmap_data)
        self.agent = Agent_InfoPlanner(dim=3, sampling_period=self.sampling_period, limit=self.limit['agent'],
                            collision_func=lambda x: self.MAP.is_collision(x, margin=0.0),
                            se2_env=se2_env, sensor_obj=sensor)

        if const_q is None:
            self.const_q = np.random.choice([0.001, 0.1, 1.0])
        else:
            self.const_q = const_q

        # Build targets
        self.targetA = np.concatenate((np.concatenate((np.eye(2), self.sampling_period*np.eye(2)), axis=1),
                                        [[0,0,1,0],[0,0,0,1]]))
        self.target_noise_cov = self.const_q * np.concatenate((
                            np.concatenate((self.sampling_period**3/3*np.eye(2), self.sampling_period**2/2*np.eye(2)), axis=1),
                        np.concatenate((self.sampling_period**2/2*np.eye(2), self.sampling_period*np.eye(2)),axis=1) ))
        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = self.const_q_true * np.concatenate((
                        np.concatenate((self.sampling_period**2/2*np.eye(2), self.sampling_period/2*np.eye(2)), axis=1),
                        np.concatenate((self.sampling_period/2*np.eye(2), self.sampling_period*np.eye(2)),axis=1) ))

        self.targets = [AgentDoubleInt2D_Nonlinear(self.target_dim,
                            self.sampling_period, self.limit['target'],
                            lambda x: self.MAP.is_collision(x),
                            W=self.target_true_noise_sd, A=self.targetA,
                            obs_check_func=lambda x: self.MAP.get_closest_obstacle(
                                x, fov=2*np.pi, r_max=10e2))
                            for _ in range(self.num_targets)]
        self.belief_targets = [KFbelief(dim=self.target_dim,
                            limit=self.limit['target'], A=self.targetA,
                            W=self.target_noise_cov,
                            obs_noise_func=self.observation_noise,
                            collision_func=lambda x: self.MAP.is_collision(x, margin=0.0))
                            for _ in range(self.num_targets)]

class TargetTrackingInfoPlanner1(TargetTrackingEnv1):
    """
    Double Integrator
    """
    def __init__(self, num_targets=1, map_name='empty', is_training=True, known_noise=True):
        TargetTrackingEnv1.__init__(self, num_targets=num_targets,
            map_name=map_name, is_training=is_training, known_noise=known_noise)
        self.id = 'TargetTracking-info1'

        map_dir_path = '/'.join(map_utils.__file__.split('/')[:-1])
        # Setup Ground Truth Target Simulation
        map_nd = infoplanner.IGL.map_nd(self.MAP.mapmin, self.MAP.mapmax, self.MAP.mapres)
        if self.MAP.map is None:
            cmap_data = list(map(str, [0] * map_nd.size()[0] * map_nd.size()[1]))
        else:
            cmap_data = list(map(str, np.squeeze(self.MAP.map.astype(np.int8).reshape(-1, 1)).tolist()))
        se2_env = infoplanner.IGL.SE2Environment(map_nd, cmap_data, os.path.join(map_dir_path,'mprim_SE2_RL.yaml'))

        self.cfg = Configure(map_nd, cmap_data)
        sensor = infoplanner.IGL.RangeBearingSensor(self.sensor_r, self.fov, self.sensor_r_sd, self.sensor_b_sd, map_nd, cmap_data)
        self.agent = Agent_InfoPlanner(dim=3, sampling_period=self.sampling_period, limit=self.limit['agent'],
                            collision_func=lambda x: self.MAP.is_collision(x),
                            se2_env=se2_env, sensor_obj=sensor)
        self.belief_targets = BeliefWrapper(num_targets)
        self.targets = TargetWrapper(num_targets)

    def reset(self, **kwargs):
        self.state = []
        t_init_sets = []
        t_init_b_sets = []
        init_pose = self.get_init_pose(**kwargs)
        a_init_igl = infoplanner.IGL.SE3Pose(init_pose['agent'], np.array([0, 0, 0, 1]))
        for i in range(self.num_targets):
            t_init_b_sets.append(init_pose['belief_targets'][i][:2])
            t_init_sets.append(init_pose['targets'][i][:2])
            r, alpha = util.relative_distance_polar(np.array(t_init_b_sets[-1][:2]),
                                    xy_base=np.array(init_pose['agent'][:2]),
                                    theta_base=init_pose['agent'][2])
            logdetcov = np.log(LA.det(self.target_init_cov*np.eye(self.target_dim)))
            self.state.extend([r, alpha, 0.0, 0.0, logdetcov, 0.0])

        self.state.extend([self.sensor_r, np.pi])
        self.state = np.array(self.state)
        # Build a target
        target = self.cfg.setup_integrator_targets(n_targets=self.num_targets,
                                                init_pos=t_init_sets,
                                                init_vel=self.target_init_vel,
                                                q=METADATA['const_q_true'],
                                                max_vel=METADATA['target_speed_limit'])  # Integrator Ground truth Model
        belief_target = self.cfg.setup_integrator_belief(n_targets=self.num_targets, q=METADATA['const_q'],
                                                init_pos=t_init_b_sets,
                                                cov_pos=self.target_init_cov, cov_vel=self.target_init_cov,
                                                init_vel=(0.0, 0.0))
         # Build a robot
        self.agent.reset(a_init_igl, belief_target)
        self.targets.reset(target)
        self.belief_targets.update(self.agent.get_belief_state(), self.agent.get_belief_cov())
        return np.array(self.state)

    def get_reward(self, obstacles_pt, observed, is_training=True):
        if obstacles_pt is None:
            penalty = 0.0
        else:
            penalty = METADATA['margin2wall']**2 * \
                        1./max(METADATA['margin2wall']**2, obstacles_pt[0]**2)

        if sum(observed) == 0:
            reward = - penalty
        else:
            cov = self.agent.get_belief_cov()
            detcov = [LA.det(cov[self.target_dim*n: self.target_dim*(n+1), self.target_dim*n: self.target_dim*(n+1)]) for n in range(self.num_targets)]
            reward = - 0.1 * np.log(np.mean(detcov) + np.std(detcov)) - penalty
            reward = max(0.0, reward) + np.mean(observed)

        mean_nlogdetcov = None
        if not(is_training):
            cov = self.agent.get_belief_cov()
            logdetcov = [np.log(LA.det(cov[self.target_dim*n: self.target_dim*(n+1), self.target_dim*n: self.target_dim*(n+1)])) for n in range(self.num_targets)]
            mean_nlogdetcov = -np.mean(logdetcov)

        return reward, False, mean_nlogdetcov

    def step(self, action):
        self.agent.update(action, self.targets.state)

        # Update the true target state
        self.targets.update()
        # Observe
        measurements = self.agent.observation(self.targets.target)
        obstacles_pt = self.MAP.get_closest_obstacle(self.agent.state)
        # Update the belief of the agent on the target using KF
        GaussianBelief = infoplanner.IGL.MultiTargetFilter(measurements, self.agent.agent, debug=False)
        self.agent.update_belief(GaussianBelief)
        self.belief_targets.update(self.agent.get_belief_state(), self.agent.get_belief_cov())

        observed = [m.validity for m in measurements]
        reward, done, mean_nlogdetcov = self.get_reward(obstacles_pt, observed, self.is_training)
        if obstacles_pt is None:
            obstacles_pt = (self.sensor_r, np.pi)

        self.state = []
        target_b_state = self.agent.get_belief_state()
        target_b_cov = self.agent.get_belief_cov()
        control_input = self.action_map[action]
        for n in range(self.num_targets):
            r_b, alpha_b = util.relative_distance_polar(
                        target_b_state[self.target_dim*n: self.target_dim*n+2],
                        xy_base=self.agent.state[:2], theta_base=self.agent.state[2])
            r_dot_b, alpha_dot_b = util.relative_velocity_polar(
                                    target_b_state[self.target_dim*n: self.target_dim*n+2],
                                    target_b_state[self.target_dim*n+2:],
                                    self.agent.state[:2], self.agent.state[2],
                                    control_input[0], control_input[1])
            self.state.extend([r_b, alpha_b, r_dot_b, alpha_dot_b,
                                    np.log(LA.det(target_b_cov[self.target_dim*n: self.target_dim*(n+1), self.target_dim*n: self.target_dim*(n+1)])),
                                        float(observed[n])])

        self.state.extend([obstacles_pt[0], obstacles_pt[1]])
        self.state = np.array(self.state)
        return self.state, reward, done, {'mean_nlogdetcov': mean_nlogdetcov}

class Agent_InfoPlanner(Agent):
    def __init__(self,  dim, sampling_period, limit, collision_func,
                    se2_env, sensor_obj, margin=METADATA['margin']):
        Agent.__init__(self, dim, sampling_period, limit, collision_func, margin=margin)
        self.se2_env = se2_env
        self.sensor = sensor_obj
        self.sampling_period = sampling_period
        self.action_map = {}
        self.action_map_rev = {}
        for (i,v) in enumerate(METADATA['action_v']):
            for (j,w) in enumerate(METADATA['action_w']):
                self.action_map[len(METADATA['action_w'])*i+j] = (v,w)
                self.action_map_rev[(v,w)] = len(METADATA['action_w'])*i+j
        self.vw = [0,0]

    def reset(self, init_state, belief_target=None):
        if belief_target is None:
            self.state = init_state
        else:
            self.agent = infoplanner.IGL.Robot(init_state, self.se2_env, belief_target, self.sensor)
            self.state = self.get_state()
        return self.state

    def update(self, action, target_state):
        self.vw = self.action_map[action]
        action, is_col = self.update_filter(action, target_state)
        self.agent.applyControl([int(action)], 1)
        self.state = self.get_state()
        return is_col

    def get_state(self):
        return np.concatenate((self.agent.getState().position[:2], [self.agent.getState().getYaw()]))

    def get_state_object(self):
        return self.agent.getState()

    def observation(self, target_obj):
        return self.agent.sensor.senseMultiple(self.get_state_object(), target_obj)

    def get_belief_state(self):
        return self.agent.tmm.getTargetState()

    def get_belief_cov(self):
        return self.agent.tmm.getCovarianceMatrix()

    def update_belief(self, GaussianBelief):
        self.agent.tmm.updateBelief(GaussianBelief.mean, GaussianBelief.cov)

    def update_belief_state(self, b_means, b_covs):
        b_means_input = np.concatenate(b_means)
        num_targets = len(b_covs)
        target_dim = len(b_covs[0])
        b_covs_input = np.zeros((num_targets*target_dim, num_targets*target_dim))
        for (i,cov) in enumerate(b_covs):
            b_covs_input[i*target_dim:(i+1)*target_dim,i*target_dim:(i+1)*target_dim] = cov
        self.agent.tmm.updateBelief(b_means_input, b_covs_input)

    def update_filter(self, action, target_state):
        state = self.get_state()
        control_input = self.action_map[action]
        tw = self.sampling_period*control_input[1]
        # Update the agent state
        if abs(tw) < 0.001:
            diff = np.array([self.sampling_period*control_input[0]*np.cos(state[2]+tw/2),
                    self.sampling_period*control_input[0]*np.sin(state[2]+tw/2),
                    tw])
        else:
            diff = np.array([control_input[0]/control_input[1]*(np.sin(state[2]+tw) - np.sin(state[2])),
                    control_input[0]/control_input[1]*(np.cos(state[2]) - np.cos(state[2]+tw)),
                    tw])
        new_state = state + diff
        if len(target_state.shape)==1:
            target_state = [target_state]
        target_col = False
        for n in range(target_state.shape[0]): # For each target
            target_col = np.sqrt(np.sum((new_state[:2] - target_state[n][:2])**2)) < METADATA['margin']
            if target_col:
                break
        is_col = self.collision_check(new_state)
        new_action = action
        return new_action, is_col
