import tensorflow as tf
import numpy as np
from gym import spaces, logger
from ttenv.target_tracking import TargetTrackingEnv1

METADATA={
        'sensor_r': 10.0,
        'fov' : 120,
        'sensor_r_sd': 0.15, # sensor range noise
        'sensor_b_sd': 0.001, # sensor bearing noise
    }
TARGET_INIT_COV = 30.0
TARGET_INIT_POS = (10.0, 10.0)
MARGIN = 3.0
MARGIN2WALL = 0.5

class TargetTrackingEnvRNN(TargetTrackingEnv1):
    """
    State: relative distance , relative angle, relative vel_0, relative vel_1, logdet(target uncertainty), observation boolean, cloest_obstacle/range (r, angle)
    Target: Target velocity is fixed.
    Target Estimate is only on its pose.
    Using a RNN for Target State Estimation
    """
    def __init__(self, agent_init_pos=(0.0, 0.0, 0.0), target_init_pos=TARGET_INIT_POS, target_init_cov=TARGET_INIT_COV,
        q_true=0.02, target_init_vel = 0.2, known_noise=True, q = 0.0, map_name='empty', is_training=True, num_targets=1):

        TargetTrackingEnv1.__init__(self, agent_init_pos, target_init_pos, target_init_cov, q_true=q_true,
            target_init_vel=target_init_vel, known_noise=known_noise, q = q, map_name=map_name, is_training=is_training,
            num_targets=num_targets)
        self.id = 'TargetTracking-vRNN'
        self.rnn_input_dim = 5
        self.launch_belief_models()#target_init_pos=np.concatenate((target_init_pos, [0.0, 0.0])))
        self.cost_hist = []
        """
        self.target = AgentVelControl2D(3, self.sampling_period,
                                        limit=self.limit['agent'],
                                        collision_func=lambda x: map_utils.is_collision(self.MAP, x, MARGIN2WALL))
        """
    def launch_belief_models(self, num_steps=30, batch_size=20, init_scale = 1.0, target_init_pos=None):
        from state_estimation import BeliefTargetModel, TargetHistory
        with tf.Graph().as_default():
            initializer = tf.random_uniform_initializer(-init_scale, init_scale)
            with tf.name_scope("Train"):
                with tf.variable_scope("Model", reuse=None, initializer=initializer):
                    self.belief_model_update= BeliefTargetModel( batch_size=batch_size,
                                                    num_steps=num_steps, dim=(self.rnn_input_dim, self.target_dim),
                                                    output_b_init=target_init_pos)
            with tf.name_scope("Test"):
                with tf.variable_scope("Model", reuse=True, initializer=initializer):
                    self.belief_model_predict = BeliefTargetModel( is_training=False, batch_size=1,
                                                    num_steps=1, dim=(self.rnn_input_dim, self.target_dim),
                                                    output_b_init=target_init_pos)
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        self.belief_model_update.assign_lr(self.sess)

        self.target_mem = TargetHistory(memory_size=500, batch_size=batch_size, num_steps=num_steps)
        self.episode_costs = []
        self.cost_hist = []

    def reset(self, init_random = True):
        self.target_b_cov = self.target_init_cov*np.eye(self.target_dim)
        if init_random:
            isvalid = False
            while(not isvalid):
                if self.MAP.map is None:
                    a_init = self.agent_init_pos[:2]
                else:
                    a_init = np.random.random((2,))*(self.MAP.mapmax-self.MAP.mapmin) + self.MAP.mapmin

                if not(map_utils.is_collision(self.MAP, a_init, MARGIN2WALL)):
                    rand_ang = np.random.rand()*2*np.pi - np.pi
                    t_r = np.random.rand()*(self.target_init_pos[0]-MARGIN) + MARGIN
                    t_init = np.array([t_r*np.cos(rand_ang), t_r*np.sin(rand_ang)]) + a_init
                    if (np.sqrt(np.sum((t_init - a_init)**2)) < MARGIN):
                        isvalid = False
                    else:
                        isvalid = not(map_utils.is_collision(self.MAP, t_init, MARGIN2WALL))
            self.agent.reset(np.concatenate((a_init, [np.random.random()*2*np.pi-np.pi])))
        else:
            self.agent.reset(self.agent_init_pos)
            t_init = self.target_init_pos
        self.target_b_state = np.concatenate((t_init + 10*(np.random.rand(2)-0.5), np.zeros(2)))
        self.target.reset(np.concatenate((t_init, self.target_init_vel)))
        #self.target.reset([10.0,10.0,0.0])
        r, alpha, _ = util.relative_measure(self.target_b_state, self.agent.state)
        self.state = np.array([r, alpha, 0.0, 0.0, 0.0, self.sensor_r, np.pi])
        self.target_mem.reset()
        self.belief_model_predict.reset()
        self.belief_model_update.reset(lr_decay=True, session=self.sess)

        if self.episode_costs:
            self.cost_hist.append(np.mean(self.episode_costs))
            self.episode_costs = []
            print(self.cost_hist)

        return np.array(self.state)

    def step(self, action):
        action_val = self.action_map[action]
        boundary_penalty = self.agent.update(action_val, self.target.state[:2])
        # Update the true target state
        self.target.update()
        #target_action = sin_control(self.target.state, x_interval=0.1, y_magnitude=5.0, sampling_period=self.sampling_period)
        #self.target.update(target_action, self.agent.state[:2])
        # Observe
        observed, z_tp1 = self.observation()
        obstacles_pt = map_utils.get_cloest_obstacle(self.MAP, self.agent.state)

        # Update the belief of the agent on the target using RNN
        if z_tp1 is None:
            rnn_input = np.zeros((self.rnn_input_dim,))
        else:
            rnn_input = np.concatenate((self.agent.state, z_tp1))

        self.target_mem.store(rnn_input, self.target.state)

        if len(self.target_mem) > self.target_mem.batch_size*5:
            for _ in range(5):
                batch_rnn_inputs, batch_rnn_targets = self.target_mem.get_batch()
                cost, err_cov = self.belief_model_update.update(self.sess, np.array(batch_rnn_inputs), np.array(batch_rnn_targets))
                self.episode_costs.append(cost)
            self.target_b_cov = err_cov
            self.target_b_state = self.belief_model_predict.predict(self.sess, rnn_input[np.newaxis, np.newaxis, :])

        reward, done = self.get_reward(obstacles_pt, observed)
        if obstacles_pt is None:
            obstacles_pt = (self.sensor_r, np.pi)
        r_b, alpha_b, _ = util.relative_measure(self.target_b_state, self.agent.state)
        rel_target_vel = util.coord_change2b(self.target_b_state[2:], alpha_b+self.agent.state[-1])
        self.state = np.array([r_b, alpha_b, rel_target_vel[0], rel_target_vel[1], float(observed), obstacles_pt[0], obstacles_pt[1]])
        return self.state, reward, done, {}
