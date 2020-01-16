"""
This binding file was modified from the original file in https://bitbucket.org/brentsc/infoplanner.git.
for this target tracking environment.
"""
import numpy as np
import ttenv.infoplanner_python as infoplanner

# Example target locations
'''
Fixed Target Locations
'''
y0 = np.array([3.5, 6.8])
y1 = np.array([3.5, 2.2])
y2 = np.array([6.5, 2.2])
y3 = np.array([7.7, 6.8])
y4 = np.array([3.5, 4])
Y = [y0, y1, y2, y3, y4]

class Policy:
    """
    Policy class contains several simple policy implementations for SE2 moving targets.
    """

    def zero_policy(state):
        """
        Policy returns a zero control input.
        :return: Control input of v=0, w=0.
        """
        return np.array([0, 0])

    def linear_policy(speed):
        """
        Returns a linear policy with the requested speed.
        :return: Control input with v=speed, w=0.
        """
        return lambda state: np.array([speed, 0])

    def random_policy(v_sd, kw):
        """
        Returns a linear policy with the requested speed.
        :return: Control input with v=speed, w=0.
        """
        v = v_sd #np.random.normal(0.0, v_sd)
        if np.random.random() < 0.1: # with probability p,
            w = np.random.normal(0.0, kw) / 180 * 2 * np.pi - np.pi
        else:
            w = 0.0
        return lambda state: np.array([v, w])


class Configure(object):
    """
    Configure class manages the construction of targets.
    """

    def __init__(self, map_nd, cmap):
        """
        Initialize the Configuration Object.
        :param map_nd: The map for boundary checking.
        :param cmap: The costmap for collision checking.
        """
        self.map_nd = map_nd
        self.cmap = cmap

    def setup_integrator_targets(self, n_targets=1, init_pos=None, init_vel=(0.0, 0.0), max_vel=1.0, tau=0.5, q=0.0):
        """
        Setup Ground Truth Integrator Moving Targets.
        :param n_targets: The number of targets to add.
        :param init_pos : Initial positions for all targets - (N by 2 for N = # of targets). If None is given, initialized to the fixed value.
        :param init_vel : Initial target velocity. Assumed all targets have the same initial velocity.
        :param max_vel: The maximum velocity targets can attain.
        :param tau: The time discretization.
        :param q: The Noise diffusion parameter for the simulation.
        :return: The world model containing the requested targets.
        """
        if init_pos is None:
            init_pos = Y
        else:
            assert(len(init_pos)==n_targets)
        world_model = infoplanner.IGL.target_model(self.map_nd, self.cmap)
        vel = np.array(init_vel)
        # Add All targets to belief
        [world_model.addTarget(i, infoplanner.IGL.DoubleInt2D(i, init_pos[i], vel, tau, max_vel, q)) for i in range(0, n_targets)]
        return world_model

    def setup_integrator_belief(self, n_targets=1, init_pos=None, init_vel=(0.0, 0.0), max_vel=1.0, tau=0.5, q=0.0, cov_pos=.25, cov_vel=.1):
        """
        Setup Integrator Belief Model for Moving Targets.
        :param n_targets: The number of targets to add to the belief.
        :param init_pos : Initial positions for all targets - (N by 2 for N = # of targets). If None is given, initialized to the fixed value.
        :param init_vel : Initial target velocity. Assumed all targets have the same initial velocity.
        :param max_vel: The maximum velocity target beliefs can attain.
        :param tau: The time discretization.
        :param q: The Noise diffusion parameter for the model.
        :param cov_pos: The initial covariance in position.
        :param cov_vel: The initial covariance in velocity.
        :return: The belief model containing the requested targets.
        """
        if init_pos is None:
            init_pos = Y
        else:
            assert(len(init_pos)==n_targets)
        belief_model = infoplanner.IGL.info_target_model(self.map_nd, self.cmap)
        sigma = np.identity(4)
        sigma[0:2, 0:2] = cov_pos * np.identity(2)
        sigma[2:4, 2:4] = cov_vel * np.identity(2)
        vel = np.array(init_vel)
        [belief_model.addTarget(i, infoplanner.IGL.DoubleInt2DBelief(
            infoplanner.IGL.DoubleInt2D(i, init_pos[i], vel, tau, max_vel, q), sigma)) for i in
         range(0, n_targets)]  # Add All targets to belief

        return belief_model

    def setup_static_targets(self, n_targets=1, init_pos=None, q=0.0):
        """
        Set up a Ground Truth static target model.
        :param n_targets: The number of targets to add.
        :param q: The noise parameter of the simulation.
        :return: The requested world model.
        """
        if init_pos is None:
            init_pos = Y
        else:
            assert(len(init_pos)==n_targets)
        world_model = infoplanner.IGL.target_model(self.map_nd, self.cmap)
        # Add All targets to belief
        [world_model.addTarget(i, infoplanner.IGL.Static2D(i, init_pos[i], q)) for i in range(0, n_targets)]
        return world_model

    def setup_static_belief(self, n_targets=1, init_pos=None, q=0.0, cov_pos=0.25):
        """
        Setup a Static Belief target Model.
        :param n_targets: The number of requested targets in the belief.
        :param q: The noise parameter of the belief model.
        :param cov_pos: The initial position covariance.
        :return: The requested belief model.
        """
        if init_pos is None:
            init_pos = Y
        else:
            assert(len(init_pos)==n_targets)
        belief_model = infoplanner.IGL.info_target_model(self.map_nd, self.cmap)
        # Add All targets to belief
        [belief_model.addTarget(i, infoplanner.IGL.Static2DBelief(infoplanner.IGL.Static2D(i, init_pos[i], q), cov_pos * np.identity(2))) for i in
         range(0, n_targets)]
        return belief_model

    def setup_se2_targets(self, n_targets=1, init_odom=None, policy=Policy.zero_policy, tau=0.5, q=0.0):
        """
        Set up a Ground Truth simulation of SE(2) moving targets.
        :param n_targets: The number of targets requested.
        :param policy: The control policy used for the targets.
        :param tau: The time discretization.
        :param q: The Noise parameter.
        :return: The requested belief model.
        """
        if init_odom is None:
            init_odom = [np.concatenate((y, 0.0)) for y in Y]
        else:
            assert(len(init_odom)==n_targets)
        world_model = infoplanner.IGL.target_model(self.map_nd, self.cmap)
        controller = infoplanner.IGL.SE2Policy(policy)
        [world_model.addTarget(i, infoplanner.IGL.SE2Target(i, np.array(init_odom[i]), policy=controller, tau=tau,
                                                     q=q))
         for i in
         range(0, n_targets)]
        return world_model

    # def setup_se2_belief(self, n_targets=1, init_odom=None, policy=Policy.zero_policy, tau=0.5, q=0.0, cov=0.25):
    #     """
    #     Set up a Ground Truth simulation of SE(2) moving targets.
    #     :param n_targets: The number of targets requested.
    #     :param policy: The control policy used for the targets.
    #     :param tau: The time discretization.
    #     :param q: The Noise parameter.
    #     :return: The requested belief model.
    #     """
    #     if init_odom is None:
    #         init_odom = [np.concatenate((y, 0.0)) for y in Y]
    #     else:
    #         assert(len(init_odom)==n_targets)
    #     belief_model = infoplanner.IGL.info_target_model(self.map_nd, self.cmap)
    #     controller = infoplanner.IGL.SE2Policy(policy)
    #     [belief_model.addTarget(i, infoplanner.IGL.SE2Belief(infoplanner.IGL.SE2Target(i, np.array(init_odom[i]), policy=controller, tau=tau,
    #                                                  q=q), cov*np.identity(3)))
    #      for i in
    #      range(0, n_targets)]
    #     return belief_model
