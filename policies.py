# Control policies
import numpy as np
import envs.env_util as util

class RandomPolicy():
    def __init__(self, lim_vel=2.0):
        self.limit = np.array([[-lim_vel, -np.pi], [lim_vel, np.pi]])

    def get_control(self, odom):
        return np.random.random(2) * (self.limit[1]-self.limit[0]) + self.limit[0]

    def collision(self):
        pass

class SinePolicy():
    def __init__(self, x_interval, a, b, sampling_period, lim_vel=2.0):
        self.x_interval = x_interval
        self.a = a # constant factor to x
        self.b = b # y magnitude
        self.sampling_period = sampling_period
        self.limit = np.array([[-lim_vel, -np.pi], [lim_vel, np.pi]])

    def get_control(self, odom):
        R = np.array([[np.cos(self.th), -np.sin(self.th)],[np.sin(self.th), np.cos(self.th)]])
        p_b = - np.matmul(R.T, self.init_org) + np.matmul(R.T, odom[:2])
        p_b_tp1 = [p_b[0] + self.x_interval, self.b * np.sin(self.a * (p_b[0] + self.x_interval))]
        p_g_tp1 = self.init_org + np.matmul(R, p_b_tp1)
        th_b_tp1 = np.arctan(self.b * self.a * np.cos( p_b_tp1[0] * self.a ))
        th_g_tp1 = util.wrap_around(th_b_tp1 + self.th)

        ang_vel = util.wrap_around(th_g_tp1 - odom[2]) / self.sampling_period
        lin_vel = np.sqrt(np.sum((p_g_tp1 - odom[:2])**2)) / self.sampling_period

        return np.clip(np.array([lin_vel, ang_vel]), self.limit[0], self.limit[1])

    def collision(self, odom):
        self.init_org = odom[:2]
        self.th = util.wrap_around(odom[2] + np.random.random()*np.pi/2 - np.pi/4)

    def reset(self, init_odom):
        self.th = init_odom[2]
        self.init_org = init_odom[:2]

class CirclePolicy():
    def __init__(self, sampling_period, maporigin, d_th, lim_vel=2.0):
        self.sampling_period = sampling_period
        self.maporigin = maporigin
        self.d_th = d_th
        self.limit = np.array([[-lim_vel, -np.pi], [lim_vel, np.pi]])

    def get_control(self, odom):
        th = self.d_th/180.0*np.pi
        r = np.sqrt((odom[0] - self.maporigin[0])**2 + (odom[1] - self.maporigin[1])**2)
        alpha = np.arctan2(odom[1] - self.maporigin[0], odom[0] - self.maporigin[1])
        x = r*np.cos(alpha+th) + self.maporigin[0] + np.random.random() - 0.5
        y = r*np.sin(alpha+th) + self.maporigin[1] + np.random.random() - 0.5
        v = np.sqrt((x - odom[0])**2 + (y - odom[1])**2)/self.sampling_period
        w = util.wrap_around(np.arctan2(y - odom[1], x - odom[0]) - odom[2])/self.sampling_period
        return np.clip(np.array([v,w]), self.limit[0], self.limit[1])

    def collision(self):
        self.d_th = -self.d_th

    def reset(self):
        self.d_th = np.random.random()*np.pi

class ConstantPolicy():
    def __init__(self, noise_cov, lim_vel=2.0):
        self.noise_cov = noise_cov
        self.limit = np.array([[-lim_vel, -np.pi/5], [lim_vel, np.pi/5]])

    def get_control(self, state):
        uv = np.random.multivariate_normal(state[-2:], self.noise_cov)
        return np.clip(uv, self.limit[0], self.limit[1])

    def collision(self, state):
        return np.array([state[-2], -state[-1]])


    def reset(self, init_state):
        pass
