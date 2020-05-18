import numpy as np
import ttenv.infoplanner_python as infoplanner_python

class InfoPlanner():
    def __init__(self, n_controls=5, T=12, delta=3, eps=np.infty, arvi_time=1,
                    range_limit=np.infty, debug=0):
        self.T = T
        self.delta = delta
        self.n_controls = n_controls
        self.eps = eps
        self.arvi_time = arvi_time
        self.range_limit = range_limit
        self.debug=debug

    def reset(self):
        self.planner = infoplanner_python.IGL.InfoPlanner()
        self.planner_outputs = [0] * 1 # 1 for one agent.
        self.step = 0

    def act(self, agent):
        if self.step % self.n_controls == 0:
            self.planner_outputs[0] = self.planner.planARVI(
                agent, self.T, self.delta, self.eps, self.arvi_time, self.debug, 0)
        action = self.planner_outputs[0].action_idx[-1]
        self.planner_outputs[0].action_idx = self.planner_outputs[0].action_idx[:-1]
        self.step += 1
        return action
