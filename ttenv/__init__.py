from gym import wrappers
import ttenv.target_tracking

def make(env_name, render=False, figID=0, record=False, ros=False, directory='',
                                        T_steps=None, num_targets=1, **kwargs):
    """
    env_name : str
        name of an environment. (e.g. 'Cartpole-v0')
    type : str
        type of an environment. One of ['atari', 'classic_control',
        'classic_mdp','target_tracking']
    """
    if T_steps is None:
        if num_targets > 1:
            T_steps = 150
        else:
            T_steps = 100
    if env_name == 'TargetTracking-v0':
        env0 = target_tracking.TargetTrackingEnv0(**kwargs)
    elif env_name == 'TargetTracking-v1':
        env0 = target_tracking.TargetTrackingEnv1(**kwargs)
    elif env_name == 'TargetTracking-v2':
        env0 = target_tracking.TargetTrackingEnv2(**kwargs)
    elif env_name == 'TargetTracking-v3':
        env0 = target_tracking.TargetTrackingEnv3(**kwargs)
    elif env_name == 'TargetTracking-v4':
        env0 = target_tracking.TargetTrackingEnv4(**kwargs)
    elif env_name == 'TargetTracking-info1':
        from ttenv.infoplanner_python.target_tracking_infoplanner import TargetTrackingInfoPlanner1
        env0 = TargetTrackingInfoPlanner1(**kwargs)
    else:
        raise ValueError('No such environment exists.')

    env = wrappers.TimeLimit(env0, max_episode_steps=T_steps)
    if ros:
        from ttenv.ros_wrapper import Ros
        env = Ros(env)
    if render:
        from ttenv.display_wrapper import Display2D
        env = Display2D(env, figID=figID)
    if record:
        from ttenv.display_wrapper import Video2D
        env = Video2D(env, dirname = directory)

    return env
