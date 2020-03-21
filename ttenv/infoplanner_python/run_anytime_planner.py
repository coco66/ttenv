import numpy as np
import pickle
import datetime, json, os, argparse, time

import ttenv
from ttenv.infoplanner_python.infoplanner import InfoPlanner

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--n_controls', help='number of controls', type=int, default=5)
parser.add_argument('--render', help='render', type=int, default=0)
parser.add_argument('--record', help='record', type=int, default=0)
parser.add_argument('--map', type=str, default="emptySmall")
parser.add_argument('--env', help='environment ID', default='TargetTracking-info1')
parser.add_argument('--ros', type=int, default=0)
parser.add_argument('--log_dir', type=str, default='.')
parser.add_argument('--nb_targets', type=int, default=1)
parser.add_argument('--repeat', type=int, default=1)
args = parser.parse_args()

if __name__ == "__main__":
    np.random.seed(args.seed)
    logdir = os.path.join(args.log_dir, '_'.join([args.env, datetime.datetime.now().strftime("%m%d%H%M")]))
    if not os.path.exists(logdir):
            os.makedirs(logdir)
    else:
            ValueError("The directory already exists...", logdir)
    json.dump(vars(args), open(os.path.join(logdir, 'test_prop.json'), 'w'))

    planner = InfoPlanner(n_controls=args.n_controls)

    env = ttenv.make(args.env,
                    render = bool(args.render),
                    record = bool(args.record),
                    ros = bool(args.ros),
                    directory=logdir,
                    map_name=args.map,
                    num_targets=args.nb_targets,
                    is_training=False
                    )

    env_core = env
    while( not hasattr(env_core, '_elapsed_steps')):
        env_core = env_core.env
    env_core = env_core.env

    num_target_dep_vars = env_core.num_target_dep_vars

    init_params = {}
    given_init_pose, test_init_pose = [], []
    nb_test_episodes = args.repeat

    for ep in range(nb_test_episodes):
        # Recording
        total_rewards, total_nlogdetcov = 0, 0

        obs = env.reset(init_pose_list=given_init_pose, **init_params)
        done = False
        env_core.MAP.reset_visit_map()

        planner.reset()

        s_time = time.time()
        while(not done):
            if args.render:
                env.render()

            # Apply Control
            obs, rew, done, info = env.step(planner.act(env_core.agent.agent))

            total_rewards += rew
            # Prediction Measure.
            total_nlogdetcov += - np.mean([env_core.state[i*num_target_dep_vars+4] for i in range(args.nb_targets)]) # info['mean_nlogdetcov']

        print("Ep.%d - Episode reward : %.2f, Episode nLogDetCov : %.2f"%(ep, total_rewards, total_nlogdetcov))

    if args.record :
        env.finish()
