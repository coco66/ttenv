"""
Generates a set of initial positions of targets and the agent as well as target
paths in ttenv.
If you want to have more conditions to generate initial positions other than
the current metadata version, provide values for the additional variables to
the reset function. For example,
    ex_var = {'init_distance_min':10.0,
                'init_distacne_max':15.0,
                'target_direction':False,
                'belief_direction':False,
                'blocked':True }
    env.reset(**ex_var)
"""
import numpy as np
import envs
import argparse
import pickle
import os, time

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', default='TargetTracking-v1')
parser.add_argument('--render', type=int, default=0)
parser.add_argument('--map', type=str, default="obstacles02")
parser.add_argument('--nb_targets', type=int, default=1)
parser.add_argument('--nb_paths', type=int, default=10)
parser.add_argument('--log_dir', type=str, default='.')
parser.add_argument('--manual_check', type=int, default=1)
args = parser.parse_args()

def main():
    env = envs.make(args.env,
                    'target_tracking',
                    render=True,
                    directory=args.log_dir,
                    map_name=args.map,
                    num_targets=args.nb_targets,
                    is_training=False,
                    )
    env_core = env
    while( not hasattr(env_core, '_elapsed_steps')):
        env_core = env_core.env
    env_core = env_core.env

    from logger import TTENV_TEST_SET_PUB, TTENV_MULTI_TEST_SET, TTENV_TEST_SET_PUB_MORE
    if args.nb_targets > 1:
        test_params = TTENV_MULTI_TEST_SET
    else:
        test_params = TTENV_TEST_SET_PUB
    for eval_num in range(len(test_params)):
        print("TTENV_TEST_SET_PUB: Eval Num %d ..."%eval_num)
        init_pose = []
        target_paths = []
        map_info = []
        while(len(init_pose) < args.nb_paths): # test episode
            _, done = env.reset(**test_params[eval_num]), False
            env_core.has_discovered = [1] * args.nb_targets
            proceed = False
            if args.manual_check:
                env.render()
                proceed = ("y" == input("%d, Init Pose Pass? (y/n) "%len(init_pose)))
            if proceed or not(args.manual_check):
                init_pose_k = {'agent':env_core.agent.state,
                                'targets':[env_core.targets[i].state for i in range(args.nb_targets)],
                                'belief_targets':[env_core.belief_targets[i].state for i in range(args.nb_targets)]}
                target_path_t = [[]] * args.nb_targets
                while not done:
                    _, _, done, _ = env.step(env.action_space.sample())
                    if args.render:
                        env.render()
                    for i in range(args.nb_targets):
                        target_path_t[i].append(env_core.targets[i].state)
                proceed = False
                if args.manual_check:
                    env.render()
                    proceed = ("y" == input("%d, Pass? (y/n) "%len(init_pose)))
                if proceed or not(args.manual_check):
                    init_pose.append(init_pose_k)
                    target_paths.append(target_path_t)
                    if 'dynamic_map' in args.map:
                        map_info.append({'chosen_idx': env_core.MAP.chosen_idx, 'rot_angs': env_core.MAP.rot_angs })

        np.save(open(os.path.join(args.log_dir,'path_%d.npy'%eval_num), 'wb'), target_paths)
        pickle.dump(init_pose, open(os.path.join(args.log_dir,'init_eval_%d.pkl'%eval_num), 'wb'))
        if 'dynamic_map' in args.map :
            pickle.dump(map_info, open(os.path.join(args.log_dir, 'map_info_%d.pkl'%eval_num), 'wb'))

if __name__ == "__main__":
    main()
