import numpy as np
import pickle
import datetime, json, os, argparse, time

import ttenv
from ttenv.infoplanner_python.infoplanner import InfoPlanner

from pub.test_util import *

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
parser.add_argument('--eval_type', choices=['random', 'random_zone', 'fixed_init', 'fixed_path'], default='random')
parser.add_argument('--eval_param_dir', type=str, default=".")
parser.add_argument('--eval_id_max', type=int, default=4)
args = parser.parse_args()

def evaluate():
    np.random.seed(args.seed)
    logdir = os.path.join(args.log_dir, '_'.join([args.env, datetime.datetime.now().strftime("%m%d%H%M")]))
    if not os.path.exists(logdir):
            os.makedirs(logdir)
    else:
            ValueError("The directory already exists...", logdir)
    json.dump(vars(args), open(os.path.join(logdir, 'test_prop.json'), 'w'))

    planner = InfoPlanner(n_controls=args.n_controls)

    for eval_id in range(args.eval_id_max):
        print("\nEvaluating for EVAL ID %d"%eval_id)
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

        # Evaluation Type
        # 'fixed_init' : to use a fixed set of initial positions.
        # 'fixed_path' : to use a fixed set of target paths.
        if args.eval_type == 'fixed_init' or args.eval_type == 'fixed_path':
            given_init_pose = pickle.load(open(os.path.join(args.eval_param_dir,
                                            'init_eval_%d.pkl'%eval_id), "rb"))
            nb_test_episodes = len(given_init_pose)
            if args.map == 'dynamic_map':
                map_info = pickle.load(open(os.path.join(args.eval_param_dir,
                                            'map_info_%d.pkl'%eval_id), "rb"))
            if args.eval_type == 'fixed_path':
                target_path = np.load(args.eval_param_dir + '/path_%d.npy'%eval_id)
                init_params['target_path'] = target_path
                f_desc = "Eval type: %s from %s, %d \n"%(args.eval_type,
                                                args.eval_param_dir, eval_id)
            else:
                f_desc = "Eval type: %s from %s, %d \n"%(args.eval_type,
                                                args.eval_param_dir, eval_id)
        # 'random_zone' : to randomly initialize the agent, the targets, and the
        # beliefs in a given set of zones.
        elif args.eval_type == 'random_zone':
            if args.nb_targets == 1:
                from logger import TTENV_TEST_SET
                init_params = TTENV_TEST_SET[eval_id]
                f_desc = "Eval type: %s from %s set %d \n"%(args.eval_type,
                                                        "TTENV_TEST_SET", eval_id)
            else:
                from logger import TTENV_MULTI_TEST_SET
                init_params = TTENV_MULTI_TEST_SET[eval_id]
                f_desc = "Eval type: %s from %s set %d \n"%(args.eval_type,
                                                        "TTENV_MULTI_TEST_SET", eval_id)
        else:
            from ttenv.metadata import METADATA
            f_desc = "Eval type: %s from METADATA V%s"%(args.eval_type, str(METADATA['version']))

        log_data = {
                    'time_elapsed': [],
                    'ep_nlogdetcovs': [],
                    'ep_rewards': [],
                    'num_col_attempts': [],
                    'visit_ratio': [],
                    'resilience_rate': [],
                    'b_pos_a': [],
                    't_pos_a': [],
        }
        for ep in range(nb_test_episodes):
            # Recording
            total_rewards, total_nlogdetcov = 0, 0
            observed_history = []
            agent_state_history = []
            belief_state_history = []
            target_state_history = []
            t_speed, a_speed = [], []

            # Reset
            if args.map == 'dynamic_map' and (args.eval_type == 'fixed_init' or args.eval_type == 'fixed_path'):
                for (k,v) in map_info[ep].items():
                    init_params[k] = v

            obs = env.reset(init_pose_list=given_init_pose, **init_params)
            done = False
            env_core.MAP.reset_visit_map()
            lb, ub = get_nlogdetcov_bounds(
                            env_core.target_init_cov*np.eye(env_core.target_dim),
                            env_core.targetA, env_core.target_noise_cov, 100)

            planner.reset()

            s_time = time.time()
            while(not done):
                if args.render:
                    env.render()

                # Record
                observed_history.append([env_core.state[i*num_target_dep_vars+5]
                                    for i in range(args.nb_targets)])
                agent_state_history.append(env_core.agent.state)
                belief_state_history.append([env_core.belief_targets[i].state
                                    for i in range(args.nb_targets)])
                target_state_history.append([env_core.targets[i].state
                                    for i in range(args.nb_targets)])
                t_speed.append(np.sqrt(np.sum(env_core.targets[0].state[2:]**2)))
                a_speed.append(env_core.agent.vw[0])

                # Apply Control
                obs, rew, done, info = env.step(planner.act(env_core.agent.agent))

                total_rewards += rew
                # Prediction Measure.
                total_nlogdetcov += - np.mean([env_core.state[i*num_target_dep_vars+4] for i in range(args.nb_targets)]) # info['mean_nlogdetcov']

            log_data['time_elapsed'].append(time.time() - s_time)
            log_data['ep_nlogdetcovs'].append((total_nlogdetcov-lb)/(ub-lb))
            log_data['ep_rewards'].append(total_rewards)
            log_data['num_col_attempts'].append(env_core.num_collisions)

            n_visited_cells = np.sum(env_core.MAP.visit_map > 0)
            n_available_cells = np.sum(env_core.MAP.map == 0)
            log_data['visit_ratio'].append(n_visited_cells/n_available_cells)

            observed_history = np.array(observed_history)
            log_data['resilience_rate'].append(np.mean(
                                        [resilience_rate(observed_history[:,i])
                                                for i in range(args.nb_targets)]))
            b_pos_a, t_pos_a = state_stat(np.array(agent_state_history),
                                                np.array(belief_state_history),
                                                np.array(target_state_history))
            log_data['b_pos_a'].append(b_pos_a)
            log_data['t_pos_a'].append(t_pos_a)

            print("Ep.%d - Episode reward : %.2f, Episode nLogDetCov : %.2f, Num Collision : %d"%(ep, total_rewards, total_nlogdetcov,env_core.num_collisions))

        f_b_states = plot_states(np.concatenate(log_data['b_pos_a']))
        f_t_states = plot_states(np.concatenate(log_data['t_pos_a']))
        f_b_states.suptitle('Belief Positions in the Agent Frame')
        f_t_states.suptitle('Target Positions in the Agent Frame')
        f_b_states.savefig(os.path.join(logdir, "pos_stat_%d_b.png"%eval_id))
        f_t_states.savefig(os.path.join(logdir, "pos_stat_%d_t.png"%eval_id))

        if args.record :
            env.finish()

        pickle.dump(log_data,
                open(os.path.join(logdir,'test_result_%d.pkl'%eval_id), 'wb'))
        write_log_data(log_data, f_desc, logdir)

def visit_test(seed):
    np.random.seed(seed)

    logdir = os.path.join(args.log_dir, '_'.join(['test', datetime.datetime.now().strftime("%m%d%H%M")]))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    else:
        ValueError("The directory already exists...", logdir)

    planner = InfoPlanner(n_controls=args.n_controls)

    print("Visit Frequency Evaluation")
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

    nb_test_episodes = args.repeat
    from logger import TTENV_TEST_SET
    init_params = TTENV_TEST_SET[4]
    f_desc = "Evaluation for Visit frequency -  TTENV_TEST_SET[4]"

    f, ax = plt.subplots()
    ep = 0
    visit_map_total = np.zeros(env_core.MAP.mapdim)
    while(ep < nb_test_episodes):
        # Reset
        obs = env.reset(**init_params)
        done = False
        env_core.MAP.reset_visit_map()
        planner.reset()

        # # Plot the initial target and belief positions.
        # for i in range(args.nb_targets):
        #     ax.plot(env_core.targets[i].state[1], env_core.targets[i].state[0],
        #             'ro', markersize=8, fillstyle='full')
        #     ax.plot(env_core.belief_targets[i].state[1],
        #             env_core.belief_targets[i].state[0], 'ro', markersize=10,
        #             markeredgewidth=2, fillstyle='none')

        while(not done):
            if args.render:
                env.render()

            # Apply Control
            obs, rew, done, info = env.step(planner.act(env_core.agent.agent))
        visit_map_total += env_core.MAP.visit_map
        ep += 1

    if args.record :
        env.finish()
    log_data = {'visit_map': visit_map_total/nb_test_episodes}
    pickle.dump(log_data,
            open(os.path.join(logdir,'visit_test_result.pkl'), 'wb'))

    cmap = 'viridis'
    ax.imshow(visit_map_total/nb_test_episodes, cmap=cmap, origin='lower',
                        extent=[env_core.MAP.mapmin[0], env_core.MAP.mapmax[0],
                                env_core.MAP.mapmin[1], env_core.MAP.mapmax[1]])
    from matplotlib import cm as cm
    f.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax)
    f.savefig(os.path.join(logdir, "visit_map.png"))

if __name__ == "__main__":
    evaluate()
    # visit_test(args.seed)
