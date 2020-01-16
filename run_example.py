import ttenv
import numpy as np
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', type=str, default='TargetTracking-v1')
parser.add_argument('--render', help='whether to render', type=int, default=0)
parser.add_argument('--record', help='whether to record', type=int, default=0)
parser.add_argument('--ros', help='whether to use ROS', type=int, default=0)
parser.add_argument('--nb_targets', help='the number of targets', type=int, default=1)
parser.add_argument('--log_dir', help='a path to a directory to log your data', type=str, default='.')
parser.add_argument('--map', type=str, default="emptySmall")

args = parser.parse_args()

def main():
    env = ttenv.make(args.env,
                    render=args.render,
                    record=args.record,
                    ros=args.ros,
                    map_name=args.map,
                    directory=args.log_dir,
                    num_targets=args.nb_targets,
                    is_training=False,
                    )
    num_actions = env.action_space.n
    nlogdetcov = []
    obs, done = env.reset(), False
    while(not done):
        if args.render:
            env.render()
        action = np.random.randint(0, high=num_actions-1)
        obs, rew, done, info = env.step(action)
        nlogdetcov.append(info['mean_nlogdetcov'])

    print("Sum of negative logdet of the target belief covariances : %.2f"%np.sum(nlogdetcov))

if __name__ == "__main__":
    main()
