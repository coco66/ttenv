#!/usr/bin/env python

from ig_manager import IGManager
import numpy as np
import rospy
import pdb


# To use RL generated data in ROS
if __name__ == "__main__":
	node = IGManager()
	filename = node.data 

	data = pickle.load(open(filename), 'rb')
	# Load data
	sensing_range = data['sensor_r']
	sensing_fov = date['fov']
	robot_path = data['robots']
	target_path = data['targets']
	belief_target_path = data['belief_targets']
	belief_cov = data['belief_covs']
	Horizon = len(robot_path)
	num_robots = data['num_robots']
	num_targets = data['num_targets']
	# # Map Offsets
 #    x_off = + 8.8223
 #    y_off = + 8.8223

	rate = rospy.Rate(5)

    while node.robot_states[0] is None:  # Sleep Until ODOM is received to ensure things are set up.
        rate.sleep()

	# Main Loop
	for t in range(Horizon):
		for i in range(num_robots):
			node.publish_robot_wp(robot_path[t][i], i, 0.0)

		for i in range(num_targets):
			node.publish_target_wp(target_path[t][i], i , z=0.0)
			node.publish_target_belief(belief_target_path[t][i], belief_cov[t][i], i, z=0.0)


		rate.sleep()