import numpy as np
import matplotlib.pyplot as plt

from ttenv.maps import map_utils

n_rows = 20
grid_map = np.zeros((n_rows,n_rows))
n_obs_cell = 2
grid_map[:n_obs_cell,:] = 1.0
grid_map[:,:n_obs_cell] = 1.0

res = 0.25
target_speed_limit = 3.0
r_margin = 0.1
sampling_period = 0.5
arrow_params = {'length_includes_head': False, 'shape': 'full',
                    'head_starts_at_zero': True}
scale = 0.2
def plot_arrows(ax, odom_2, color, speed):
    direction = [0.1*np.cos(odom_2), 0.1*np.sin(odom_2)]
    for r in range(n_obs_cell, n_rows):
        for c in range(n_obs_cell, n_rows):
            if r!=c:
                if r < c:
                    closest_obs_cell = [n_obs_cell-1,c]
                else:
                    closest_obs_cell = [r,n_obs_cell-1]

                closest_obs_pos = map_utils.cell_to_se2(closest_obs_cell, [0.0, 0.0], res)
                cell_pos = map_utils.cell_to_se2([r,c], [0.0, 0.0], res)
                obs_pos = []
                rel_obs_pos = np.array(closest_obs_pos) - np.array(cell_pos)
                obs_pos.append(np.sqrt(np.sum(rel_obs_pos**2)))
                obs_pos.append(np.arctan2(rel_obs_pos[1], rel_obs_pos[0])-odom_2)
                while(obs_pos[1] > np.pi):
                    obs_pos[1] -= 2*np.pi
                while(obs_pos[1] < -np.pi):
                    obs_pos[1] += 2*np.pi
                rot_ang = np.pi/2 * (1. + 1./(1. + np.exp(-(speed-0.5*target_speed_limit))))

                acc = max(0.0, speed * np.cos(obs_pos[1])) / max(1.0, obs_pos[0] - r_margin)
                th = obs_pos[1] - rot_ang if obs_pos[1] >= 0 else obs_pos[1] + rot_ang
                del_vx = acc * np.cos(th + odom_2) * sampling_period
                del_vy = acc * np.sin(th + odom_2) * sampling_period
                # ax.plot(cell_pos[0], cell_pos[1], 'bo', markerfacecolor='b', markersize=2)
                # ax.arrow(cell_pos[0], cell_pos[1], direction[0], direction[1],
                #     fc='b', ec='b', head_width=0.05, head_length=0.03, **arrow_params)
                ax.arrow(cell_pos[0], cell_pos[1], del_vx*scale, del_vy*scale,
                    fc=color, ec=color, head_width=max(0.02, acc*0.015), head_length=max(0.02, acc*0.015), **arrow_params)

    return ax

if __name__=="__main__":
    f,ax = plt.subplots()
    odom_2s = [-3/2*np.pi/2, np.pi/4]
    ax.imshow(grid_map, cmap='gray_r', extent=[0.0, n_rows*res, 0.0, n_rows*res], origin='lower')

    ax = plot_arrows(ax, odom_2s[0], 'b', speed=3.0)
    ax = plot_arrows(ax, odom_2s[0], 'r', speed=1.0)
    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    plt.show()
