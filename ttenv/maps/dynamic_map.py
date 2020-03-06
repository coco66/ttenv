import numpy as np
import os
import yaml


from ttenv.maps.map_utils import GridMap
import ttenv.util as util

class DynamicMap(GridMap):
    def __init__(self, obj_lib_path, map_path, margin2wall=0.5):
        """
        Parameters
        ---------
        obj_lib_path : A path to a folder where all obstacle objects are stored.
            The folder must contain only obstacle objects in .npy files.
        """
        map_config = yaml.load(open(map_path+".yaml", "r"))
        self.mapdim = map_config['mapdim']
        self.mapres = np.array(map_config['mapres'])
        self.mapmin = np.array(map_config['mapmin'])
        self.mapmax = np.array(map_config['mapmax'])
        self.margin2wall = margin2wall
        self.origin = map_config['origin']
        self.submap_coordinates = [[map_config['submaporigin'][2*i], map_config['submaporigin'][2*i+1]] for i in range(4)]

        self.obstacles_idx = []
        obj_files = os.listdir(obj_lib_path)
        for obj_f in obj_files:
            if '.npy' in obj_f:
                # 2 by #-of-nonzero-cells array.
                self.obstacles_idx.append(np.array(np.nonzero(np.load(os.path.join(obj_lib_path, obj_f)))))
        self.visit_freq_map = None
        self.visit_map = None

    def generate_map(self):
        self.map = np.zeros(self.mapdim)
        map_tmp = np.zeros(self.mapdim)
        chosen_idx = np.random.choice(len(self.obstacles_idx), 4, replace=False)
        for (i, c_id) in enumerate(chosen_idx):
            rot_ang = (np.random.random() - 0.5) * 2 * np.pi

            # rotate in the local frame.
            xy_local = np.matmul([[np.cos(rot_ang), np.sin(rot_ang)],
                            [-np.sin(rot_ang), np.cos(rot_ang)]],
                            self.cell_to_se2_batch(self.obstacles_idx[c_id]) \
                            - 25 * np.reshape(self.mapres, (2,1)))

            xy_global = xy_local + np.reshape(
                                    self.cell_to_se2(self.submap_coordinates[i]),
                                    (2,1))
            cell_global = self.se2_to_cell_batch(xy_global)
            map_tmp[cell_global[0], cell_global[1]] = 1.0

        for r in range(self.mapdim[0]-2):
            if np.sum(map_tmp[r+1,:]) > 0:
                for c in range(self.mapdim[1]-2):
                    if map_tmp[r+1, c+1] == 0:
                        self.map[r+1, c+1] = int(np.sum(map_tmp[r:r+3, c:c+3]) > 3)
                    else:
                        self.map[r+1, c+1] = map_tmp[r+1, c+1]

        self.map_linear = np.squeeze(self.map.astype(np.int8).reshape(-1, 1))

    def se2_to_cell_batch(self, pos):
        """
        pos : 2 by N numpy array with axis 0 for x,y coordinate.
        """
        cell_idx_0 = (pos[0,:] - self.mapmin[0])/self.mapres[0] - 0.5
        cell_idx_1 = (pos[1,:] - self.mapmin[1])/self.mapres[1] - 0.5
        return np.asarray(cell_idx_0.round(), dtype=np.int32), np.asarray(cell_idx_1.round(), dtype=np.int32)

    def cell_to_se2_batch(self, cell_idx):
        """
        cell_idx : 2 by N numpy array with axis 0 is for row and column indices.
        """
        return (cell_idx + 0.5 ) * np.reshape(self.mapres, (2,1)) \
                    + np.reshape(self.mapmin, (2,1))

if __name__ == '__main__':
    print("Test DynamicMap")
    d = DynamicMap(obj_lib_path='maps/lib_obstacles', map_path='maps/dynamic_map')
    for _ in range(10):
        d.generate_map()
        import matplotlib.pyplot as plt
        plt.imshow(d.map, cmap='gray_r')
        plt.show()
    plt.close()
