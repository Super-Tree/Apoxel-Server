
import numpy as np
from network.config import cfg


def voxel_grid(point_cloud,cfg,thread_sum=4):
    from tools.utils import bound_trans_lidar2bv

    # Input:
    #   (N, 3):only x,y,z
    # Output:
    #   voxel_dict
    from tools.data_visualize import pcd_vispy

    keep = np.where(np.isfinite(point_cloud[:,0]))[0]
    point_cloud=point_cloud[keep,0:3]
    max_point_number = cfg.VOXEL_POINT_COUNT
    # pcd_vispy(point_cloud)
    center = np.array([cfg.DETECTION_RANGE, cfg.DETECTION_RANGE,0],dtype=np.float32)
    shifted_coord = bound_trans_lidar2bv(point_cloud, center)
    np.random.shuffle(shifted_coord)
    # pcd_vispy(shifted_coord)

    voxel_size = np.array(cfg.CUBIC_RES, dtype=np.float32)
    voxel_index = np.floor(shifted_coord[:,0:2]/ voxel_size).astype(np.int)

    # [K, 3] coordinate buffer as described in the paper
    coordinate_buffer = np.unique(voxel_index, axis=0)

    K = len(coordinate_buffer)
    T = max_point_number

    # [K, 1] store number of points in each voxel grid
    number_buffer = np.zeros(shape=(K), dtype=np.int64)

    # [K, T, 6] feature buffer as described in the paper
    feature_buffer = np.zeros(shape=(K, T, 6), dtype=np.float32)

    # build a reverse index for coordinate buffer
    index_buffer = {}
    for i in range(K):
        index_buffer[tuple(coordinate_buffer[i])] = i

    for voxel, point in zip(voxel_index, shifted_coord):
        index = index_buffer[tuple(voxel)]
        number = number_buffer[index]
        if number < T:
            feature_buffer[index, number, :3] = point
            number_buffer[index] += 1

    feature_buffer[:, :, -3:] = feature_buffer[:, :, :3] - feature_buffer[:, :, :3].sum(axis=1, keepdims=True) / number_buffer.reshape(K, 1, 1)#TODO:to learn from it

    voxel_dict = {'feature_buffer': feature_buffer,
                  'coordinate_buffer': coordinate_buffer,
                  'number_buffer': number_buffer}
    return voxel_dict


if __name__ == '__main__':
    from dataset import DataSetTrain

    dataset = DataSetTrain()
    name = '/home/hexindong/Videos/Apoxel-Server/RSdata32b/32_gaosulu_test/pcd/32_gaosulu_test_435.pcd'
    data = dataset.check_name_get_data(name)
    points = data['lidar3d_data']

    grid = voxel_grid(points,cfg)
    pass
