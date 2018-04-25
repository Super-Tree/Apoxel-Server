
import numpy as np
from tools.utils import bounding_trans_lidar2bv

def voxel_grid(point_cloud,cfg,thread_sum=4):
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
    shifted_coord = bounding_trans_lidar2bv(point_cloud,center)
    np.random.shuffle(point_cloud)
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

    for voxel, point in zip(voxel_index, point_cloud):
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
    # from tools.py_pcd import point_cloud as pcd2np
    # from network.config import cfg
    # fname = '/home/yan/Documents/32_yuanqu_11804041320_1.pcd'
    # lidar_data = pcd2np.from_path(fname)
    # grid_voxel = voxel_grid(lidar_data.pc_data, cfg, thread_sum=cfg.CPU_CNT)
    # outfea = '/home/yan/Documents/deeplearning_lidar/apollo_tensorflow_c_realtime/src/tensorflow_detection/model/feature.txt'
    # outcoordinate = '/home/yan/Documents/deeplearning_lidar/apollo_tensorflow_c_realtime/src/tensorflow_detection/model/coordinate.txt'
    # outnum = '/home/yan/Documents/deeplearning_lidar/apollo_tensorflow_c_realtime/src/tensorflow_detection/model/number.txt'
    # ou = np.array(grid_voxel['feature_buffer'])
    # np.save(outfea,ou)
    # co = grid_voxel['coordinate_buffer']
    # co = np.sort(co,axis=0)
    # print co[:10]
    # cou = np.unique(co, axis=0)
    # np.save(outcoordinate,np.array(grid_voxel['coordinate_buffer']))
    # np.save(outnum,np.array(grid_voxel['number_buffer']))
    pass
