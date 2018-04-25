import numpy as np

def bounding_filter(points,box=(0,0)):

    x_min = box[0] - float(120) / 2
    x_max = box[0] + float(120) / 2
    y_min = box[1] - float(120) / 2
    y_max = box[1] + float(120) / 2
    z_min = -2.0
    z_max = 4.0

    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    f_filt = np.logical_and((x_points > x_min), (x_points < x_max))
    s_filt = np.logical_and((y_points > y_min), (y_points < y_max))
    z_filt = np.logical_and((z_points > z_min), (z_points < z_max))
    fliter = np.logical_and(np.logical_and(f_filt, s_filt), z_filt)
    indice = np.flatnonzero(fliter)
    filter_points = points[indice]

    return filter_points

def bounding_trans_lidar2bv(points,center):
    points = bounding_filter(points)
    points = (points-center)*np.array([-1,-1,1])
    points = points[:,(0,1,2)]#TODO:be careful

    return points

pointcloud = np.fromfile('/home/hexindong/Videos/apoxel-local/000001.bin',dtype=np.float32).reshape(-1,4)

map = np.array([[16,0],[3,4],[10,10],[5,6],[4,8],[4,6],[5,3]],dtype=np.int32)
coordinate = np.array(np.where(map != 0), dtype=np.int32).transpose()
print 'coordinate shape:{}'.format(coordinate.shape)
print coordinate

pointcloud[:, 3] = np.zeros([pointcloud.shape[0]], dtype=np.float32)

center = np.array([640,640, 0], dtype=np.float32)
shifted_coord = bounding_trans_lidar2bv(pointcloud[:,0:3], center)
voxel_size = np.array([1.,1.], dtype=np.float32)
voxel_index = np.floor(pointcloud[:, 0:2] / voxel_size).astype(np.int)


f = np.where(np.array([True if voxel_index[i] in map else False for i in range(voxel_index.shape[0])],dtype=np.bool)==True)[0]
       # pointcloud[map[:,0], map[:,1]] = np.ones([map.shape[0]], dtype=np.float32)
pointcloud[f,3] =np.ones([f.shape[0]], dtype=np.float32)
a =0