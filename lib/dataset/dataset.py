import re
import os
import time
import random
import cv2
import sys
sys.path.insert(0, "/home/hexindong/he/Apoxel-Server/lib")
import cPickle
import numpy as np
from tools.printer import red,blue,yellow,darkcyan,blue,green
from tools.timer import Timer
from network.config import cfg
from os.path import join as path_add
from tools.py_pcd import point_cloud as pcd2np
from point2grid import voxel_grid
from numpy import random as np_random

class DataSetTrain(object):  # read txt files one by one
    def __init__(self):
        self.data_path = cfg.DATA_DIR
        # self.folder_list = ['32_yuanqu_11804041320']
        self.folder_list = ['32_yuanqu_11804041320','p3_beihuan_B16_11803221546','32beams_dingdianlukou_2018-03-12-11-02-41', '32_daxuecheng_01803191740', '32_gaosulu_test','xuanchuan']
        self._classes = ['unknown', 'smallMot', 'bigMot', 'nonMot', 'pedestrian']#TODO:declare: there is another label dont care! becareful
        self.type_to_keep = ['smallMot', 'bigMot', 'nonMot', 'pedestrian']
        self.num_classes = len(self._classes)
        self.class_convert = dict(zip(self._classes, xrange(self.num_classes)))
        self.total_roidb = []
        self.filter_roidb = []
        self.percent_train = 0.7
        self.percent_valid = 0.3
        self.train_set, self.valid_set, self.test_set = self.load_dataset()
        self.validing_rois_length = len(self.valid_set)
        self.training_rois_length = len(self.train_set)
        print blue('Dataset initialization has been done successfully.')
        time.sleep(2)

    def load_dataset(self):
        Instruction_cache_file = path_add(self.data_path, 'Instruction_cache_data.pkl')
        train_cache_file = path_add(self.data_path, 'train_cache_data.pkl')
        valid_cache_file = path_add(self.data_path, 'valid_cache_data.pkl')
        test_cache_file = path_add(self.data_path, 'test_cache_data.pkl')
        if os.path.exists(train_cache_file) & os.path.exists(valid_cache_file) & os.path.exists(test_cache_file) & os.path.exists(Instruction_cache_file):
            print blue('Loaded the STi dataset from pkl cache files ...')
            with open(Instruction_cache_file, 'rb') as fid:
                key_points = cPickle.load(fid)
                print yellow('  NOTICE: the groundtruth range is [{}] meters, the label to keep is {},\n          including folders:{},\n  Please verify that meets requirement !' \
                    .format(key_points[0], key_points[1], key_points[2]))
            with open(train_cache_file, 'rb') as fid:
                train_set = cPickle.load(fid)
                print '  train gt set(cnt:{}) loaded from {}'.format(len(train_set), train_cache_file)

            with open(valid_cache_file, 'rb') as fid:
                valid_set = cPickle.load(fid)
                print '  valid gt set(cnt:{}) loaded from {}'.format(len(valid_set), valid_cache_file)

            with open(test_cache_file, 'rb') as fid:
                test_set = cPickle.load(fid)
                print '  test gt set(cnt:{}) loaded from {}'.format(len(test_set), test_cache_file)

            return train_set, valid_set, test_set

        print blue('Prepare the STi dataset for training, please wait ...')
        self.total_roidb = self.load_sti_annotation()
        self.filter_roidb = self.filter(self.total_roidb, self.type_to_keep)
        train_set, valid_set, test_set = self.assign_dataset(self.filter_roidb)  # train,valid percent
        with open(Instruction_cache_file, 'wb') as fid:
            cPickle.dump([cfg.DETECTION_RANGE, self.type_to_keep,self.folder_list], fid, cPickle.HIGHEST_PROTOCOL)
            print yellow('  NOTICE: the groundtruth range is [{}] meters, the label to keep is {},\n          use the dataset:{},\n  Please verify that meets requirement !' \
                .format(cfg.DETECTION_RANGE, self.type_to_keep, self.folder_list))
        with open(train_cache_file, 'wb') as fid:
            cPickle.dump(train_set, fid, cPickle.HIGHEST_PROTOCOL)
            print '  Wrote and loaded train gt roidb(cnt:{}) to {}'.format(len(train_set), train_cache_file)
        with open(valid_cache_file, 'wb') as fid:
            cPickle.dump(valid_set, fid, cPickle.HIGHEST_PROTOCOL)
            print '  Wrote and loaded valid gt roidb(cnt:{}) to {}'.format(len(valid_set), valid_cache_file)
        with open(test_cache_file, 'wb') as fid:
            cPickle.dump(test_set, fid, cPickle.HIGHEST_PROTOCOL)
            print '  Wrote and loaded test gt roidb(cnt:{}) to {}'.format(len(test_set), test_cache_file)

        return train_set, valid_set, test_set

    def load_sti_annotation(self):
        total_box_labels, total_fnames, total_object_labels, total_height_labels = [], [], [], []
        for index, folder in enumerate(self.folder_list):
            print(green('  Process the folder {}'.format(folder)))
            #  TODO:declaration: the result.txt file in shrink_box_label_bk contains illegal number like: "x":"-1.#IND00","y":"-1.#IND00","z":"-1.#IND00"
            libel_fname = path_add(self.data_path, folder, 'label', 'result.txt')
            pixel_libel_folder = path_add(self.data_path, folder, 'label_rect')
            box_label, files_names, one_object_label, one_height_label = [], [], [], []
            with open(libel_fname, 'r') as f:
                frames = f.readlines()
            for idx__,one_frame in enumerate(frames):  # one frame in a series data
                one_frame = one_frame.replace('unknown', '0.0').replace('smallMot', '1.0').replace('bigMot', '2.0') \
                    .replace('nonMot', '3.0').replace('pedestrian', '4.0').replace('dontcare', '0.0')
                object_str = one_frame.translate(None, '\"').split('position:{')[1:]
                label_in_frame = []
                if idx__ % 150 == 0:
                    print ("    Process is going on {}/{} ".format(idx__,len(frames)))
                for obj in object_str:
                    f_str_num = re.findall('[-+]?\d+\.\d+', obj)
                    f_num = map(float, f_str_num)
                    if len(f_num) == 11:  # filter the  wrong type label like   type: position
                        label_in_frame.append(f_num)
                    else:  # toxic label ! shit!
                        print(red('    There is a illegal lbael(length:{}) in result.txt in frame-{} without anything in folder {} and it has been dropped'.format(len(f_num),
                                idx__, folder)))
                        print f_num
                        # print one_frame

                label_in_frame_np = np.array(label_in_frame, dtype=np.float32).reshape(-1, 11)
                if label_in_frame_np.shape[0] == 0:
                    print(red('    There is a empty frame-{} without anything in folder {} and it has been dropped'.format(idx__,folder)))
                    continue
                if len(np.where(label_in_frame_np[:,9]!= 0)[0])==0:
                    print(red('    There is a frame-{} without any object in folder {} and it has been dropped'.format(idx__,folder)))
                    continue
                box_label.append(label_in_frame_np[:, (0, 1, 2, 6, 7, 8, 3, 9)])  # extract the valuable data:x,y,z,l,w,h,theta,type
                files_names.append(self.get_fname_from_label(one_frame))

            print ("    Loading .npy labels ... ")
            for file_ in sorted(os.listdir(pixel_libel_folder), key=lambda name: int(name[0:-4])):
                data_matrix = np.load(path_add(pixel_libel_folder, file_))
                one_object_label.append(data_matrix[:, :, 0:1])  # TODO:check
                one_height_label.append(data_matrix[:, :, 1:2])
            assert len(one_object_label) == len(files_names), "There happens a ERROR when generating dataset in dataset.py"
            total_box_labels.extend(box_label)
            total_fnames.extend(files_names)
            total_object_labels.extend(one_object_label)
            total_height_labels.extend(one_height_label)
            print ("  Completing loading {} is done!  ".format(folder))

        print("  Zip data in one dict ... ")
        return_dataset = [dict({'files_name': total_fnames[i],
                                'boxes_labels': total_box_labels[i],
                                'object_labels': total_object_labels[i],
                                'height_labels': total_height_labels[i]
                                }
                               ) for i in range(len(total_fnames))]

        print("  Total number of frames is {}".format(len(total_fnames)))
        return return_dataset

    def extract_name(self, file_name):
        """32_gaosulu_test_1.pcd
        file_name:string"""
        return int(file_name.split('_')[-1][0:-4])

    def assign_dataset(self, data):
        cnt = len(data)
        test_index = []
        train_index = []

        temp_index = sorted(random.sample(range(cnt), int(cnt * (self.percent_train + self.percent_valid))))
        for i in xrange(cnt):
            if i not in temp_index:
                test_index.append(i)
        valid_index = sorted(random.sample(temp_index, int(cnt * self.percent_valid)))
        for k in temp_index:
            if k not in valid_index:
                train_index.append(k)

        train_roidb = [data[k] for k in train_index]
        valid_roidb = [data[k] for k in valid_index]
        test_roidb = [data[k] for k in test_index]

        return train_roidb, valid_roidb, test_roidb

    def filter(self, data, filter_type):
        """Remove roidb entries that out of bounds and category."""
        # numpy:->   x,y,z,l,w,h,theta,type
        keep_type = [float(self.class_convert[element]) for element in filter_type]

        def is_valid(data_):
            boxes = data_['boxes_labels']
            bool_stack = []
            for idx_ in xrange(boxes.shape[0]):
                bool_stack.append(True if boxes[idx_, 7] in keep_type else False)
            bool_stack_np = np.array(bool_stack, dtype=np.bool)

            bounding = cfg.DETECTION_RANGE
            indice_inside = np.where((boxes[:, 0] >= -bounding) & (boxes[:, 0] <= bounding)
                                     & (boxes[:, 1] >= -bounding) & (boxes[:, 1] <= bounding)
                                     & bool_stack_np
                                     )[0]
            if len(indice_inside) == 0:
                return False, None
            else:
                return True, boxes[indice_inside]

        keep_indice = []
        num = len(data)
        for index in range(num):
            keep, result = is_valid(data[index])
            if keep:
                data[index]['boxes_labels'] = result
                keep_indice.append(index)

        filter_data = [data[k] for k in keep_indice]

        num_after = len(filter_data)
        print '  Filtered {} roidb entries: {} -> {}'.format(num - num_after, num, num_after)
        return filter_data

    def augmentation_of_data(self):
        # Rotation of the image or change the scale
        pass

    def rotation(self, points, rotation):
        # points: numpy array;  translation: moving scalar which should be small
        assert len(points.shape) == 2, "dataset.py->DataSetTrain.rotation: input data has illegal shape {} ".format(points.shape)
        if points.shape[1] >= 3:
            R = np.array([[np.cos(rotation), -np.sin(rotation), 0.],
                          [np.sin(rotation), np.cos(rotation), 0.],
                          [0, 0, 1]], dtype=np.float32)
            points_rot = np.matmul(R, points[:, 0:3].transpose())#TODO:declaration: discard intensity
            return points_rot.transpose()
        elif points.shape[1] == 2:
            R = np.array([[np.cos(rotation), -np.sin(rotation)],
                          [np.sin(rotation), np.cos(rotation)],], dtype=np.float32)
            label_rot = np.matmul(R, points[:, 0:2].transpose())
            return label_rot.transpose()

    def get_minibatch(self, _idx=0, name='train'):
        """Given a roidb, construct a minibatch sampled from it."""
        if name == 'train':
            index_dataset = self.train_set
        elif name == 'valid':
            index_dataset = self.valid_set
        else:
            index_dataset = self.test_set

        fname = index_dataset[_idx]['files_name']

        timer = Timer()
        timer.tic()
        lidar_data = pcd2np.from_path(path_add(self.data_path, fname.split('/')[0], 'pcd', fname.split('/')[1]))
        angel = (np_random.rand() - 0.500) * np.pi * 0.95
        points_rot = self.rotation(lidar_data.pc_data, angel)
        boxes_rot = np.add(index_dataset[_idx]['boxes_labels'],[0.,0.,0.,0.,0.,0.,angel,0.])  # yaw
        category_rot = self.label_rotation(index_dataset[_idx]['object_labels'], degree=angel*57.29578)
        timer.toc()
        time1 = timer.average_time

        timer.tic()
        grid_voxel = voxel_grid(points_rot, cfg, thread_sum=cfg.CPU_CNT)
        timer.toc()
        time2 = timer.average_time

        timer.tic()
        apollo_8feature = np.load(path_add(self.data_path, fname.split('/')[0], 'feature_pcd_name', fname.split('/')[1][0:-4]+'.npy')).reshape(-1, cfg.CUBIC_SIZE[0], cfg.CUBIC_SIZE[1], 8)
        apollo_8feature_rot = self.apollo_feature_rotation(apollo_8feature,degree=angel*57.29578)
        timer.toc()
        time3 = timer.average_time

        blob = dict({'serial_num': fname,
                     'voxel_gen_time': (time1, time2,time3),

                     'lidar3d_data': np.hstack((points_rot,lidar_data.pc_data[:,3:4])),
                     'boxes_labels': boxes_rot,
                     'object_labels': category_rot,

                     'grid_stack': grid_voxel['feature_buffer'],
                     'coord_stack': grid_voxel['coordinate_buffer'],
                     'ptsnum_stack': grid_voxel['number_buffer'],

                     'apollo_8feature': apollo_8feature_rot,
                     })

        return blob

    def label_rotation(self,labels,degree):

        shape_ = labels.shape
        labels = labels.reshape([cfg.CUBIC_SIZE[0],cfg.CUBIC_SIZE[1]])
        # indice_ctr = np.subtract(np.array(np.where(labels[:,:]==1.0),dtype=np.float32).transpose(),np.array([cfg.CUBIC_SIZE[0]/2,cfg.CUBIC_SIZE[1]/2,]))
        # rot_indice = np.around(self.rotation(indice_ctr,theta)).astype(int)
        # indice_min = np.add(rot_indice,np.array([cfg.CUBIC_SIZE[0]/2,cfg.CUBIC_SIZE[1]/2,]))
        # keep_indice= np.logical_and(np.logical_and(indice_min[:,0]<cfg.CUBIC_SIZE[0],indice_min[:,0]>=0),np.logical_and(indice_min[:,1]>=0,indice_min[:,1]<cfg.CUBIC_SIZE[1]))
        # fliter = np.where(keep_indice)[0]
        # filter_indice = indice_min[fliter]
        # return_res = np.zeros([cfg.CUBIC_SIZE[0],cfg.CUBIC_SIZE[1]],np.float32)
        # return_res[filter_indice[:,0],filter_indice[:,1]]=1.0
        matRotation = cv2.getRotationMatrix2D((cfg.CUBIC_SIZE[0] / 2, cfg.CUBIC_SIZE[1] / 2), degree, 1.0)
        label_rot = cv2.warpAffine(labels, matRotation, (cfg.CUBIC_SIZE[0],cfg.CUBIC_SIZE[1]), borderValue=0)

        return label_rot.reshape(shape_)

    def apollo_feature_rotation(self,feature,degree):
        shape_ = feature.shape
        features = feature.reshape([cfg.CUBIC_SIZE[0],cfg.CUBIC_SIZE[1],8])
        matRotation = cv2.getRotationMatrix2D((cfg.CUBIC_SIZE[0] / 2, cfg.CUBIC_SIZE[1] / 2), degree, 1.0)
        feature_rot_part = cv2.warpAffine(features[:,:,(0,1,2,4,5,7)], matRotation, (cfg.CUBIC_SIZE[0],cfg.CUBIC_SIZE[1]), borderValue=0)

        rot_feature = np.dstack((feature_rot_part,features[:,:,(3,6)]))

        return rot_feature.reshape(shape_)

    @staticmethod
    def get_fname_from_label(strings):
        """
        files/32beams_dingdianlukou_2018-03-12-11-02-41/32beams_dingdianlukou_2018-03-12-11-02-41_0.pcd
        files/32_daxuecheng_01803191740/32_daxuecheng_01803191740_1.pcd
        files/32_gaosulu_test/32_gaosulu_test_1.pcd
        files/xuanchuan/xuanchuan_200.pcd
        :param strings:
        :return:
        """

        regulars = ['files/32_gaosulu_test/32_gaosulu_test_\d+.pcd',
                    'files/32_daxuecheng_01803191740/32_daxuecheng_01803191740_\d+.pcd',
                    'files/32beams_dingdianlukou_2018-03-12-11-02-41/32beams_dingdianlukou_2018-03-12-11-02-41_\d+.pcd',
                    'files/xuanchuan/xuanchuan_\d+.pcd',
                    'files/p3_beihuan_B16_11803221546/p3_beihuan_B16_11803221546_\d+.pcd',
                    'files/32_yuanqu_11804041320/32_yuanqu_11804041320_\d+.pcd',
                    ]  # TODO:add more regular
        for i in range(len(regulars)):
            res = re.findall(regulars[i], strings)
            if len(res) != 0:
                if len(res) == 1:
                    return res[0][6:]
                else:
                    print red('File->dataset_sti,function->get_fname_from_label \n  regular expression get more than one qualified file name,string:{}'.format(strings))
                    exit(22)
        print red('File->dataset_sti,function->get_fname_from_label: There is no illegal file name in string: {}'.format(strings))
        exit(23)

    def check_name_get_data(self,pcd_path):
        lidar_data = pcd2np.from_path(pcd_path)
        angel =0  # (np_random.rand() - 0.500) * np.pi * 0.95
        points_rot = self.rotation(lidar_data.pc_data, angel)

        grid_voxel = voxel_grid(points_rot, cfg, thread_sum=cfg.CPU_CNT)
        blob = dict({'lidar3d_data': np.hstack((points_rot, lidar_data.pc_data[:, 3:4])),
                     'grid_stack': grid_voxel['feature_buffer'],
                     'coord_stack': grid_voxel['coordinate_buffer'],
                     'ptsnum_stack': grid_voxel['number_buffer'],
                     })

        return blob


class DataSetTest(object):  # read txt files one by one
    def __init__(self):
        self.data_path = cfg.DATA_DIR
        self.folder = 'demo_test_gaosulu'
        self.test_set = self.load_dataset()
        self.testing_rois_length = len(self.test_set)
        print blue('Dataset {} initialization has been done successfully.'.format(self.testing_rois_length))
        time.sleep(2)

    def extract_name(self, file_name):
        """32_gaosulu_test_1.pcd
        file_name:string"""
        return int(file_name.split('_')[-1][0:-4])

    def traversal_path_with_format(self,path,format_ ='.pcd'):
        f_names = os.listdir(path)
        f_format_names = filter(lambda filename:filename[-4:]==format_,f_names)
        f_format_sort_names = sorted(f_format_names,key=lambda _name: int(_name.split('_')[-1][0:-4]))#,
        return f_format_sort_names

    def load_dataset(self):
        f_names = sorted(os.listdir(os.path.join(cfg.DATA_DIR,self.folder,'pcd')),key=lambda _name: int(_name.split('_')[-1][0:-4]))#,
        name_list = []
        for file_ in f_names:
            name_list.append(os.path.join(cfg.DATA_DIR,self.folder,'pcd',file_))

        return name_list

    def get_minibatch(self, _idx=0):
        """Given a roidb, construct a minibatch sampled from it."""
        index_dataset = self.test_set
        fname = index_dataset[_idx]
        timer = Timer()
        timer.tic()
        lidar_data = pcd2np.from_path(fname)
        angel = 0  # (np_random.rand() - 0.500) * np.pi * 0.9
        points_rot = self.rotation(lidar_data.pc_data,angel)
        timer.toc()
        time1 = timer.average_time

        timer.tic()
        grid_voxel = voxel_grid(points_rot, cfg, thread_sum=cfg.CPU_CNT)
        timer.toc()
        time2 = timer.average_time

        timer.tic()
        apollo_8feature = np.load(path_add(self.data_path, fname.split('/')[-3], 'feature_pcd_name', fname.split('/')[-1][0:-4]+'.npy')).reshape(-1, cfg.CUBIC_SIZE[0], cfg.CUBIC_SIZE[1], 8)
        apollo_8feature_rot = self.apollo_feature_rotation(apollo_8feature, degree=angel*57.29578)
        timer.toc()
        time3 = timer.average_time

        blob = dict({'serial_num': fname.split('/')[-1],
                     'lidar3d_data': lidar_data.pc_data,

                     'grid_stack': grid_voxel['feature_buffer'],
                     'coord_stack': grid_voxel['coordinate_buffer'],
                     'ptsnum_stack': grid_voxel['number_buffer'],

                     'apollo_8feature': apollo_8feature_rot,
                     'voxel_gen_time': (time1, time2, time3)
                     })

        return blob

    def apollo_feature_rotation(self,feature,degree):
        shape_ = feature.shape
        features = feature.reshape([cfg.CUBIC_SIZE[0],cfg.CUBIC_SIZE[1],8])
        matRotation = cv2.getRotationMatrix2D((cfg.CUBIC_SIZE[0] / 2, cfg.CUBIC_SIZE[1] / 2), degree, 1.0)
        feature_rot_part = cv2.warpAffine(features[:,:,(0,1,2,4,5,7)], matRotation, (cfg.CUBIC_SIZE[0],cfg.CUBIC_SIZE[1]), borderValue=0)

        rot_feature = np.dstack((feature_rot_part,features[:,:,(3,6)]))

        return rot_feature.reshape(shape_)

    def rotation(self,points, rotation):
        # points: numpy array;  translation: moving scalar which should be small
        R = np.array([[np.cos(rotation), -np.sin(rotation), 0.],
                      [np.sin(rotation), np.cos(rotation), 0.],
                      [0, 0, 1]], dtype=np.float32)
        points_rot = np.matmul(R, points[:, 0:3].transpose())
        return points_rot.transpose()

    def check_name_get_data(self,pcd_path):
        lidar_data = pcd2np.from_path(pcd_path)
        angel = 0  # (np_random.rand() - 0.500) * np.pi * 0.95
        points_rot = self.rotation(lidar_data.pc_data, angel)

        grid_voxel = voxel_grid(points_rot, cfg, thread_sum=cfg.CPU_CNT)
        blob = dict({'lidar3d_data': np.hstack((points_rot, lidar_data.pc_data[:, 3:4])),
                     'grid_stack': grid_voxel['feature_buffer'],
                     'coord_stack': grid_voxel['coordinate_buffer'],
                     'ptsnum_stack': grid_voxel['number_buffer'],
                     })

        return blob

def init_dataset(arguments):
    """Get an imdb (image database) by name."""
    if arguments.method == 'train':
        return DataSetTrain()
    else:
        return DataSetTest()


if __name__ == '__main__':
    import rospy
    from visualization_msgs.msg import MarkerArray
    from sensor_msgs.msg import PointCloud
    from tools.data_visualize import PointCloud_Gen, Boxes_labels_Gen

    # rospy.init_node('node_labels')
    # label_pub = rospy.Publisher('labels', MarkerArray, queue_size=100)
    # point_pub = rospy.Publisher('points', PointCloud, queue_size=100)
    # rospy.loginfo('Ros begin ...')
    # while True:
    #     blobs = dataset.get_minibatch(idx)
    #     pointcloud = PointCloud_Gen(blobs["lidar3d_data"], frameID='rslidar')
    #     label_box = Boxes_labels_Gen(blobs["boxes_labels"], ns='test_box')
    #     label_pub.publish(label_box)
    #     point_pub.publish(pointcloud)
    #     rospy.loginfo('Send {} frame'.format(idx))
    #     idx += 1

    dataset = DataSetTrain()
    print red('Generate dataset Done!!')

    # name = '/home/hexindong/Videos/Apoxel-Server/RSdata32b/32_gaosulu_test/pcd/32_gaosulu_test_435.pcd'
    # a = dataset.check_name_get_data(name)
    # print(yellow('Convert {} data into pkl file ...').format(dataset.training_rois_length))
    for idx in range(dataset.training_rois_length):
        blobs = dataset.get_minibatch(idx)
        name = blobs['serial_num']
        points = blobs['grid_stack']
        a = 0
        # np.save('/home/hexindong/he/Apoxel-Server/32_yuanqu_11804041320_152.npy',points)
        # exit()
        # data_pkl_name = os.path.join(cfg.DATA_DIR,name.split('/')[0],'data_pkl',name.split('/')[1][:-4]+'.pkl')
        # with open(data_pkl_name, 'wb') as fid:
        #     cPickle.dump(blobs, fid, cPickle.HIGHEST_PROTOCOL)
        #     print '  Wrote data_pkl to {}'.format(data_pkl_name)




