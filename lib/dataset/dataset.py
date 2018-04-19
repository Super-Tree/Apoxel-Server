import cv2
import re
import os
import time
import random

import cPickle
import numpy as np
from tools.timer import Timer
from network.config import cfg
from os.path import join as path_add
from tools.py_pcd import point_cloud as pcd2np
from point2grid import voxel_grid

class DataSetTrain(object):  # read txt files one by one
    def __init__(self):
        self.data_path = cfg.DATA_DIR
        # self.folder_list = ['170818-1743-LM120', '170825-1708-LM120', '170829-1743-LM120', '170829-1744-LM120',
        #                     '1180254121101']
        self.folder_list = ['32_gaosulu_test']
        self._classes = ['unknown', 'smallMot', 'bigMot', 'nonMot', 'pedestrian']
        self.type_to_keep = ['smallMot', 'bigMot', 'nonMot', 'pedestrian']
        self.num_classes = len(self._classes)
        self.class_convert = dict(zip(self._classes, xrange(self.num_classes)))
        self.total_roidb = []
        self.filter_roidb = []
        self.percent_train = 0.66
        self.percent_valid = 0.26
        self.train_set, self.valid_set, self.test_set = self.load_dataset()
        self.validing_rois_length = len(self.valid_set)
        self.training_rois_length = len(self.train_set)
        print 'Dataset initialization has been done successfully.'
        time.sleep(1)

    def load_dataset(self):
        Instruction_cache_file = path_add(self.data_path, 'Instruction_cache_data.pkl')
        train_cache_file = path_add(self.data_path, 'train_cache_data.pkl')
        valid_cache_file = path_add(self.data_path, 'valid_cache_data.pkl')
        test_cache_file = path_add(self.data_path, 'test_cache_data.pkl')
        if os.path.exists(train_cache_file) & os.path.exists(valid_cache_file) & os.path.exists(
                test_cache_file) & os.path.exists(Instruction_cache_file):
            print 'Loaded the STi dataset from pkl cache files ...'
            with open(Instruction_cache_file, 'rb') as fid:
                key_points = cPickle.load(fid)
                print ' \033[31;1m NOTICE: the groundtruth range is [{}] meters, the label to keep is {} ,please verify that meets requirement ! \033[0m' \
                    .format(key_points[0], key_points[1], )
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

        print 'Prepare the STi dataset for training, please wait ...'
        self.total_roidb = self.load_sti_annotation()
        self.filter_roidb = self.filter(self.total_roidb, self.type_to_keep)
        train_set, valid_set, test_set = self.assign_dataset(self.filter_roidb)  # train,valid percent
        with open(Instruction_cache_file, 'wb') as fid:
            cPickle.dump([cfg.DETECTION_RANGE, self.type_to_keep], fid, cPickle.HIGHEST_PROTOCOL)
            print '  NOTICE: the groundtruth range is [{}] meters, the label to keep is {} ,please verify that meets requirement !' \
                .format(self.type_to_keep[0], self.type_to_keep[1], )

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
        total_box_labels, total_fnames, total_category_labels, total_confidence_labels = [], [], [], []
        for index, folder in enumerate(self.folder_list):
            libel_fname = path_add(self.data_path, folder, 'shrink_box_label_bk', 'result.txt')
            pixel_libel_folder = path_add(self.data_path, folder, 'label_rect')
            box_label, files_names, one_category_label, one_confidence_label = [], [], [], []
            with open(libel_fname, 'r') as f:
                frames = f.readlines()
            for one_frame in frames:  # one frame in a series data
                one_frame = one_frame.replace('unknown', '0.0').replace('smallMot', '1.0').replace('bigMot',
                                                                                                   '2.0').replace(
                    'nonMot', '3.0').replace('pedestrian', '4.0')
                object_str = one_frame.translate(None, '\"').split('position:{')[1:]
                label_in_frame = []
                for obj in object_str:
                    f_str_num = re.findall('[-+]?\d+\.\d+', obj)
                    f_num = map(float, f_str_num)
                    if len(f_num) == 11:  # filter the  wrong type label like   type: position
                        label_in_frame.append(f_num)
                label_in_frame_np = np.array(label_in_frame, dtype=np.float32).reshape(-1, 11)
                if label_in_frame_np.shape[0] == 0:
                    continue
                box_label.append(label_in_frame_np[:, (0, 1, 2, 6, 7, 8, 3, 9)])  # extract the valuable data:x,y,z,l,w,h,theta,type
                files_names.append(self.get_fname_from_label(one_frame))

            for file_ in sorted(os.listdir(pixel_libel_folder), key=lambda name: int(name[0:-4])):
                data_matrix = np.load(path_add(pixel_libel_folder, file_))
                one_category_label.append(data_matrix[:, :, 0:1])  # TODO:check
                one_confidence_label.append(data_matrix[:, :, 6:7])

            assert len(one_category_label) == len(files_names), "There happens a ERROR when generating dataset in dataset.py"
            total_box_labels.extend(box_label)
            total_fnames.extend(files_names)
            total_category_labels.extend(one_category_label)
            total_confidence_labels.extend(one_confidence_label)
        return_dataset = [dict({'files_name': total_fnames[i],
                                'boxes_labels': total_box_labels[i],
                                'category_labels': total_category_labels[i],
                                'confidence_labels': total_confidence_labels[i]})
                          for i in range(len(total_fnames))]
        print("  Total number of frames is {}".format(len(total_fnames)))
        return return_dataset

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
        print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after, num, num_after)
        return filter_data

    def augmentation_of_data(self):
        # Rotation of the image or change the scale
        pass

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
        timer.toc()
        time1 = timer.average_time

        timer.tic()
        grid_voxel = voxel_grid(lidar_data.pc_data[:,0:3],cfg,thread_sum=cfg.CPU_CNT)
        timer.toc()
        time2 = timer.average_time

        blob = dict({'lidar3d_data': lidar_data.pc_data,
                     'serial_num': fname,
                     'boxes_labels': index_dataset[_idx]['boxes_labels'],
                     'category_labels': index_dataset[_idx]['category_labels'],
                     'confidence_labels': index_dataset[_idx]['confidence_labels'],

                     'grid_stack': grid_voxel['feature_buffer'],
                     'coord_stack': grid_voxel['coordinate_buffer'],
                     'ptsnum_stack': grid_voxel['number_buffer'],

                     'voxel_gen_time':(time1,time2)
                     })

        return blob

    @staticmethod
    def get_fname_from_label(strings):
        # "32_gaosulu_test/32_gaosulu_test_1.pcd"
        regulars = ['files/32_gaosulu_test/32_gaosulu_test_\d+.pcd']#TODO:add more regular
        for i in range(len(regulars)):
            res = re.findall(regulars[i], strings)
            if len(res) != 0:
                if len(res) == 1:
                    return res[0][6:]
                else:
                    print'File: dataset_sti,function:get_fname_from_label \n  regular expression get more than one qualified file name'
                    exit(23)

class DataSetTest(object):  # read txt files one by one
    def __init__(self):
        self.data_path = cfg.DATA_DIR

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

    rospy.init_node('node_labels')
    label_pub = rospy.Publisher('labels', MarkerArray, queue_size=100)
    point_pub = rospy.Publisher('points', PointCloud, queue_size=100)
    rospy.loginfo('Ros begin ...')

    dataset = DataSetTrain()
    idx = 0
    while True:
        blobs = dataset.get_minibatch(idx)
        pointcloud = PointCloud_Gen(blobs["lidar3d_data"], frameID='rslidar')
        label_box = Boxes_labels_Gen(blobs["boxes_labels"], ns='test_box')
        label_pub.publish(label_box)
        point_pub.publish(pointcloud)
        rospy.loginfo('Send {} frame'.format(idx))
        idx += 1
