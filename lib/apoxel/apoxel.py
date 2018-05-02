import random
import os
import numpy as np
import tensorflow as tf
from tools.timer import Timer
from network.config import cfg
from tensorflow.python.client import timeline
from tools.data_visualize import pcd_vispy, vispy_init

DEBUG = False
SUFFIX ='M7-E44_gaosulu_pointcnt_50'

class TrainProcessor(object):
    def __init__(self, network, data_set, args):
        self.saver = tf.train.Saver(max_to_keep=100)
        self.net = network
        self.dataset = data_set
        self.args = args
        self.random_folder = cfg.RANDOM_STR
        self.epoch = self.dataset.training_rois_length
        self.val_epoch = self.dataset.validing_rois_length

    def snapshot(self, sess, iter=None):
        output_dir = os.path.join(cfg.ROOT_DIR, 'output', self.random_folder)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = os.path.join(output_dir, 'Apoxel_Epoch_{:d}'.format(iter) + '.ckpt')
        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)
    #
    # def save_res_as_pcd(self,pointcloud,map,fname):
    #     import numpy as np
    #     from tools.utils import bound_trans_lidar2bv
    #
    #     map = map.reshape(cfg.CUBIC_RES[0], cfg.CUBIC_RES[1])
    #     coordinate = np.array(np.where(map!=0),dtype=np.int32).transpose()
    #     print 'coordinate shape:{}'.format(coordinate.shape)
    #     print coordinate
    #
    #     pointcloud[:,3]=np.zeros([pointcloud.shape[0]],dtype=np.float32)
    #
    #     center = np.array([cfg.DETECTION_RANGE, cfg.DETECTION_RANGE, 0], dtype=np.float32)
    #     shifted_coord = bound_trans_lidar2bv(pointcloud[:, 0:3], center)
    #     voxel_size = np.array(cfg.CUBIC_RES, dtype=np.float32)
    #     voxel_index = np.floor(shifted_coord[:, 0:2] / voxel_size).astype(np.int)
    #
    #     keep = np.where(np.array([True if voxel_index[i] in map else False for i in range(voxel_index.shape[0])],
    #                           dtype=np.bool) == True)[0]
    #     # pointcloud[map[:,0], map[:,1]] = np.ones([map.shape[0]], dtype=np.float32)
    #     pointcloud[keep, 3] = np.ones([keep.shape[0]], dtype=np.float32)
    #     print 'Done'

    def processor(self, sess, train_writer):
        with tf.name_scope('loss_design'):
            epsilon = tf.constant(value=1e-10)
            scores = tf.reshape(self.net.predicted_map, (-1, 2)) + epsilon
            labels = tf.reshape(tf.one_hot(tf.reshape(self.net.gt_map, (-1, 1)), depth=2, dtype=tf.float32), (-1, 2))
            scores_softmax = tf.nn.softmax(scores)
            # focal loss
            # balance = np.array([1,1],dtype=np.float32)
            balance = 50.0
            if cfg.TRAIN.FOCAL_LOSS:
                # TODO:add +- balance
                cross_entropy = -tf.reduce_sum(tf.multiply(labels * ((1 - scores_softmax) ** 3) * tf.log(scores_softmax + epsilon), balance), axis=[1])
            else:
                pass
                cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(scores_softmax + epsilon), balance), axis=[1])

            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
            loss = cross_entropy_mean

        with tf.name_scope('train_op'):
            global_step = tf.Variable(1, trainable=False, name='Global_Step')
            lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step, 10000, 0.90, name='decay-Lr')
            Optimizer = tf.train.AdamOptimizer(lr)
            # var_and_grad = Optimizer.compute_gradients(loss, var_list=tf.trainable_variables())
            train_op = Optimizer.minimize(loss, global_step=global_step)

        with tf.name_scope('TrainingBoard'):
            res_map = tf.cast(tf.reshape(tf.argmax(self.net.predicted_map, axis=3), [-1, 640, 640, 1]),dtype=tf.float32)
            gt_map = tf.reshape(tf.cast(self.net.gt_map, dtype=tf.float32), (-1, 640, 640, 1))
            cnt = tf.shape(self.net.coordinate)[0]
            updates = tf.ones([cnt], dtype=tf.float32)
            input_map = tf.reshape(tf.scatter_nd(self.net.coordinate, updates, shape=[640, 640]), (-1, 640, 640, 1))
            tf.summary.image('InputData', input_map)
            tf.summary.image('PredMap', res_map)
            tf.summary.image('GtMap', gt_map)

            # apollo_feature = tf.transposes(self.net.apollo_8feature, perm=[3, 1, 2, 0])
            # tf.summary.image('ApolloFeature', apollo_feature, max_outputs=8)

            tf.summary.scalar('TrainLoss', loss)

            glb_var = tf.trainable_variables()
            for i in range(len(glb_var)):
                tf.summary.histogram(glb_var[i].name, glb_var[i])

            main_merged = tf.summary.merge_all()  # hxd: before the next summary ops

        with tf.name_scope('TrainingMonitor'):
            epoch_valid_loss = tf.placeholder(dtype=tf.float32)
            epoch_valid_loss_sum_op = tf.summary.scalar('epoch_los', epoch_valid_loss)

        sess.run(tf.global_variables_initializer())
        if self.args.fine_tune:
            print 'Loading pre-trained model weights from {:s}'.format(self.args.weights)
            self.net.load_weigths(self.args.weights, sess, self.saver)
        # trainable_var_for_chk = tf.trainable_variables()  # tf.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
        # print 'Variables to train: ', trainable_var_for_chk

        timer = Timer()
        if DEBUG:
            pass  # TODO: Essential step(before sess.run) for using vispy beacuse of the bug of opengl or tensorflow
            vispy_init()

        training_series = range(self.epoch)  # self.epoch
        for epo_cnt in range(self.args.epoch_iters):
            for data_idx in training_series:  # DO NOT EDIT the "training_series",for the latter shuffle
                iter = global_step.eval()  # function "minimize()"will increase global_step
                blobs = self.dataset.get_minibatch(data_idx, 'train')  # get one batch
                feed_dict = {
                    self.net.pc_input: blobs['lidar3d_data'],
                    self.net.voxel_feature: blobs['grid_stack'],
                    self.net.coordinate: blobs['coord_stack'],
                    self.net.number: blobs['ptsnum_stack'],
                    self.net.gt_map: blobs['object_labels'],
                    self.net.apollo_8feature: blobs['apollo_8feature'],
                }

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                timer.tic()
                res_map_, loss_, merged_, _ = sess.run([res_map, loss, main_merged, train_op], feed_dict=feed_dict,
                                                       options=run_options, run_metadata=run_metadata)
                timer.toc()

                if iter % cfg.TRAIN.ITER_DISPLAY == 0:
                    print 'Iter: %d/%d, Serial_num: %s, Speed: %.3fs/iter, Loss: %.3f ' % (
                    iter, self.args.epoch_iters * self.epoch, blobs['serial_num'], timer.average_time, loss_)
                    print 'Loading pcd use: {:.3}s, and generating voxel points use: {:.3}s'.format(
                        blobs['voxel_gen_time'][0], blobs['voxel_gen_time'][1])
                if iter % 20 == 0 and cfg.TRAIN.TENSORBOARD:
                    train_writer.add_summary(merged_, iter)
                    # train_writer.add_run_metadata(run_metadata, 'step%03d' % iter,iter)
                    pass
                if (iter % 4000 == 0 and cfg.TRAIN.DEBUG_TIMELINE) or (iter == 300):
                    # chrome://tracing
                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    trace_file = open(cfg.LOG_DIR + '/' + 'training-step-' + str(iter).zfill(7) + '.ctf.json', 'w')
                    trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                    trace_file.close()
                if DEBUG:
                    scan = blobs['lidar3d_data']
                    pcd_vispy(scan, boxes=None, name='CubicNet training', index=iter, vis_size=(800, 600),
                              save_img=False, visible=False)

            if cfg.TRAIN.EPOCH_MODEL_SAVE:
                self.snapshot(sess, epo_cnt + 1)
                pass
            if cfg.TRAIN.USE_VALID:  # TODO: to complete the valid process
                with tf.name_scope('Validations_' + str(epo_cnt + 1)):

                    epoch_valid_gt_sum_op = tf.summary.image('valid_gt', gt_map)
                    epoch_valid_det_sum_op = tf.summary.image('valid_predict', res_map)
                    epoch_valid_input_sum_op = tf.summary.image('valid_input_data', input_map)

                    valid_image_summary = tf.summary.merge([epoch_valid_gt_sum_op, epoch_valid_det_sum_op, epoch_valid_input_sum_op])

                    print 'Valid the net at the end of epoch_{} ...'.format(epo_cnt + 1)
                    valid_loss_total = 0.0
                    for data_idx in range(self.val_epoch):  # self.val_epoch
                        blobs = self.dataset.get_minibatch(data_idx, 'valid')
                        feed_dict_ = {
                            self.net.pc_input: blobs['lidar3d_data'],
                            self.net.voxel_feature: blobs['grid_stack'],
                            self.net.coordinate: blobs['coord_stack'],
                            self.net.number: blobs['ptsnum_stack'],
                            self.net.gt_map: blobs['object_labels'],
                            self.net.apollo_8feature: blobs['apollo_8feature'],

                        }
                        valid_sum_,loss_valid_ = sess.run([valid_image_summary,loss], feed_dict=feed_dict_)
                        # train_writer.add_summary(valid, data_idx)

                        valid_loss_total += loss_valid_
                        if cfg.TRAIN.VISUAL_VALID and data_idx % 20 == 0:
                            pass
                            print 'Valid step: {:d}/{:d} , theta_loss = {:.3f}'.format(data_idx + 1, self.val_epoch, float(loss_valid_))

                        if data_idx % 10 == 0 and cfg.TRAIN.TENSORBOARD:
                            pass
                            train_writer.add_summary(valid_sum_, data_idx)

                valid_summary = tf.summary.merge([epoch_valid_loss_sum_op])
                valid_res = sess.run(valid_summary,feed_dict={epoch_valid_loss: float(valid_loss_total) / self.val_epoch})
                train_writer.add_summary(valid_res, epo_cnt + 1)
                print 'Validation of epoch_{}:theta_loss_total = {:.3f}\n'.format(epo_cnt + 1, float(valid_loss_total) / self.val_epoch)
            random.shuffle(training_series)  # shuffle the training series
        print 'Training process has done, enjoy every day !'


class TestProcessor(object):
    def __init__(self, network, data_set, args):
        self.saver = tf.train.Saver(max_to_keep=100)
        self.net = network
        self.dataset = data_set
        self.args = args
        self.random_folder = cfg.RANDOM_STR
        self.epoch = self.dataset.testing_rois_length

    def save_res_as_pcd(self,pointcloud,map,save_path,folder,idx_):
        import numpy as np
        from tools.utils import bound_trans_lidar2bv
        from tools.py_pcd import point_cloud

        map = map.reshape(cfg.CUBIC_SIZE[0], cfg.CUBIC_SIZE[1])
        map_coordinate = np.array(np.where(map!=0),dtype=np.int32).transpose()
        # print 'coordinate shape:{}'.format(coordinate.shape)
        # print coordinate
        pointcloud[:,3]=np.zeros([pointcloud.shape[0]],dtype=np.float32)

        center = np.array([cfg.DETECTION_RANGE, cfg.DETECTION_RANGE, 0], dtype=np.float32)
        shifted_coord = bound_trans_lidar2bv(pointcloud[:, 0:3], center)
        voxel_size = np.array(cfg.CUBIC_RES, dtype=np.float32)
        voxel_index = np.floor(shifted_coord[:, 0:2] / voxel_size).astype(np.int)

        keep = np.where(np.array([True if voxel_index[i] in map_coordinate else False for i in range(voxel_index.shape[0])],
                              dtype=np.bool) == True)[0]
        # pointcloud[map[:,0], map[:,1]] = np.ones([map.shape[0]], dtype=np.float32)
        pointcloud[keep, 3] = np.ones([keep.shape[0]], dtype=np.float32)*125
        pointcloud =pointcloud[keep]
        cnt = pointcloud.shape[0]
        metadata = dict({'count': [1, 1, 1, 1],
                         'data': 'ascii',
                         'fields': ['x', 'y', 'z', 'intensity'],
                         'height': 1,
                         'points': cnt,
                         'size': [4, 4, 4, 4],
                         'type': ['F', 'F', 'F', 'F'],
                         'version': '0.7',
                         'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                         'width': cnt,
                         })

        pointcloud = point_cloud(metadata, pointcloud)
        pcd_name = os.path.join(save_path,folder, str(idx_).zfill(6) + '.pcd')
        pointcloud.save(pcd_name)

        print 'Done,save file to {}'.format(pcd_name)

    def save_pred_as_npy(self,map,save_path,folder,idx_):
        import numpy as np
        if not os.path.exists(os.path.join(save_path,folder)):
            os.mkdir(os.path.join(save_path,folder))
        npy_name = os.path.join(save_path,folder,str(idx_).zfill(6) + '.npy')
        np.save(npy_name,map)
        print 'Done,save file to {}'.format(npy_name)

    def processor(self, sess, train_writer):
        with tf.name_scope('test_debug_board'):
            res_map = tf.cast(tf.reshape(tf.argmax(self.net.predicted_map, axis=3), [-1, 640, 640, 1]),dtype=tf.float32)
            cnt = tf.shape(self.net.coordinate)[0]
            updates = tf.ones([cnt], dtype=tf.float32)
            input_map = tf.reshape(tf.scatter_nd(self.net.coordinate, updates, shape=[640, 640]), (-1, 640, 640, 1))

            tf.summary.image('Input_Data', input_map)
            tf.summary.image('Pred-Map', res_map)

            merged = tf.summary.merge_all()  # hxd: before the next summary ops

        sess.run(tf.global_variables_initializer())
        if self.args.fine_tune:
            print 'Loading pre-trained model weights from {:s}'.format(self.args.weights)
            self.net.load_weigths(self.args.weights, sess, self.saver)

        timer = Timer()
        training_series = range(self.epoch)
        for data_idx in training_series:  # DO NOT EDIT the "training_series",for the latter shuffle
            blobs = self.dataset.get_minibatch(data_idx)  # get one batch
            feed_dict = {
                # self.net.pc_input: blobs['lidar3d_data'],
                self.net.voxel_feature: blobs['grid_stack'],
                self.net.coordinate: blobs['coord_stack'],
                self.net.number: blobs['ptsnum_stack'],
                self.net.apollo_8feature: blobs['apollo_8feature'],
            }

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            timer.tic()
            res_map_,merged_ = sess.run([res_map,merged], feed_dict=feed_dict,options=run_options, run_metadata=run_metadata)
            timer.toc()
            self.save_pred_as_npy(res_map_, save_path=cfg.TEST_RESULT,folder=SUFFIX,idx_=data_idx)

            if data_idx % cfg.TEST.ITER_DISPLAY == 0:
                print 'Iter: %d/%d, Serial_num: %s, Speed: %.3fs/iter' % (data_idx, self.epoch, blobs['serial_num'], timer.average_time)
                print 'Loading pcd use: {:.3}s, and generating voxel points use: {:.3}s'.format(blobs['voxel_gen_time'][0], blobs['voxel_gen_time'][1])
            if data_idx % 1 == 0 and cfg.TRAIN.TENSORBOARD:
                train_writer.add_summary(merged_, data_idx)
                pass
            if (data_idx+1) % 200 == 0 and cfg.TEST.DEBUG_TIMELINE:
                # chrome://tracing
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open(cfg.LOG_DIR + '/' + 'testing-step-' + str(data_idx).zfill(7) + '.ctf.json', 'w')
                trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                trace_file.close()

        print 'Training process has done, enjoy every day !'


def start_process(network, data_set, args):
    if args.method == 'train':
        net = TrainProcessor(network, data_set, args)
        suffix = None
    else:
        net = TestProcessor(network, data_set, args)
        suffix = SUFFIX

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(cfg.LOG_DIR, sess.graph, max_queue=300,filename_suffix=suffix)
        net.processor(sess, train_writer)


if __name__ =='__main__':
    pass