import random
import os
import tensorflow as tf
from tools.timer import Timer
from network.config import cfg
from tensorflow.python.client import timeline
from tools.data_visualize import pcd_vispy,vispy_init
DEBUG = False

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

        filename = os.path.join(output_dir, 'CubicNet_epoch_{:d}'.format(iter) + '.ckpt')
        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)

    def training(self, sess, train_writer):
        with tf.name_scope('loss_function'):
            epsilon = tf.constant(value=1e-10)
            scores = tf.reshape(self.net.predicted_map,(-1,2))+epsilon
            labels = tf.reshape(tf.one_hot(tf.reshape(self.net.gt_map,(-1,1)), depth=2,dtype=tf.float32), (-1, 2))
            scores_softmax = tf.nn.softmax(scores)

            # focal loss
            if cfg.TRAIN.FOCAL_LOSS:
                cross_entropy = -tf.reduce_sum(
                    tf.multiply(labels * ((1 - scores_softmax) ** 2) * tf.log(scores_softmax + epsilon), 25), axis=[1])
            else:
                cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(scores_softmax + epsilon), 25), axis=[1])

            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
            loss = cross_entropy_mean

        with tf.name_scope('train_op'):
            global_step = tf.Variable(1, trainable=False, name='Global_Step')
            lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step, 10000, 0.90, name='decay-Lr')
            Optimizer = tf.train.AdamOptimizer(lr)
            var_and_grad = Optimizer.compute_gradients(loss,var_list=tf.trainable_variables())
            train_op = Optimizer.minimize(loss, global_step=global_step)

        with tf.name_scope('debug_board'):
            res_map = tf.cast(tf.reshape(tf.argmax(self.net.predicted_map,axis=3),[-1,640,640,1]),dtype=tf.float32)
            gt_map = tf.reshape(tf.cast(self.net.gt_map,dtype=tf.float32),(-1,640,640,1))
            cnt = tf.shape(self.net.coordinate)[0]
            updates = tf.ones([cnt],dtype=tf.float32)
            input_map = tf.reshape(tf.scatter_nd(self.net.coordinate,updates,shape=[640,640]),(-1,640,640,1))
            tf.summary.image('Pred-Map', input_map)
            tf.summary.image('Pred-Map', res_map)
            tf.summary.image('Gt-Map', gt_map )
            tf.summary.scalar('loss', loss )

        #     tf.summary.scalar('total_loss', loss)
        #     glb_var = tf.trainable_variables()
        #     for i in range(len(glb_var)):
        #         tf.summary.histogram(glb_var[i].name, glb_var[i])
        #     tf.summary.image('theta', self.net.get_output('RNet_theta')[0],max_outputs=50)
            merged = tf.summary.merge_all() #hxd: before the next summary ops
        #
        # with tf.name_scope('epoch_valid'):
        #     epoch_cube_theta = tf.placeholder(dtype=tf.float32)
        #     epoch_cube_theta_sum_op = tf.summary.scalar('valid_los', epoch_cube_theta)

        sess.run(tf.global_variables_initializer())
        if self.args.fine_tune:
            print 'Loading pre-trained model weights from {:s}'.format(self.args.weights)
            self.net.load_weigths(self.args.weights, sess, self.saver)
        trainable_var_for_chk = tf.trainable_variables()#tf.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
        print 'Variables to train: ', trainable_var_for_chk

        timer = Timer()
        if DEBUG:
            pass # TODO: Essential step(before sess.run) for using vispy beacuse of the bug of opengl or tensorflow
            vispy_init()

        training_series = range(self.epoch)  #self.epoch
        for epo_cnt in range(self.args.epoch_iters):
            for data_idx in training_series:  # DO NOT EDIT the "training_series",for the latter shuffle
                global_step = tf.add(global_step,1)
                iter = global_step.eval()  # function "minimize()"will increase global_step
                blobs = self.dataset.get_minibatch(data_idx, 'train')  # get one batch
                feed_dict = {
                    self.net.pc_input: blobs['lidar3d_data'],
                    self.net.feature: blobs['grid_stack'],
                    self.net.coordinate: blobs['coord_stack'],
                    self.net.number: blobs['ptsnum_stack'],
                    self.net.gt_map: blobs['category_labels'],

                }

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                timer.tic()
                res_map_, loss_,merged_, _ = sess.run([res_map, loss, merged,train_op],feed_dict=feed_dict,options=run_options, run_metadata=run_metadata)
                timer.toc()
                if iter % cfg.TRAIN.ITER_DISPLAY == 0:
                    print 'Iter: %d/%d, Serial_num: %s, Speed: %.3fs/iter, Loss: %.3f '%(iter,self.args.epoch_iters * self.epoch, blobs['serial_num'],timer.average_time,loss_)
                    print 'Loading pcd use: {}s, and generating voxel points use: {}s'.format(blobs['voxel_gen_time'][0],blobs['voxel_gen_time'][1])
                if iter % 20 == 0 and cfg.TRAIN.TENSORBOARD:
                    train_writer.add_summary(merged_, iter)
                    pass
                if (iter % 4000==0 and cfg.TRAIN.DEBUG_TIMELINE) or (iter == 400):
                    #chrome://tracing
                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    trace_file = open(cfg.LOG_DIR+'/' +'training-step-'+ str(iter).zfill(7) + '.ctf.json', 'w')
                    trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                    trace_file.close()
                if DEBUG:
                    scan = blobs['lidar3d_data']
                    pcd_vispy(scan, boxes=boxes,name='CubicNet training',index=iter,vis_size=(800, 600),save_img=False,visible=False)

            if cfg.TRAIN.EPOCH_MODEL_SAVE:
                self.snapshot(sess, epo_cnt+1)
                pass
            if cfg.TRAIN.USE_VALID:#TODO: to complete the valid process
                with tf.name_scope('valid_cubic_' + str(epo_cnt + 1)):
                    print 'Valid the net at the end of epoch_{} ...'.format(epo_cnt + 1)
                    valid_loss_total = 0.0
                    for data_idx in range(self.val_epoch):  # self.val_epoch
                        blobs = self.dataset.get_minibatch(data_idx, 'valid')
                        feed_dict_ = {
                            self.net.lidar3d_data: blobs['lidar3d_data'],
                            self.net.lidar_bv_data: blobs['lidar_bv_data'],
                            self.net.im_info: blobs['im_info'],
                            self.net.keep_prob: 0.5,
                            self.net.gt_boxes_bv: blobs['gt_boxes_bv'],
                            self.net.gt_boxes_3d: blobs['gt_boxes_3d'],
                            self.net.gt_boxes_corners: blobs['gt_boxes_corners'],
                            self.net.calib: blobs['calib'],
                        }
                        loss_valid = sess.run(loss, feed_dict=feed_dict_)
                        # train_writer.add_summary(valid, data_idx)

                        valid_loss_total += loss_valid
                        if cfg.TRAIN.VISUAL_VALID and data_idx % 20 == 0:
                            print 'Valid step: {:d}/{:d} , theta_loss = {:.3f}'\
                                  .format(data_idx + 1,self.val_epoch,float(loss_valid))

                        if data_idx % 20 ==0 and cfg.TRAIN.TENSORBOARD:
                            pass
                            # train_writer.add_summary(valid_result_, data_idx/20+epo_cnt*1000)

                valid_summary = tf.summary.merge([epoch_cube_theta_sum_op])
                valid_res = sess.run(valid_summary, feed_dict={epoch_cube_theta:float(valid_loss_total)/self.val_epoch})
                train_writer.add_summary(valid_res, epo_cnt + 1)
                print 'Validation of epoch_{}:theta_loss_total = {:.3f}\n'\
                      .format(epo_cnt + 1,float(valid_loss_total)/self.val_epoch)
            random.shuffle(training_series)  # shuffle the training series
        print 'Training process has done, enjoy every day !'


class TestProcessor(object):
    def __init__(self, network, data_set, args):
        pass


def start_process(network, data_set, args):
    if args.method == 'train':
        net = TrainProcessor(network, data_set, args)
    else:
        net = TestProcessor(network, data_set, args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(cfg.LOG_DIR, sess.graph, max_queue=300)
        net.training(sess, train_writer)