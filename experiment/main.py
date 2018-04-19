"""
PYTHONPATH=/opt/ros/indigo/lib/python2.7/dist-packages;
PYTHONUNBUFFERED=1;
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
CUDA_VISIBLE_DEVICES = 0
NOTICE: 1.kitti data rotation orientation is clockwise ,contrasting with vispy's counter clockwise
"""
import sys
import argparse
import _init_paths
from dataset.dataset import init_dataset
from network.model import init_network
from apoxel.apoxel import start_process

def parse_args():
    parser = argparse.ArgumentParser(description='Train a ApoxelNet network')
    parser.add_argument('--gpu_id', dest='gpu_id',help=' which gpu to use',
                        default=[0,1,2,3], type=list)
    parser.add_argument('--method', dest='method',help=' train or test',choices=['train', 'test'],
                        default="train", type=str)
    parser.add_argument('--weights', dest='weights',help='which network weights',
                        default=None, type=str)
    parser.add_argument('--epoch_iters', dest='epoch_iters',help='number of iterations to train',
                        default=100, type=int)
    parser.add_argument('--imdb_type', dest='imdb_type',help='dataset to train on(sti/kitti)', choices=['kitti', 'sti'],
                        default='sti', type=str)

    parser.add_argument('--useDemo', dest='useDemo',help='whether use continues frame demo',
                        default="False", type=str)
    parser.add_argument('--fineTune', dest='fineTune',help='whether finetune the existing network weight',
                        default='True', type=str)

    parser.add_argument('--use_demo', dest='use_demo', default=False, type=bool)
    parser.add_argument('--fine_tune', dest='fine_tune', default=True, type=bool)

    # if len(sys.argv) == 1:
    #   parser.print_help()
    #    sys.exit(1)
    return parser.parse_args()


def checkArgs(Args):

    # print('Using config:')
    # pprint.pprint(cfg)
    print "Checking the args ..."

    if Args.fineTune == 'True':
        Args.fine_tune = True
    else:
        Args.fine_tune = False

    if Args.useDemo == 'True':
        Args.use_demo = True
    else:
        Args.use_demo = False

    if Args.method == 'test':
        if Args.weights is None:
            print "  Specify the testing network weights!"
            sys.exit(3)
        else:
            print "  Test the weight: \n {}".format(Args.weights)
    elif Args.fine_tune:
            if Args.weights is None:
                print "  Specify the finetune network weights!"
                sys.exit(4)
            else:
                print "  Finetune the weight:  {}".format(Args.weights)
    else:
            print "  The network will RE-TRAIN from empty ! ! "
    print '  Called with args:',args


if __name__ == '__main__':
    # import numpy as np
    # np.random.seed(5)
    args = parse_args()

    checkArgs(args)

    data_set = init_dataset(args)  # load  dataset

    network = init_network(args)  # load network model

    start_process(network, data_set, args)
