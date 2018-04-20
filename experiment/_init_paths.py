
"""Set up paths for apoxel."""
import sys
import os.path as osp

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)

# Add ROS  to PYTHONPATH
add_path("/opt/ros/indigo/lib/python2.7/dist-packages")
add_path("/opt/ros/indigo/lib")