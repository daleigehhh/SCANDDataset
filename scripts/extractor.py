import rosbag
import numpy as np
import pickle
import glob
import os
import torch
from omegaconf import OmegaConf
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, PointCloud2
import ros_numpy
from tf.transformations import euler_from_quaternion
from concurrent.futures import ProcessPoolExecutor
import time
from typing import Tuple, Dict, Any, Optional
from scripts.local_occ_grid_map import LocalMap
from scripts.hooks import SaveBackViewHook, SaveFrontViewHook, SavePoseControlHook


class ROSBagExtractor(object):
    def __init__(self,
                 base_path: str,
                 target: str,
                 config: str,
                 synchronize: bool = True,
                 enable_reverse_aug: bool = True):
        assert base_path and target, "Invalid Path"
        self.base_path = os.path.abspath(base_path)
        self.bag_list = [os.path.join(self.base_path, i) for i in
                         glob.glob(os.path.join(self.base_path,'*.bag'))]
        self.target = target
        self.enable_reverse_aug = enable_reverse_aug
        os.makedirs(self.target, exist_ok=True)

        # Load configs
        self.config = OmegaConf.load(config)
        odom_topic = self.config['odom']
        scan_topic = self.config['scan']
        # pc_topic = self.config['point_cloud']['topic']
        self.topics = [odom_topic, scan_topic]
        self.device = self.config['device']
        # Flag for extract synchronized messages
        self.synchronize = synchronize

        # Const
        self.batch_size = 1

    def extract_all(self) -> None:
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=8) as executor:
            ret = executor.map(self.extract, self.bag_list)
            blobs, count = zip(*executor.map(self.extract, self.bag_list))
        with open(os.path.join(self.target, 'data.pkl'), 'wb') as f:
            pickle.dump(blobs, f)
        end_time = time.time()
        count = sum(count)
        self.summary(start_time, end_time, count)

    def extract(self, bag_path: str) -> Tuple[Dict[str, np.array], Optional[int]]:
        data_no_sync = self._extract_one_bag_no_syn(bag_path)
        if self.synchronize is False:
            return data_no_sync, None
        else:
            data_synced, cnt = self._synchornize(data_no_sync)
            return data_synced, cnt

    @SavePoseControlHook('viz/statistics')
    def _synchornize(self,
                     data: Dict[str, np.array]) -> Tuple[Dict[str, np.array], int]:
        bag_name = data['bag_name']
        timestamp_odom = data['timestamp_odom'][:, None]
        timestamp_scan = data['timestamp_scan'][None, :]
        distance  = np.square((timestamp_odom - timestamp_scan))
        argmin = np.argmin(distance, axis=0)
        velocity = data['velocity'][argmin]
        pose = data['pose'][argmin]
        scan = data['scan']
        self.seq_len = len(scan)
        assert self.seq_len == len(argmin)
        count = self.seq_len
        mapper_kwargs = self.config['mapper']
        mapper_kwargs['size'] = [self.batch_size, self.seq_len]
        binary_maps, _ = self._get_map(bag_name, scan, mapper_kwargs)
        if self.enable_reverse_aug:
            reversed_maps, _ = self._reverse_augmentation(bag_name, scan, mapper_kwargs)
            binary_maps = np.concatenate([binary_maps, reversed_maps], axis=0)
            velocity = np.concatenate([velocity, np.flip(velocity*-1, 0)], axis=0)
            pose = np.concatenate([pose, np.flip(pose, 0)], axis=0)
            count = len(binary_maps)
        data_synced = {
            'bag_name': bag_name,
            'velocity': velocity,
            'pose': pose,
            'maps': binary_maps,
        }
        print(f'{bag_name}: [synced sensor data] {len(velocity)} pose-vel pairs,'
              f' {count} frames')
        return data_synced, count

    def _extract_one_bag_no_syn(self, bag_path: str) -> Dict[str, np.array]:
        velocity_data, pose_data, scan_data = [], [], []
        timestamp_odom, timestamp_scan = [], []
        bag_name = bag_path.split('/')[-1]
        with rosbag.Bag(bag_path, 'r') as data_bag:
            for topic, msg, t in data_bag.read_messages(topics=[topic for topic in self.topics if topic]):
                if 'Odometry' in msg._type:
                    velocity, pose, _ = self._extract_pose_vel(bag_name, msg)
                    velocity_data.append(velocity)
                    pose_data.append(pose)
                    timestamp_odom.append(t.to_nsec() / 1e8) # For numerical stability
                elif 'LaserScan' in msg._type:
                    scan = self._extract_scan(msg)
                    scan_data.append(scan)
                    timestamp_scan.append(t.to_nsec() / 1e8)
                elif 'PointCloud2' in msg._type:
                    scan = self._extract_pc2(msg)
                    scan_data.append(scan)
                    timestamp_scan.append(t.to_nsec() / 1e8)
                else:
                    raise NotImplementedError
        bag_name = bag_path.split('/')[-1]
        velocity_data = np.stack(velocity_data, axis=0)
        pose_data = np.stack(pose_data, axis=0)
        scan_data = np.stack(scan_data, axis=0)
        timestamp_scan = np.stack(timestamp_scan, axis=0)
        timestamp_odom = np.stack(timestamp_odom, axis=0)
        data = {
            'bag_name': bag_name,
            'velocity': velocity_data,
            'pose': pose_data,
            'scan': scan_data,
            'timestamp_odom': timestamp_odom,
            'timestamp_scan': timestamp_scan
        }
        print(f'{bag_name}: [raw sensor data] {len(velocity_data)} pose-vel pairs,'
              f' {len(scan_data)} laser_scan frames')
        return data

    def _extract_pose_vel(self,
                          bag_name: str,
                          msg: Odometry) -> Tuple[np.array, ...]:
        '''
            Extract pose and velocity from a single Odometry message
        '''
        pose = msg.pose.pose
        twist = msg.twist.twist
        # Get Yaw from ROS-style Quaternion
        _, _, yaw = euler_from_quaternion([pose.orientation.x, pose.orientation.y,
                                                     pose.orientation.z, pose.orientation.w])
        # Non-holonomic constraint wheeled robot only has 2 dim velocity
        velocity = np.array([twist.linear.x, twist.angular.z])
        # For robot navigate in a plane, position is (x, y)
        pose = np.array([pose.position.x, pose.position.y, yaw])
        # Validation
        pose[np.bitwise_or(np.isinf(pose), np.isnan(pose))] = 0.
        velocity[np.bitwise_or(np.isinf(velocity), np.isnan(velocity))] = 0.

        # reverse augment
        # pose_reverse = np.flip(pose, 0)
        # velocity_reverse = np.flip(-velocity, 0)
        return velocity, pose, bag_name

    def _extract_scan(self, msg: LaserScan) -> np.array:
        '''
            Extract scan
        '''
        scan = np.array(msg.ranges)
        range_max = msg.range_max
        range_min = msg.range_min
        # Validation
        scan[np.bitwise_or(np.isinf(scan), np.isnan(scan))] = range_max

        return scan

    def _extract_pc2(self, msg: PointCloud2):
        '''
            Extract pointcloud2 as scan
        '''
        knee_range = sorted(self.config['point_cloud']['z_range'])
        pc_np = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
        print(pc_np.shape)
        pass

    @SaveFrontViewHook('viz/video/front')
    def _get_map(self,
                 bag_name: str,
                 scan: np.array,
                 mapper_kwargs: Dict[str, Any]) -> np.array:
        scan = torch.from_numpy(scan).unsqueeze(0)
        mapper = LocalMap(**mapper_kwargs)
        x_odom = torch.zeros(self.batch_size, self.seq_len).to(self.device)
        y_odom = torch.zeros(self.batch_size, self.seq_len).to(self.device)
        theta_odom = torch.zeros(self.batch_size, self.seq_len).to(self.device)
        angles = torch.linspace(-np.pi, np.pi, scan.shape[-1]).to(self.device)
        distances_x, distances_y = mapper.lidar_scan_xy(scan, angles, x_odom, y_odom, theta_odom)
        binary_maps = mapper.discretize(distances_x, distances_y).cpu().numpy().squeeze()
        return binary_maps, bag_name

    @SaveBackViewHook('viz/video/back')
    def _reverse_augmentation(self,
                              bag_name: str,
                              scan: np.array,
                              mapper_kwargs: Dict[str, Any]) -> np.array:
        '''
            NOTE: MUST BE CALLED AFTER _get_map
        '''
        X_lim = mapper_kwargs['X_lim']
        X_lim[0], X_lim[1] = X_lim[0] * -1, X_lim[1] * -1
        X_lim = list(reversed(X_lim))
        mapper_kwargs['X_lim'] = X_lim
        get_map = self._get_map.__wrapped__
        binary_maps, bag_name = get_map(self, bag_name, scan, mapper_kwargs)
        binary_maps_flipped = np.flip(binary_maps, 0)
        return binary_maps_flipped, bag_name

    def summary(self, start_time: float, end_time: float, count: int) -> None:
        print(f'All bags processed, {end_time - start_time} seconds elapsed, '
              f'total {count} frames')


if __name__ == "__main__":
    extractor = ROSBagExtractor('../bags/Spot', 'blobs', '../configs/Spot.yaml')
    extractor.extract_all()