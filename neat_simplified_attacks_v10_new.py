import carla
import time
import os
import numpy as np
import sys
import random
import queue
import cv2
from datetime import datetime
import csv
import argparse
import traceback
from tqdm import tqdm
from PIL import Image
from types import MethodType
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import logging
from logging.handlers import RotatingFileHandler
from leaderboardcodes.sensor_interface import SensorInterface, CallBack
from multiprocessing import Process
from yaspin import yaspin
from yaspin.spinners import Spinners
from scipy import stats  # For anomaly detection in steering bias

# Configure logging
logger = logging.getLogger(__name__)

def setup_logging(detailed_logging):
    file_handler = RotatingFileHandler('carla_simulation_detailed.log', maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(logging.DEBUG if detailed_logging else logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    console_handler.setLevel(logging.INFO)
    
    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG if detailed_logging else logging.INFO)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PCLA import PCLA, route_maker
from agents.neat.aim_mt_2d.config import GlobalConfig

# Profile the main function
# import cProfile
# import pstats
# profiler = cProfile.Profile()

random.seed(0)
np.random.seed(0)

class AttackState:
    def __init__(self):
        self.current_attack = 'none'
        self.last_sensor_update = {} 

attack_state = AttackState()

def point_to_segment_distance(P, A, B):
    AP = P - A
    AB = B - A
    if np.all(AB == 0):
        return np.linalg.norm(P - A)
    t = np.dot(AP, AB) / np.dot(AB, AB)
    t = max(0, min(1, t))
    closest = A + t * AB
    return np.linalg.norm(P - closest)

sensor_configs = {
    'rgb': {
        'type': 'sensor.camera.rgb',
        'attributes': {
            'image_size_x': '400',
            'image_size_y': '300',
            'fov': '100',
            'lens_circle_multiplier': '3.0',
            'lens_circle_falloff': '3.0',
            'chromatic_aberration_intensity': '0.5',
            'chromatic_aberration_offset': '0',
            'gamma': '2.2',
            'iso': '100.0',
            'shutter_speed': '200.0',
        },
        'transform': carla.Transform(
            carla.Location(x=1.3, y=0.0, z=2.3),
            carla.Rotation(pitch=0.0, roll=0.0, yaw=0.0)
        ),
    },
    'rgb_front': {
        'type': 'sensor.camera.rgb',
        'attributes': {
            'image_size_x': '800',
            'image_size_y': '600',
            'fov': '100',
            'lens_circle_multiplier': '3.0',
            'lens_circle_falloff': '3.0',
            'chromatic_aberration_intensity': '0.5',
            'chromatic_aberration_offset': '0',
            'gamma': '2.2',
            'iso': '100.0',
            'shutter_speed': '200.0',
        },
        'transform': carla.Transform(
            carla.Location(x=1.3, y=0.2, z=2.3),
            carla.Rotation(pitch=0.0, roll=0.0, yaw=10.0)
        ),
    },
    'dvs_front': {
        'type': 'sensor.camera.dvs',
        'attributes': {
            'image_size_x': '800',
            'image_size_y': '600',
            'fov': '100',
            'positive_threshold': '0.3',
            'negative_threshold': '0.3'
        },
        'transform': carla.Transform(
            carla.Location(x=1.3, y=0.0, z=2.3),
            carla.Rotation(pitch=0.0, roll=0.0, yaw=0.0)
        ),
    },
    'depth_front': {
        'type': 'sensor.camera.depth',
        'attributes': {
            'image_size_x': '800',
            'image_size_y': '600',
            'fov': '100',
        },
        'transform': carla.Transform(
            carla.Location(x=1.3, y=0.0, z=2.3),
            carla.Rotation(pitch=0.0, roll=0.0, yaw=0.0)
        ),
    },
    'lidar': {
        'type': 'sensor.lidar.ray_cast',
        'attributes': {
            'channels': '64',
            'range': '85',
            'points_per_second': '600000',
            'rotation_frequency': '10',
            'upper_fov': '10',
            'lower_fov': '-30',
            'atmosphere_attenuation_rate': '0.004',
            'dropoff_general_rate': '0.45',
            'dropoff_intensity_limit': '0.8',
            'dropoff_zero_intensity': '0.4'
        },
        'transform': carla.Transform(
            carla.Location(x=0.0, y=0.0, z=2.5),
            carla.Rotation(pitch=0.0, roll=0.0, yaw=0.0)
        ),
    },
    'gps': {
        'type': 'sensor.other.gnss',
        'attributes': {
            'noise_alt_stddev': '0.000005',
            'noise_lat_stddev': '0.000005',
            'noise_lon_stddev': '0.000005',
            'noise_alt_bias': '0.0',
            'noise_lat_bias': '0.0',
            'noise_lon_bias': '0.0',
            'sensor_tick': '0.01'
        },
        'transform': carla.Transform(
            carla.Location(x=0.0, y=0.0, z=0.0),
            carla.Rotation()
        ),
    },
    'imu': {
        'type': 'sensor.other.imu',
        'attributes': {
            'noise_accel_stddev_x': '0.001',
            'noise_accel_stddev_y': '0.001',
            'noise_accel_stddev_z': '0.015',
            'noise_gyro_stddev_x': '0.001',
            'noise_gyro_stddev_y': '0.001',
            'noise_gyro_stddev_z': '0.001',
            'sensor_tick': '0.05'
        },
        'transform': carla.Transform(
            carla.Location(x=0.0, y=0.0, z=0.0),
            carla.Rotation(pitch=0.0, roll=0.0, yaw=0.0)
        ),
    },
}

def make_rgb_callback(queue, sensor_interface, tag, enable_defenses=False):
    def callback(image):
        try:
            # 1. Read raw RGB data
            array = np.frombuffer(image.raw_data, dtype=np.uint8) \
                      .reshape((image.height, image.width, 4))[:, :, :3].copy()
            noise_percentage = 0.0

            # 2. Inject 80% salt-and-pepper noise
            if attack_state.current_attack == 'rgb_noise':
                noise_density = 0.8
                mask = np.random.rand(image.height, image.width) < noise_density
                pepper_salt = np.random.choice([0, 255], size=array.shape, p=[0.5, 0.5])
                array[mask] = pepper_salt[mask]
                noise_percentage = mask.mean() * 100

            # 3. Defense: Decision-based Adaptive Median Filter
            if enable_defenses:
                def decision_adaptive_median(img, max_kernel=7):
                    out = img.copy()
                    h, w, c = img.shape
                    # Process each channel separately
                    for ch in range(c):
                        channel = img[:, :, ch]
                        # Identify salt-and-pepper candidates
                        noisy = (channel == 0) | (channel == 255)
                        ys, xs = np.where(noisy)
                        for y, x in zip(ys, xs):
                            k = 3
                            replaced = False
                            # Grow window until valid median found or max reached
                            while k <= max_kernel:
                                y1, y2 = max(0, y-k//2), min(h, y+k//2+1)
                                x1, x2 = max(0, x-k//2), min(w, x+k//2+1)
                                window = channel[y1:y2, x1:x2]
                                vals = window[(window != 0) & (window != 255)]
                                if vals.size > 0:
                                    out[y, x, ch] = np.median(vals)
                                    replaced = True
                                    break
                                k += 2
                            # If no valid values found, retain original
                            if not replaced:
                                out[y, x, ch] = channel[y, x]
                    return out

                array = decision_adaptive_median(array)

            # 4. Update sensor and queue
            sensor_interface.update_sensor(tag, array, image.frame)
            queue.put((array, noise_percentage))

        except Exception as e:
            logger.error(f"Error in RGB callback for {tag}: {e}\n{traceback.format_exc()}")
            # Fallback: zeros image
            queue.put((np.zeros((image.height, image.width, 3), dtype=np.uint8), 0.0))

    return callback

def make_dvs_callback(queue, sensor_interface, tag, enable_defenses=False):
    def callback(data):
        try:
            image = np.zeros((data.height, data.width, 3), dtype=np.uint8)
            image[:, :] = [128, 128, 128]
            events = data
            event_count = len(events)
            for event in events:
                x, y, pol = event.x, event.y, event.pol
                image[y, x] = [255, 0, 0] if pol == 1 else [0, 0, 255]
            logger.debug(f"DVS callback ({tag}): event_count={event_count}, image_shape={image.shape}")
            if attack_state.current_attack == 'dvs_noise':
                num_noise_events = int(0.6 * data.width * data.height)
                noise_x = np.random.randint(0, data.width, num_noise_events)
                noise_y = np.random.randint(0, data.height, num_noise_events)
                noise_pol = np.random.choice([0, 1], num_noise_events)
                for x, y, pol in zip(noise_x, noise_y, noise_pol):
                    image[y, x] = [255, 0, 0] if pol == 1 else [0, 0, 255]
                event_count += num_noise_events
                logger.debug(f"DVS noise applied ({tag}): added {num_noise_events} events")
            if enable_defenses:
                max_events = 5000  # Stricter limit
                if event_count > max_events:
                    # Spatial clustering to detect noise
                    coords = np.array([(event.x, event.y) for event in events])
                    if len(coords) > 0:
                        from scipy.spatial import cKDTree
                        tree = cKDTree(coords)
                        distances, _ = tree.query(coords, k=2)
                        avg_neighbor_distance = np.mean(distances[:, 1])
                        if avg_neighbor_distance < 2.0:  # Dense clusters indicate noise
                            image = np.zeros_like(image)
                            image[:, :] = [128, 128, 128]
                            event_count = 0
                            logger.debug(f"DVS defense applied ({tag}): reset image due to dense events, avg_distance={avg_neighbor_distance}")
                        else:
                            image = cv2.dilate(image, np.ones((3, 3), np.uint8))  # Smooth sparse events
                            logger.debug(f"DVS defense applied ({tag}): dilated image")
                else:
                    image = cv2.dilate(image, np.ones((3, 3), np.uint8))  # Standard smoothing
                    logger.debug(f"DVS defense applied ({tag}): dilated image")
            sensor_interface.update_sensor(tag, image, data.frame)
            queue.put((image, event_count))
            image = None
            logger.debug(f"DVS callback ({tag}): data queued")
        except Exception as e:
            logger.error(f"Error in DVS callback for {tag}: {e}\n{traceback.format_exc()}")
            queue.put((np.zeros((data.height, data.width, 3), dtype=np.uint8), 0))
    return callback

def make_depth_callback(queue, sensor_interface, tag, enable_defenses=False):
    def callback(image):
        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
            depth = array.astype(np.float32)
            depth = depth[:, :, 0] * (256.0 * 256.0) + depth[:, :, 1] * 256.0 + depth[:, :, 2]
            depth = depth / (256.0 * 256.0 * 256.0 - 1.0) * 1000.0
            logger.debug(f"Depth callback ({tag}): shape={depth.shape}, dtype={depth.dtype}, range=[{depth.min()}, {depth.max()}]")
            if attack_state.current_attack == 'depth_tampering':
                num_patches = 5
                patch_size = 50
                for _ in range(num_patches):
                    x = random.randint(0, image.width - patch_size)
                    y = random.randint(0, image.height - patch_size)
                    depth[y:y+patch_size, x:x+patch_size] += random.uniform(5.0, 15.0)
                depth = np.clip(depth, 0.0, 1000.0)
                logger.debug(f"Depth tampering applied ({tag}): added {num_patches} patches")
            if enable_defenses:
                max_depth = 100.0  # Realistic range limit
                min_depth = 0.0
                depth = np.clip(depth, min_depth, max_depth)
                # Detect tampering via gradient analysis
                grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=5)
                grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=5)
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                if np.max(grad_mag) > 50.0:  # High gradients indicate tampering
                    depth = cv2.GaussianBlur(depth, (7, 7), 0)  # Stronger smoothing
                    logger.debug(f"Depth defense applied ({tag}): Gaussian blur due to high gradients")
                else:
                    depth = cv2.GaussianBlur(depth, (5, 5), 0)  # Standard smoothing
                    logger.debug(f"Depth defense applied ({tag}): Gaussian blur")
            depth_normalized = np.clip(depth / 1000.0, 0.0, 1.0)
            depth_uint8 = (depth_normalized * 255).astype(np.uint8)
            depth_rgb = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2RGB)
            sensor_interface.update_sensor(tag, depth, image.frame)
            queue.put((depth_rgb, np.mean(depth)))
            array = None
            logger.debug(f"Depth callback ({tag}): data queued")
        except Exception as e:
            logger.error(f"Error in Depth callback for {tag}: {e}\n{traceback.format_exc()}")
            queue.put((np.zeros((image.height, image.width, 3), dtype=np.uint8), 0.0))
    return callback

def make_lidar_callback(queue, sensor_interface, tag, enable_defenses=False):
    def callback(data):
        try:
            points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4).copy()
            logger.debug(f"LiDAR callback ({tag}): shape={points.shape}, dtype={points.dtype}")
            if attack_state.current_attack == 'lidar_tampering':
                num_objects = 5
                points_per_object = 100
                for _ in range(num_objects):
                    obj_x = random.uniform(-10, 10)
                    obj_y = random.uniform(5, 15)
                    obj_z = random.uniform(0, 2)
                    noise_points = np.random.normal(loc=[obj_x, obj_y, obj_z], scale=0.5, size=(points_per_object, 3))
                    noise_points = np.hstack((noise_points, np.ones((points_per_object, 1))))
                    points = np.vstack((points, noise_points))
                logger.debug(f"LiDAR noise applied ({tag}): added {num_objects * points_per_object} points")
            if enable_defenses:
                # Distance and density filtering
                distances = np.linalg.norm(points[:, :3], axis=1)
                points = points[distances < 50.0]  # Max range
                if points.shape[0] > 0:
                    from scipy.spatial import cKDTree
                    tree = cKDTree(points[:, :3])
                    distances, _ = tree.query(points[:, :3], k=2)
                    avg_neighbor_distance = np.mean(distances[:, 1])
                    if avg_neighbor_distance < 0.1:  # Dense clusters indicate tampering
                        points = points[distances[:, 1] > 0.1]  # Remove dense points
                        logger.debug(f"LiDAR defense applied ({tag}): filtered dense points, remaining={points.shape[0]}")
                    logger.debug(f"LiDAR defense applied ({tag}): filtered to {points.shape[0]} points")
            sensor_interface.update_sensor(tag, points, data.frame)
            queue.put(points)
            points = None
            array = None
            image = None
            logger.debug(f"LiDAR callback ({tag}): data queued")
        except Exception as e:
            logger.error(f"Error in LiDAR callback for {tag}: {e}\n{traceback.format_exc()}")
            queue.put(np.array([]))
    return callback

def make_gps_callback(queue, sensor_interface, tag):
    def callback(data):
        try:
            array = np.array([data.latitude, data.longitude, data.altitude], dtype=np.float64)
            logger.debug(f"GPS callback ({tag}): shape={array.shape}, dtype={array.dtype}, value={array}")
            sensor_interface.update_sensor(tag, array, data.frame)
            queue.put(data)
            array = None
            logger.debug(f"GPS callback ({tag}): data queued")
        except Exception as e:
            logger.error(f"Error in GPS callback for {tag}: {e}\n{traceback.format_exc()}")
            queue.put(carla.GnssMeasurement(0.0, 0.0, 0.0))
    return callback

def make_imu_callback(queue, sensor_interface, tag):
    def callback(data):
        try:
            array = np.array([
                data.accelerometer.x, data.accelerometer.y, data.accelerometer.z,
                data.gyroscope.x, data.gyroscope.y, data.gyroscope.z,
                data.compass
            ], dtype=np.float64)
            logger.debug(f"IMU callback ({tag}): shape={array.shape}, dtype={array.dtype}, value={array}")
            sensor_interface.update_sensor(tag, array, data.frame)
            queue.put(data)
            array = None
            logger.debug(f"IMU callback ({tag}): data queued")
        except Exception as e:
            logger.error(f"Error in IMU callback for {tag}: {e}\n{traceback.format_exc()}")
            queue.put(carla.IMUMeasurement(0, 0, 0, 0, 0, 0, 0))
    return callback

def visualize_lidar(points, height, width, range_max=50.0):
    try:
        if points.size == 0:
            logger.debug("LiDAR visualization: empty points array")
            return np.zeros((height, width, 3), dtype=np.uint8)
        distances = np.linalg.norm(points[:, :3], axis=1)
        points = points[distances < range_max]
        logger.debug(f"LiDAR visualization: filtered to {points.shape[0]} points")
        if points.size == 0:
            logger.debug("LiDAR visualization: no points after filtering")
            return np.zeros((height, width, 3), dtype=np.uint8)
        x = points[:, 0]
        y = points[:, 1]
        x_img = np.clip(((x + range_max) / (2 * range_max)) * (width - 1), 0, width - 1).astype(int)
        y_img = np.clip(((y + range_max) / (2 * range_max)) * (height - 1), 0, height - 1).astype(int)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[y_img, x_img] = [255, 255, 255]
        logger.debug(f"LiDAR visualization: generated image shape={img.shape}")
        return img
    except Exception as e:
        logger.error(f"Error in LiDAR visualization: {e}\n{traceback.format_exc()}")
        return np.zeros((height, width, 3), dtype=np.uint8)

def generate_combined_video(episode_dir, episode):
    transform_path = os.path.join(episode_dir, 'transform.csv')
    if not os.path.exists(transform_path):
        logger.error(f"Transform data not found for episode {episode}")
        return
    transform_data = []
    try:
        with open(transform_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                transform_data.append({
                    'tick': int(row[0]),
                    'attack': row[17],
                    'defense_enabled': bool(int(row[18]))
                })
        logger.debug(f"Video generation (episode {episode}): loaded {len(transform_data)} transform entries")
    except Exception as e:
        logger.error(f"Error reading transform data for episode {episode}: {e}\n{traceback.format_exc()}")
        return
    if not transform_data:
        logger.error(f"No transform data for episode {episode}")
        return
    video_sensors = ['third_person_rgb', 'rgb_front', 'dvs_front', 'depth_front', 'lidar']
    height, width = 600, 800
    total_width = width * 3
    total_height = height * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(episode_dir, f'episode_{episode}_combined.mp4')
    try:
        video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (total_width, total_height))
        placeholder = np.zeros((height, width, 3), dtype=np.uint8)
        for data in tqdm(transform_data, desc=f"Generating video for Episode {episode}"):
            tick = data['tick']
            attack = data['attack']
            defense = data['defense_enabled']
            frames = {}
            for sensor_id in video_sensors:
                if sensor_id == 'lidar':
                    lidar_path = os.path.join(episode_dir, 'lidar', f'lidar_{tick:06d}.npy')
                    if os.path.exists(lidar_path):
                        frames['lidar'] = visualize_lidar(np.load(lidar_path), height, width)
                        logger.debug(f"Video generation: loaded lidar frame {tick} shape={frames['lidar'].shape}")
                    else:
                        frames['lidar'] = placeholder.copy()
                        logger.debug(f"Video generation: lidar frame {tick} missing, using placeholder")
                else:
                    frame_path = os.path.join(episode_dir, sensor_id, f'frame_{tick:06d}.png')
                    if os.path.exists(frame_path):
                        frames[sensor_id] = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                        logger.debug(f"Video generation: loaded {sensor_id} frame {tick} shape={frames[sensor_id].shape}")
                    else:
                        frames[sensor_id] = placeholder.copy()
                        logger.debug(f"Video generation: {sensor_id} frame {tick} missing, using placeholder")
            combined = np.zeros((total_height, total_width, 3), dtype=np.uint8)
            combined[0:height, 0:width] = cv2.resize(frames['third_person_rgb'], (width, height))
            combined[0:height, width:width*2] = frames['rgb_front']
            combined[0:height, width*2:total_width] = frames['dvs_front']
            combined[height:total_height, 0:width] = frames['depth_front']
            combined[height:total_height, width:width*2] = frames['lidar']
            font = cv2.FONT_HERSHEY_SIMPLEX
            labels = ['Third-Person RGB', 'RGB Front', 'DVS Front', 'Depth Front', 'LiDAR BEV']
            for i, (label, y) in enumerate(zip(labels, [30, 30, 30, height + 30, height + 30])):
                x = i % 3 * width + 10
                cv2.putText(combined, label, (x, y), font, 1, (255, 255, 255), 2)
            state_text = f"Attack: {attack}" if not defense else f"Attack: {attack} (Defenses)"
            cv2.putText(combined, state_text, (width*2 + 10, height + 30), font, 1, (0, 0, 255), 2)
            video_writer.write(combined)
            combined = None
            logger.debug(f"Video generation: wrote frame {tick} for episode {episode}")
        video_writer.release()
        logger.info(f"Video saved to {video_path}")
    except Exception as e:
        logger.error(f"Error generating video for episode {episode}: {e}\n{traceback.format_exc()}")
        if 'video_writer' in locals():
            video_writer.release()

def setup_sensors(world, vehicle, episode_dir, agent_type, enable_defenses=False):
    bp_library = world.get_blueprint_library()
    sensors_to_spawn = ['rgb', 'rgb_front', 'dvs_front', 'depth_front', 'lidar', 'gps', 'imu']
    image_sensors = ['third_person_rgb', 'rgb', 'rgb_front', 'dvs_front', 'depth_front', 'lidar']
    logger.debug(f"Setting up sensors: {sensors_to_spawn}")
    for sensor_id in image_sensors:
        os.makedirs(os.path.join(episode_dir, sensor_id), exist_ok=True)
        logger.debug(f"Created directory for {sensor_id}")
    sensor_queues = {sensor_id: queue.Queue(maxsize=10) for sensor_id in sensors_to_spawn + ['third_person_rgb']}
    sensor_interface = SensorInterface()
    sensors = []
    try:
        third_person_rgb_bp = bp_library.find('sensor.camera.rgb')
        third_person_rgb_bp.set_attribute('image_size_x', '1280')
        third_person_rgb_bp.set_attribute('image_size_y', '720')
        third_person_rgb_bp.set_attribute('fov', '90')
        third_person_rgb_transform = carla.Transform(carla.Location(x=-6.0, z=2.5), carla.Rotation(pitch=-10))
        third_person_rgb_camera = world.spawn_actor(third_person_rgb_bp, third_person_rgb_transform, attach_to=vehicle)
        third_person_rgb_camera.listen(make_rgb_callback(sensor_queues['third_person_rgb'], sensor_interface, 'third_person_rgb', enable_defenses))
        sensors.append(third_person_rgb_camera)
        sensor_interface.register_sensor('third_person_rgb', 'sensor.camera.rgb', third_person_rgb_camera)
        logger.info("Spawned and registered third_person_rgb camera")
    except Exception as e:
        logger.error(f"Failed to spawn or register third_person_rgb camera: {e}\n{traceback.format_exc()}")
        raise
    callback_makers = {
        'sensor.camera.rgb': lambda q, si, tag: make_rgb_callback(q, si, tag, enable_defenses),
        'sensor.camera.dvs': lambda q, si, tag: make_dvs_callback(q, si, tag, enable_defenses),
        'sensor.camera.depth': lambda q, si, tag: make_depth_callback(q, si, tag, enable_defenses),
        'sensor.lidar.ray_cast': lambda q, si, tag: make_lidar_callback(q, si, tag, enable_defenses),
        'sensor.other.gnss': lambda q, si, tag: make_gps_callback(q, si, tag),
        'sensor.other.imu': lambda q, si, tag: make_imu_callback(q, si, tag),
    }
    for sensor_id in sensors_to_spawn:
        try:
            config = sensor_configs[sensor_id]
            bp = bp_library.find(config['type'])
            for attr, value in config['attributes'].items():
                bp.set_attribute(attr, value)
            sensor = world.spawn_actor(bp, config['transform'], attach_to=vehicle)
            sensor.listen(callback_makers[config['type']](sensor_queues[sensor_id], sensor_interface, sensor_id))
            sensors.append(sensor)
            sensor_interface.register_sensor(sensor_id, config['type'], sensor)
            logger.info(f"Spawned sensor {sensor_id}")
        except Exception as e:
            logger.error(f"Failed to spawn sensor {sensor_id}: {e}\n{traceback.format_exc()}")
            raise
    writers = {}
    files = []
    headers = {
        'transform': ['tick', 'timestamp', 'perceived_x', 'perceived_y', 'perceived_z', 
                      'perceived_pitch', 'perceived_yaw', 'perceived_roll', 
                      'actual_x', 'actual_y', 'actual_z', 'actual_pitch', 'actual_yaw', 
                      'actual_roll', 'vx', 'vy', 'vz', 'attack', 'defense_enabled'],
        'control': ['timestamp', 'throttle', 'steer', 'brake'],
        'gps': ['tick', 'timestamp', 'latitude', 'longitude', 'altitude'],
        'rgb_front_mean': ['tick', 'mean_intensity'],
        'rgb_mean': ['tick', 'mean_intensity'],
        'dvs_events': ['tick', 'event_count'],
        'depth_mean': ['tick', 'mean_depth'],
        'lidar_points': ['tick', 'num_points'],
        'rgb_noise': ['tick', 'noise_percentage']
    }
    for fname, header in headers.items():
        file_path = os.path.join(episode_dir, f'{fname}.csv')
        file = open(file_path, 'w', newline='')
        writer = csv.writer(file)
        writer.writerow(header)
        writers[fname] = writer
        files.append(file)
        logger.debug(f"Created CSV file {file_path} with headers {header}")
    time.sleep(1.0)
    return sensors, sensor_queues, writers, files, sensors_to_spawn, sensor_interface

def save_sensor_data(sensor_queues, episode_dir, tick, writers, vehicle, world, sensors_to_spawn, sampling_rate, sensor_interface, enable_defenses=False):
    if tick % sampling_rate != 0:
        return
    
    logger.debug(f"Saving sensor data for tick {tick}")
    batch_size = 5
    batch_data = {sensor_id: [] for sensor_id in ['third_person_rgb', 'rgb_front', 'rgb', 'dvs_front', 'depth_front', 'lidar', 'gps']}
    
    # DoS defense: Rate limiting
    current_time = time.time()
    for sensor_id in batch_data.keys():
        if sensor_id in attack_state.last_sensor_update:
            if current_time - attack_state.last_sensor_update[sensor_id] < 0.05:  # 20 Hz max
                logger.debug(f"DoS defense: Skipped {sensor_id} update at tick {tick} due to rate limit")
                continue
        attack_state.last_sensor_update[sensor_id] = current_time
        
        if sensor_id not in sensor_queues or sensor_queues[sensor_id].empty():
            logger.debug(f"No data in queue for {sensor_id} at tick {tick}")
            continue
        try:
            data = sensor_queues[sensor_id].get_nowait()
            batch_data[sensor_id].append((tick, data))
        except queue.Empty:
            logger.debug(f"Queue empty for {sensor_id} at tick {tick}")
            continue
        except Exception as e:
            logger.error(f"Error retrieving data for {sensor_id} at tick {tick}: {e}\n{traceback.format_exc()}")

    if tick % (sampling_rate * batch_size) == 0:
        for sensor_id in ['third_person_rgb', 'rgb_front', 'rgb', 'dvs_front', 'depth_front']:
            for tick_in_batch, data in batch_data[sensor_id]:
                try:
                    path = os.path.join(episode_dir, sensor_id, f'frame_{tick_in_batch:06d}.png')
                    if sensor_id in ['rgb_front', 'rgb']:
                        array, noise_percentage = data
                        logger.debug(f"Saving {sensor_id} at tick {tick_in_batch}: shape={array.shape}, noise_percentage={noise_percentage}")
                        writers[f'{sensor_id}_mean'].writerow([tick_in_batch, np.mean(array)])
                        if attack_state.current_attack == 'rgb_noise':
                            writers['rgb_noise'].writerow([tick_in_batch, noise_percentage])
                            logger.debug(f"Logged rgb_noise for {sensor_id}: {noise_percentage}")
                        cv2.imwrite(path, array[:, :, ::-1])
                    elif sensor_id == 'dvs_front':
                        image, event_count = data
                        logger.debug(f"Saving dvs_front at tick {tick_in_batch}: shape={image.shape}, event_count={event_count}")
                        writers['dvs_events'].writerow([tick_in_batch, event_count])
                        cv2.imwrite(path, image[:, :, ::-1])
                    elif sensor_id == 'depth_front':
                        image, mean_depth = data
                        logger.debug(f"Saving depth_front at tick {tick_in_batch}: shape={image.shape}, mean_depth={mean_depth}")
                        writers['depth_mean'].writerow([tick_in_batch, mean_depth])
                        cv2.imwrite(path, image[:, :, ::-1])
                    elif sensor_id == 'third_person_rgb':
                        array, _ = data
                        logger.debug(f"Saving third_person_rgb at tick {tick_in_batch}: shape={array.shape}")
                        cv2.imwrite(path, array[:, :, ::-1])
                except Exception as e:
                    logger.error(f"Error saving data for {sensor_id} at tick {tick_in_batch}: {e}\n{traceback.format_exc()}")

        if 'lidar' in sensors_to_spawn:
            for tick_in_batch, points in batch_data['lidar']:
                try:
                    logger.debug(f"Saving lidar at tick {tick_in_batch}: shape={points.shape}")
                    np.save(os.path.join(episode_dir, 'lidar', f'lidar_{tick_in_batch:06d}.npy'), points)
                    writers['lidar_points'].writerow([tick_in_batch, points.shape[0]])
                except Exception as e:
                    logger.error(f"Error saving lidar data at tick {tick_in_batch}: {e}\n{traceback.format_exc()}")

        if 'gps' in sensors_to_spawn:
            for tick_in_batch, data in batch_data['gps']:
                try:
                    logger.debug(f"Saving gps at tick {tick_in_batch}: latitude={data.latitude}, longitude={data.longitude}")
                    writers['gps'].writerow([tick_in_batch, data.timestamp, data.latitude, data.longitude, data.altitude])
                except Exception as e:
                    logger.error(f"Error saving gps data at tick {tick_in_batch}: {e}\n{traceback.format_exc()}")

    try:
        perceived_transform = vehicle.get_transform()
        actual_transform = vehicle._original_get_transform()
        velocity = vehicle.get_velocity()
        writers['transform'].writerow([
            tick, world.get_snapshot().timestamp.elapsed_seconds,
            perceived_transform.location.x, perceived_transform.location.y, perceived_transform.location.z,
            perceived_transform.rotation.pitch, perceived_transform.rotation.yaw, perceived_transform.rotation.roll,
            actual_transform.location.x, actual_transform.location.y, actual_transform.location.z,
            actual_transform.rotation.pitch, actual_transform.rotation.yaw, actual_transform.rotation.roll,
            velocity.x, velocity.y, velocity.z,
            attack_state.current_attack, int(enable_defenses)
        ])
        logger.debug(f"Saved transform data at tick {tick}")
    except Exception as e:
        logger.error(f"Error saving transform data at tick {tick}: {e}\n{traceback.format_exc()}")

    # Memory cleanup: Clear batch data and sensor queues
    for sensor_id in batch_data:
        batch_data[sensor_id].clear()
        batch_data[sensor_id] = None  # Ensure no lingering references
    batch_data = None

    for q in sensor_queues.values():
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                pass
    logger.debug(f"Tick {tick}: cleared sensor queues and batch data")

def find_valid_frame(episode_dir, sensor_id, ticks):
    for tick in sorted(ticks):
        path = os.path.join(episode_dir, sensor_id, f'frame_{tick:06d}.png')
        if os.path.exists(path):
            return tick, path
    return None, None

def plot_behavior_async(episode_dir, comparison_plots_dir, episode):
    plot_behavior(episode_dir, comparison_plots_dir, episode)

def plot_behavior(episode_dir, comparison_plots_dir, episode):
    try:
        with yaspin(Spinners.line, text=f"Loading data ...", color="yellow") as spinner:
            transform_data = []
            with open(os.path.join(episode_dir, 'transform.csv'), 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    transform_data.append({
                        'tick': int(row[0]), 'timestamp': float(row[1]), 
                        'perceived_x': float(row[2]), 'perceived_y': float(row[3]), 
                        'actual_x': float(row[8]), 'actual_y': float(row[9]), 
                        'vx': float(row[14]), 'vy': float(row[15]), 'attack': row[17],
                        'defense_enabled': bool(int(row[18]))
                    })
            logger.debug(f"Plot behavior (episode {episode}): loaded {len(transform_data)} transform entries")

            for data in transform_data:
                data['speed'] = np.sqrt(data['vx']**2 + data['vy']**2)

            control_data = []
            with open(os.path.join(episode_dir, 'control.csv'), 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    control_data.append({'timestamp': float(row[0]), 'throttle': float(row[1]), 
                                        'steer': float(row[2]), 'brake': float(row[3])})
            logger.debug(f"Plot behavior (episode {episode}): loaded {len(control_data)} control entries")

            depth_data = []
            with open(os.path.join(episode_dir, 'depth_mean.csv'), 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    depth_data.append({'tick': int(row[0]), 'mean_depth': float(row[1])})
            logger.debug(f"Plot behavior (episode {episode}): loaded {len(depth_data)} depth entries")

            dvs_data = []
            with open(os.path.join(episode_dir, 'dvs_events.csv'), 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    dvs_data.append({'tick': int(row[0]), 'event_count': float(row[1])})
            logger.debug(f"Plot behavior (episode {episode}): loaded {len(dvs_data)} DVS entries")

            lidar_data = []
            with open(os.path.join(episode_dir, 'lidar_points.csv'), 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    lidar_data.append({'tick': int(row[0]), 'num_points': float(row[1])})
            logger.debug(f"Plot behavior (episode {episode}): loaded {len(lidar_data)} LiDAR entries")

            rgb_noise_data = []
            rgb_noise_path = os.path.join(episode_dir, 'rgb_noise.csv')
            if os.path.exists(rgb_noise_path):
                with open(rgb_noise_path, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        rgb_noise_data.append({'tick': int(row[0]), 'noise_percentage': float(row[1])})
            logger.debug(f"Plot behavior (episode {episode}): loaded {len(rgb_noise_data)} RGB noise entries")

            attacks = ['none', 'gps', 'rgb_noise', 'dvs_noise', 'depth_tampering', 'lidar_tampering', 'dos', 'steering_bias']
            colors = {
                'none': 'green', 'gps': 'red', 'rgb_noise': 'purple', 'dvs_noise': 'blue',
                'depth_tampering': 'yellow', 'lidar_tampering': 'orange', 'dos': 'cyan', 'steering_bias': 'magenta'
            }

            fig, axes = plt.subplots(6, 1, figsize=(12, 24), sharex=True)
            axes[0].plot([data['timestamp'] for data in transform_data], [data['speed'] for data in transform_data])
            for attack in attacks[1:]:
                for defense, alpha in [(False, 0.3), (True, 0.1)]:
                    times = [data['timestamp'] for data in transform_data if data['attack'] == attack and data['defense_enabled'] == defense]
                    if times:
                        axes[0].axvspan(min(times), max(times), color=colors[attack], alpha=alpha)
            axes[0].set_ylabel('Speed (m/s)')
            axes[0].set_title(f'Episode {episode}: Control Inputs and Sensor Metrics')
            axes[0].text(0.05, 0.95, 'Lighter shades: defenses', transform=axes[0].transAxes, fontsize=10)

            for i, (key, ylabel) in enumerate([('throttle', 'Throttle'), ('steer', 'Steering'), ('brake', 'Brake')], 1):
                axes[i].plot([data['timestamp'] for data in control_data], [data[key] for data in control_data])
                for attack in attacks[1:]:
                    for defense, alpha in [(False, 0.3), (True, 0.1)]:
                        times = [data['timestamp'] for data in transform_data if data['attack'] == attack and data['defense_enabled'] == defense]
                        if times:
                            axes[i].axvspan(min(times), max(times), color=colors[attack], alpha=alpha)
                axes[i].set_ylabel(ylabel)

            axes[4].plot([transform_data[i]['timestamp'] for i in range(len(depth_data))], [data['mean_depth'] for data in depth_data])
            for attack in attacks[1:]:
                for defense, alpha in [(False, 0.3), (True, 0.1)]:
                    times = [data['timestamp'] for data in transform_data if data['attack'] == attack and data['defense_enabled'] == defense]
                    if times:
                        axes[4].axvspan(min(times), max(times), color=colors[attack], alpha=alpha)
            axes[4].set_ylabel('Mean Depth (m)')
        spinner.ok("✅ Done")

        with yaspin(Spinners.line, text=f"Generating control and sensor metrics plot ...", color="yellow") as spinner:
            axes[5].plot([transform_data[i]['timestamp'] for i in range(len(dvs_data))], [data['event_count'] for data in dvs_data])
            for attack in attacks[1:]:
                for defense, alpha in [(False, 0.3), (True, 0.1)]:
                    times = [data['timestamp'] for data in transform_data if data['attack'] == attack and data['defense_enabled'] == defense]
                    if times:
                        axes[5].axvspan(min(times), max(times), color=colors[attack], alpha=alpha)
            axes[5].set_ylabel('DVS Events')
            axes[5].set_xlabel('Time (s)')
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_plots_dir, f'episode_{episode}_control_and_sensors.png'))
            plt.close()
            logger.debug(f"Plotted control and sensor metrics for episode {episode}")
        spinner.ok("✅ Done")

        with yaspin(Spinners.line, text=f"Generating RGB noise plot ...", color="yellow") as spinner:
            if rgb_noise_data:
                plt.figure(figsize=(10, 6))
                plt.plot([data['tick'] for data in rgb_noise_data], [data['noise_percentage'] for data in rgb_noise_data])
                rgb_times = [data['tick'] for data in transform_data if data['attack'] == 'rgb_noise']
                if rgb_times:
                    plt.axvspan(min(rgb_times), max(rgb_times), color='purple', alpha=0.3)
                plt.xlabel('Tick')
                plt.ylabel('Noise Percentage (%)')
                plt.title(f'Episode {episode}: RGB Noise Percentage')
                plt.savefig(os.path.join(comparison_plots_dir, f'episode_{episode}_rgb_noise.png'))
                plt.close()
                logger.debug(f"Plotted RGB noise for episode {episode}")
        spinner.ok("✅ Done")

        with yaspin(Spinners.line, text=f"Loading data ...", color="yellow") as spinner:
            route_data = []
            with open(os.path.join(episode_dir, 'route.csv'), 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    route_data.append({'x': float(row[1]), 'y': float(row[2])})
            logger.debug(f"Plot behavior (episode {episode}): loaded {len(route_data)} route points")
            route_points = np.array([[point['x'], point['y']] for point in route_data])

            for data in transform_data:
                P = np.array([data['actual_x'], data['actual_y']])
                data['deviation'] = min(point_to_segment_distance(P, route_points[i], route_points[i + 1]) 
                                    for i in range(len(route_points) - 1))
        spinner.ok("✅ Done") 

        with yaspin(Spinners.line, text=f"Generating trajectory plot ...", color="yellow") as spinner:
            plt.figure(figsize=(12, 8))
            plt.plot([point['x'] for point in route_data], [point['y'] for point in route_data], 'k--', label='Route')
            for phase, ls in [(False, '-'), (True, '--')]:
                phase_data = [data for data in transform_data if data['defense_enabled'] == phase]
                for i in range(len(phase_data) - 1):
                    start, end = phase_data[i], phase_data[i + 1]
                    if start['attack'] == end['attack']:
                        plt.plot([start['actual_x'], end['actual_x']], [start['actual_y'], end['actual_y']], 
                                color=colors[start['attack']], linestyle=ls, linewidth=2, alpha=0.8)
            for i in range(len(transform_data) - 1):
                start, end = transform_data[i], transform_data[i + 1]
                if start['attack'] == 'gps' and end['attack'] == 'gps' and not start['defense_enabled']:
                    plt.plot([start['perceived_x'], end['perceived_x']], 
                            [start['perceived_y'], end['perceived_y']], 
                            color='red', linestyle=':', linewidth=1, dashes=(1, 5), alpha=1.0)
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title(f'Episode {episode}: Trajectory (Solid: Attacks, Dashed: Defenses)')
            plt.legend([plt.Line2D([0], [0], color=colors[attack], lw=2) for attack in attacks] + 
                    [plt.Line2D([0], [0], color='k', lw=2, ls='--'), plt.Line2D([0], [0], color='r', lw=2, ls=':')], 
                    attacks + ['Route', 'GPS Perceived (Attacks)'])
            plt.savefig(os.path.join(comparison_plots_dir, f'episode_{episode}_trajectory.png'))
            plt.close()
            logger.debug(f"Plotted trajectory for episode {episode}")
        spinner.ok("✅ Done")
        
        with yaspin(Spinners.line, text=f"Generating steering distribution comparison plot ...", color="yellow") as spinner:
            fig, ax = plt.subplots(figsize=(10, 6))
            for attack in attacks:
                steer_values = [control['steer'] for control, transform in zip(control_data, transform_data) if transform['attack'] == attack]
                if steer_values:
                    ax.hist(steer_values, bins=30, alpha=0.5, label=attack, color=colors[attack])
            ax.set_xlabel('Steering Angle')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Episode {episode}: Steering Angle Distribution by Attack')
            ax.legend()
            plt.savefig(os.path.join(comparison_plots_dir, f'episode_{episode}_steering_distribution.png'))
            plt.close()
            logger.debug(f"Plotted steering distribution for episode {episode}")
        spinner.ok("✅ Done")

        attack_data = [data for data in transform_data if not data['defense_enabled']]
        defense_data = [data for data in transform_data if data['defense_enabled']]

        with yaspin(Spinners.line, text=f"Generating deviation comparison plot ...", color="yellow") as spinner:
            if attack_data and defense_data:
                attack_deviation = np.mean([data['deviation'] for data in attack_data])
                defense_deviation = np.mean([data['deviation'] for data in defense_data])
                plt.figure(figsize=(6, 6))
                plt.bar(['Attacks', 'Defenses'], [attack_deviation, defense_deviation], color=['red', 'blue'])
                plt.ylabel('Average Deviation (m)')
                plt.title(f'Episode {episode}: Deviation Comparison')
                plt.savefig(os.path.join(comparison_plots_dir, f'episode_{episode}_deviation_comparison_phases.png'))
                plt.close()
                logger.debug(f"Plotted deviation comparison for episode {episode}")
        spinner.ok("✅ Done")
        
        with yaspin(Spinners.line, text=f"Generating speed comparison plot ...", color="yellow") as spinner:
            if attack_data and defense_data:
                attack_speed = np.mean([data['speed'] for data in attack_data])
                defense_speed = np.mean([data['speed'] for data in defense_data])
                plt.figure(figsize=(6, 6))
                plt.bar(['Attacks', 'Defenses'], [attack_speed, defense_speed], color=['red', 'blue'])
                plt.ylabel('Average Speed (m/s)')
                plt.title(f'Episode {episode}: Speed Comparison')
                plt.savefig(os.path.join(comparison_plots_dir, f'episode_{episode}_speed_comparison_phases.png'))
                plt.close()
                logger.debug(f"Plotted speed comparison for episode {episode}")
        spinner.ok("✅ Done")

        attack_intervals = {}
        current_attack = None
        start_tick = None
        for data in transform_data:
            if data['attack'] != current_attack:
                if current_attack is not None and current_attack != 'none':
                    attack_intervals[current_attack] = attack_intervals.get(current_attack, []) + [(start_tick, data['tick'])]
                current_attack = data['attack']
                start_tick = data['tick']
        if current_attack is not None and current_attack != 'none':
            attack_intervals[current_attack] = attack_intervals.get(current_attack, []) + [(start_tick, transform_data[-1]['tick'])]

        plot_sensors = ['rgb_front', 'dvs_front', 'depth_front', 'lidar']

        with yaspin(Spinners.line, text=f"Generating episode_{episode}_sensors.png ...", color="yellow") as spinner:
            fig, axes = plt.subplots(len(plot_sensors), len(attacks), figsize=(4 * len(attacks), 4 * len(plot_sensors)))
            for row, sensor_id in enumerate(plot_sensors):
                for col, attack in enumerate(attacks):
                    ax = axes[row, col] if len(plot_sensors) > 1 else axes[col]
                    if attack in attack_intervals:
                        start_tick, _ = attack_intervals[attack][0]
                        tick = start_tick + 50
                        frame_found = False
                        while not frame_found:
                            if sensor_id == 'lidar':
                                path = os.path.join(episode_dir, sensor_id, f'lidar_{tick:06d}.npy')
                                if os.path.exists(path):
                                    points = np.load(path)
                                    img = visualize_lidar(points, 600, 800)
                                    ax.imshow(img)
                                    frame_found = True
                                    logger.debug(f"Frame found for {sensor_id} at tick: {tick}")
                                else:
                                    tick += 1
                                    if tick > start_tick + 100:  # Prevent infinite loop
                                        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                                        break
                            else:
                                path = os.path.join(episode_dir, sensor_id, f'frame_{tick:06d}.png')
                                if os.path.exists(path):
                                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if sensor_id == 'depth_front' else cv2.IMREAD_COLOR)
                                    if img is not None:
                                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB if sensor_id == 'depth_front' else cv2.COLOR_BGR2RGB)
                                        ax.imshow(img)
                                        frame_found = True
                                        logger.debug(f"Frame found for {sensor_id} at tick: {tick}")
                                    else:
                                        tick += 1
                                        if tick > start_tick + 100:  # Prevent infinite loop
                                            ax.text(0.5, 0.5, 'No image', ha='center', va='center')
                                            break
                                else:
                                    tick += 1
                                    if tick > start_tick + 100:  # Prevent infinite loop
                                        ax.text(0.5, 0.5, 'No image', ha='center', va='center')
                                        break
                    else:
                        ax.text(0.5, 0.5, 'No attack', ha='center', va='center')
                    ax.set_title(f'{sensor_id} - {attack}')
                    ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_plots_dir, f'episode_{episode}_sensors.png'))
            plt.close()
            logger.debug(f"Plotted sensor images (attacks) for episode {episode}")
        spinner.ok("✅ Done")

        # Repeat similar logic for the defense intervals part
        with yaspin(Spinners.line, text=f"Generating episode_{episode}_sensors_defenses.png ...", color="yellow") as spinner:
            fig, axes = plt.subplots(len(plot_sensors), len(attacks), figsize=(4 * len(attacks), 4 * len(plot_sensors)))
            for row, sensor_id in enumerate(plot_sensors):
                for col, attack in enumerate(attacks):
                    ax = axes[row, col] if len(plot_sensors) > 1 else axes[col]
                    if attack in attack_intervals:
                        defense_intervals = [(start, end) for a, intervals in attack_intervals.items() if a == attack for start, end in intervals if start >= len(transform_data)//2]
                        if defense_intervals:
                            start_tick, _ = defense_intervals[0]
                            tick = start_tick + 50
                            frame_found = False
                            while not frame_found:
                                if sensor_id == 'lidar':
                                    path = os.path.join(episode_dir, sensor_id, f'lidar_{tick:06d}.npy')
                                    if os.path.exists(path):
                                        points = np.load(path)
                                        img = visualize_lidar(points, 600, 800)
                                        ax.imshow(img)
                                        frame_found = True
                                        logger.debug(f"Frame found for {sensor_id} at tick: {tick}")
                                    else:
                                        tick += 1
                                        if tick > start_tick + 100:  # Prevent infinite loop
                                            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                                            break
                                else:
                                    path = os.path.join(episode_dir, sensor_id, f'frame_{tick:06d}.png')
                                    if os.path.exists(path):
                                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if sensor_id == 'depth_front' else cv2.IMREAD_COLOR)
                                        if img is not None:
                                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB if sensor_id == 'depth_front' else cv2.COLOR_BGR2RGB)
                                            ax.imshow(img)
                                            frame_found = True
                                            logger.debug(f"Frame found for {sensor_id} at tick: {tick}")
                                        else:
                                            tick += 1
                                            if tick > start_tick + 100:  # Prevent infinite loop
                                                ax.text(0.5, 0.5, 'No image', ha='center', va='center')
                                                break
                                    else:
                                        tick += 1
                                        if tick > start_tick + 100:  # Prevent infinite loop
                                            ax.text(0.5, 0.5, 'No image', ha='center', va='center')
                                            break
                        else:
                            ax.text(0.5, 0.5, 'No defense', ha='center', va='center')
                    else:
                        ax.text(0.5, 0.5, 'No attack', ha='center', va='center')
                    ax.set_title(f'{sensor_id} - {attack} (Defense)')
                    ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_plots_dir, f'episode_{episode}_sensors_defenses.png'))
            plt.close()
            logger.debug(f"Plotted sensor images (defenses) for episode {episode}")
        spinner.ok("✅ Done")

        attack_final = [data for data in transform_data if not data['defense_enabled']][-1]
        defense_final = [data for data in transform_data if data['defense_enabled']][-1]
        last_waypoint = (route_data[-1]['x'], route_data[-1]['y'])
        attack_distance = np.sqrt((attack_final['actual_x'] - last_waypoint[0])**2 + (attack_final['actual_y'] - last_waypoint[1])**2)
        defense_distance = np.sqrt((defense_final['actual_x'] - last_waypoint[0])**2 + (defense_final['actual_y'] - last_waypoint[1])**2)
        with open(os.path.join(episode_dir, 'task_completion.txt'), 'w') as f:
            f.write(f"Attack Phase Completed: {attack_distance < 5.0}\nAttack Phase Distance to Goal: {attack_distance:.2f} m\n")
            logger.debug(f"Attack Phase Completed: {attack_distance < 5.0}\nAttack Phase Distance to Goal: {attack_distance:.2f} m")
            f.write(f"Defense Phase Completed: {defense_distance < 5.0}\nDefense Phase Distance to Goal: {defense_distance:.2f} m")
            logger.debug(f"Defense Phase Completed: {defense_distance < 5.0}\nDefense Phase Distance to Goal: {defense_distance:.2f} m")
        logger.debug(f"Wrote task completion for episode {episode}")
        logger.info(f"Plots generation for episode {episode} completed successfully.")

    except Exception as e:
        logger.error(f"Error in plot_behavior for episode {episode}: {e}\n{traceback.format_exc()}")

def main(episodes, duration, num_vehicles, num_pedestrians, agent_type, sampling_rate, output_dir, comparison_plots_dir, eval_dir):
    client_search = False
    client = None
    logger.info("Searching for CARLA client...")
    while not client_search:
        try:
            client = carla.Client('localhost', 2000)
            client.set_timeout(20)  # shorter timeout for faster retry
            client.get_world()  # try to communicate with simulator
            client_search = True
            logger.info("Connected to CARLA client successfully.")
        except RuntimeError as e:
            logger.warning("CARLA client not found or simulator not ready. Retrying in 5 seconds...")
            time.sleep(5)

    client.load_world("Town05")
    world = client.get_world()
    traffic_manager = client.get_trafficmanager(8000)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    traffic_manager.set_synchronous_mode(True)
    bp_library = world.get_blueprint_library()
    vehicle_bp = bp_library.find('vehicle.tesla.model3')
    spawn_points = world.get_map().get_spawn_points()
    traffic_manager.set_global_distance_to_leading_vehicle(2.0)
    traffic_manager.set_respawn_dormant_vehicles(True)
    traffic_manager.set_hybrid_physics_mode(True)
    average_speed = 10.0
    with open('config.json', 'r') as f:
        config = json.load(f)
    original_intervals = [(attack['start'], attack['end'], attack['type']) for attack in config['attacks']]
    logger.debug(f"Loaded attack intervals: {original_intervals}")
    
    completed_episodes = 0
    try:
        import tracemalloc
        
        for episode in range(episodes):
            tracemalloc.start()
            snapshot1 = tracemalloc.take_snapshot()
            # profiler.enable()
            logger.info(f"Starting Episode {episode + 1}")
            sensors, ego_vehicle, walkers, controllers, files = [], None, [], [], []
            sensor_queues, pcla = None, None
            try:
                logger.info(f"Preparing world for Episode {episode + 1}")
                traffic_manager.set_synchronous_mode(True)
                traffic_manager.set_hybrid_physics_mode(True)
                
                episode_dir = os.path.join(eval_dir, f'episode_{episode + 1}')
                os.makedirs(episode_dir, exist_ok=True)
                with open(os.path.join(output_dir, 'metadata.txt'), 'w') as f:
                    f.write(f"agent_type: {agent_type}\n")
                    f.write(f"episodes: {episodes}\n")
                    f.write(f"duration: {duration}\n")
                    f.write(f"num_vehicles: {num_vehicles}\n")
                    f.write(f"num_pedestrians: {num_pedestrians}\n")
                    f.write(f"attack_intervals: {original_intervals}\n")
                    f.write(f"sampling_rate: {sampling_rate}\n")
                logger.info(f"Starting Episode {episode + 1}")
                required_distance = average_speed * duration
                waypoints = []
                for _ in range(10):
                    start_transform = random.choice(spawn_points)
                    start_waypoint = world.get_map().get_waypoint(start_transform.location)
                    waypoints = [start_waypoint]
                    current_waypoint = start_waypoint
                    distance = 0.0
                    while distance < required_distance:
                        next_waypoints = current_waypoint.next(2.0)
                        if not next_waypoints:
                            break
                        next_waypoint = random.choice(next_waypoints)
                        distance += current_waypoint.transform.location.distance(next_waypoint.transform.location)
                        waypoints.append(next_waypoint)
                        current_waypoint = next_waypoint
                    if distance >= required_distance:
                        break
                with open(os.path.join(episode_dir, 'route.csv'), 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['waypoint_id', 'x', 'y', 'z'])
                    for i, wp in enumerate(waypoints):
                        loc = wp.transform.location
                        writer.writerow([i, loc.x, loc.y, loc.z])
                logger.debug(f"Wrote route.csv with {len(waypoints)} waypoints")
                route_file = os.path.join(episode_dir, 'route.xml')
                route_maker(waypoints, route_file)
                
                try:
                    ego_vehicle = world.spawn_actor(vehicle_bp, start_transform)
                    logger.debug("Spawned ego vehicle")
                except Exception as e:
                    logger.error(f"Failed to spawn ego vehicle for Episode {episode + 1}: {e}\n{traceback.format_exc()}")
                    raise
                
                def get_transform_wrapper(self, tick, route_points=None, enable_defenses=False):
                    if attack_state.current_attack == 'gps':
                        true_transform = self._original_get_transform()
                        time_factor = min((tick - 400) / 100.0, 1.0)
                        sine_factor = np.sin(tick * 0.05) * 2.0
                        spoofed_location = carla.Location(
                            x=true_transform.location.x + 5.0 * time_factor + sine_factor,
                            y=true_transform.location.y + 7.0 * time_factor + sine_factor,
                            z=true_transform.location.z
                        )
                        if enable_defenses and route_points is not None:
                            P = np.array([spoofed_location.x, spoofed_location.y])
                            min_distance = min(point_to_segment_distance(P, route_points[i], route_points[i + 1]) 
                                              for i in range(len(route_points) - 1))
                            # Velocity consistency check
                            velocity = self.get_velocity()
                            speed = np.sqrt(velocity.x**2 + velocity.y**2)
                            if min_distance > 5.0 or speed > 20.0:  # Unrealistic speed or deviation
                                logger.debug(f"GPS defense: reverted to true transform, min_distance={min_distance}, speed={speed}")
                                return true_transform
                        logger.debug(f"GPS attack: spoofed_location={spoofed_location}")
                        return carla.Transform(spoofed_location, true_transform.rotation)
                    return self._original_get_transform()
                ego_vehicle._original_get_transform = ego_vehicle.get_transform
                spectator = world.get_spectator()
                spectator.set_transform(carla.Transform(
                    ego_vehicle.get_location() + carla.Location(x=-8, z=7), carla.Rotation(pitch=-19)))
                
                traffic_vehicles = []
                for _ in range(num_vehicles):
                    vehicle_bp = random.choice(bp_library.filter('vehicle'))
                    spawn_point = None
                    for _ in range(10):  # Try up to 10 different spawn points
                        spawn_point = random.choice(spawn_points)
                        if world.get_map().get_spawn_points():  # Validate spawn point
                            vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
                            if vehicle:
                                break
                    if vehicle:
                        try:
                            vehicle.set_autopilot(True, traffic_manager.get_port())
                            traffic_vehicles.append(vehicle)
                            logger.debug(f"Spawned additional vehicle (ID: {vehicle.id}) at Episode {episode + 1}")
                        except Exception as ve:
                            logger.warning(f"Failed to enable autopilot for vehicle: {ve}")
                            vehicle.destroy()
                    else:
                        logger.warning(f"Failed to spawn additional vehicle for Episode {episode + 1} after multiple attempts")

                for _ in range(num_pedestrians):
                    walker_bp = random.choice(bp_library.filter('walker.pedestrian.*'))
                    spawn_location = None
                    for _ in range(10):  # Try up to 10 different locations
                        spawn_location = world.get_random_location_from_navigation()
                        if spawn_location:
                            spawn_location.z += 0.6
                            walker = world.try_spawn_actor(walker_bp, carla.Transform(spawn_location))
                            if walker:
                                break
                    try:
                        controller = world.spawn_actor(bp_library.find('controller.ai.walker'), 
                                                        carla.Transform(), walker)
                        controller.start()
                        controller.go_to_location(world.get_random_location_from_navigation())
                        walkers.append(walker)
                        controllers.append(controller)
                        logger.debug(f"Spawned pedestrian (ID: {walker.id}) and controller for Episode {episode + 1}")
                    except Exception as ce:
                        walker.destroy()
                        logger.warning(f"Failed to spawn walker controller: {ce}")
                    
                sensors, sensor_queues, writers, files, sensors_to_spawn, sensor_interface = setup_sensors(
                    world, ego_vehicle, episode_dir, agent_type, enable_defenses=False)
                for _ in range(2):
                    world.tick()
                    time.sleep(0.1)
                logger.info("Sensors stabilized")
                
                pcla = PCLA(agent_type, ego_vehicle, route_file, client)
                logger.info(f"Initialized PCLA with agent_type={agent_type}")

                all_actors = world.get_actors()
                sensors = all_actors.filter('sensor.*')
                vehicles = all_actors.filter('vehicle.*')
                pedestrians = all_actors.filter('walker.pedestrian.*')
                logger.info(f"Found {len(sensors)} sensors, {len(vehicles)} vehicles, and {len(pedestrians)} pedestrians in the world")
                
                ticks_per_phase = int(duration / settings.fixed_delta_seconds / 2)
                ticks_per_episode = ticks_per_phase * 2
                phase1_intervals = original_intervals
                phase2_intervals = [(start + ticks_per_phase, end + ticks_per_phase, attack_type) 
                                    for start, end, attack_type in original_intervals]
                all_intervals = phase1_intervals + phase2_intervals
                route_points = np.array([[wp.transform.location.x, wp.transform.location.y] for wp in waypoints])
                
                steering_history = []  # For steering bias defense
                for tick in tqdm(range(ticks_per_episode), desc=f"Episode {episode + 1}"):
                    enable_defenses = tick >= ticks_per_phase
                    for start, end, attack in all_intervals:
                        if start <= tick < end:
                            attack_state.current_attack = attack
                            break
                        else:
                            attack_state.current_attack = 'none'
                    logger.debug(f"Tick {tick}: attack_state={attack_state.current_attack}, defenses={enable_defenses}")
                    
                    ego_vehicle.get_transform = MethodType(
                        lambda self: get_transform_wrapper(self, tick, route_points, enable_defenses), ego_vehicle)
                    
                    save_sensor_data(sensor_queues, episode_dir, tick, writers, ego_vehicle, world, 
                                    sensors_to_spawn, sampling_rate, sensor_interface, enable_defenses)
                    
                    for q in sensor_queues.values():
                        while not q.empty():
                            try:
                                q.get_nowait()
                            except queue.Empty:
                                pass
                    logger.debug(f"Tick {tick}: cleared sensor queues")
                    
                    try:
                        tick_data = sensor_interface.get_data()
                        logger.debug(f"Tick {tick}: sensor_interface.get_data() returned keys={list(tick_data.keys())}")
                        for k, v in tick_data.items():
                            logger.debug(f"Tick {tick}: sensor {k} type={type(v)}, value={v if not hasattr(v, 'shape') else f'shape={v.shape}, dtype={v.dtype}' if hasattr(v, 'dtype') else f'shape={v.shape}'}")
                            if isinstance(v, tuple) and len(v) > 1:
                                logger.debug(f"Tick {tick}: sensor {k} tuple[1] shape={v[1].shape if hasattr(v[1], 'shape') else 'N/A'}, dtype={v[1].dtype if hasattr(v[1], 'dtype') else 'N/A'}")
                        
                        required_sensors = ['rgb', 'rgb_front', 'depth_front', 'gps', 'imu']
                        missing_sensors = [s for s in required_sensors if s not in tick_data]
                        if missing_sensors:
                            logger.warning(f"Tick {tick}: Missing sensors: {missing_sensors}")
                            for sensor in missing_sensors:
                                if sensor == 'rgb':
                                    tick_data[sensor] = (0, np.zeros((300, 400, 3), dtype=np.uint8))
                                    logger.debug(f"Tick {tick}: Added fallback for rgb, shape=(300, 400, 3)")
                                elif sensor == 'rgb_front':
                                    tick_data[sensor] = (0, np.zeros((600, 800, 3), dtype=np.uint8))
                                    logger.debug(f"Tick {tick}: Added fallback for rgb_front, shape=(600, 800, 3)")
                                elif sensor == 'depth_front':
                                    tick_data[sensor] = (0, np.zeros((600, 800), dtype=np.float32))
                                    logger.debug(f"Tick {tick}: Added fallback for depth_front, shape=(600, 800)")
                                elif sensor == 'gps':
                                    tick_data[sensor] = (0, np.array([0.0, 0.0, 0.0], dtype=np.float64))
                                    logger.debug(f"Tick {tick}: Added fallback for gps, shape=(3,)")
                                elif sensor == 'imu':
                                    tick_data[sensor] = (0, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
                                    logger.debug(f"Tick {tick}: Added fallback for imu, shape=(7,)")
                        
                        tick_data_processed = {}
                        for sensor in ['rgb', 'rgb_front']:
                            if sensor in tick_data:
                                data = tick_data[sensor]
                                expected_shape = (300, 400, 3) if sensor == 'rgb' else (600, 800, 3)
                                frame = 0
                                if isinstance(data, tuple) and len(data) > 1:
                                    frame, array = data[0], data[1]
                                else:
                                    array = data
                                logger.debug(f"Tick {tick}: Processing {sensor}, initial shape={array.shape if hasattr(array, 'shape') else 'N/A'}")
                                if hasattr(array, 'shape'):
                                    if len(array.shape) == 2:
                                        logger.warning(f"Tick {tick}: {sensor} is 2D (shape={array.shape}), converting to 3D")
                                        array = np.stack([array, array, array], axis=-1).astype(np.uint8)
                                        logger.debug(f"Tick {tick}: Converted {sensor} to shape={array.shape}")
                                    elif len(array.shape) == 3 and array.shape != expected_shape:
                                        logger.warning(f"Tick {tick}: {sensor} has shape {array.shape}, resizing to {expected_shape}")
                                        array = cv2.resize(array, expected_shape[1::-1])[:, :, :3].astype(np.uint8)
                                        logger.debug(f"Tick {tick}: Resized {sensor} to shape={array.shape}")
                                    elif len(array.shape) != 3:
                                        logger.error(f"Tick {tick}: {sensor} has invalid shape {array.shape}, using fallback")
                                        array = np.zeros(expected_shape, dtype=np.uint8)
                                else:
                                    logger.error(f"Tick {tick}: {sensor} has no shape, using fallback")
                                    array = np.zeros(expected_shape, dtype=np.uint8)
                                tick_data_processed[sensor] = (frame, array)
                                logger.debug(f"Tick {tick}: Set {sensor} to tuple, shape={array.shape}")
                        
                        if 'depth_front' in tick_data:
                            data = tick_data['depth_front']
                            expected_shape = (600, 800)
                            frame = 0
                            if isinstance(data, tuple) and len(data) > 1:
                                frame, array = data[0], data[1]
                            else:
                                array = data
                            logger.debug(f"Tick {tick}: Processing depth_front, initial shape={array.shape if hasattr(array, 'shape') else 'N/A'}")
                            if hasattr(array, 'shape'):
                                if len(array.shape) != 2 or array.shape != expected_shape:
                                    logger.warning(f"Tick {tick}: depth_front has shape {array.shape}, resizing to {expected_shape}")
                                    array = cv2.resize(array, expected_shape[::-1]).astype(np.float32)
                                    logger.debug(f"Tick {tick}: Resized depth_front to shape={array.shape}")
                            else:
                                logger.error(f"Tick {tick}: depth_front has no shape, using fallback")
                                array = np.zeros(expected_shape, dtype=np.float32)
                            tick_data_processed['depth_front'] = (frame, array)
                            logger.debug(f"Tick {tick}: Set depth_front to tuple, shape={array.shape}")
                        
                        for sensor in ['gps', 'imu']:
                            if sensor in tick_data:
                                data = tick_data[sensor]
                                if isinstance(data, tuple) and len(data) > 1:
                                    tick_data_processed[sensor] = data
                                else:
                                    tick_data_processed[sensor] = (0, data)
                                logger.debug(f"Tick {tick}: Processed {sensor}, shape={tick_data_processed[sensor][1].shape if hasattr(tick_data_processed[sensor][1], 'shape') else 'N/A'}")
                        
                        tick_data_processed['speed'] = (0, {'speed': ego_vehicle.get_velocity().length()})
                        logger.debug(f"Tick {tick}: Added speed sensor, value={tick_data_processed['speed'][1]['speed']}")
                        
                        tick_data = tick_data_processed
                        logger.debug(f"Tick {tick}: Final tick_data keys={list(tick_data.keys())}")
                        
                        logger.debug(f"Tick {tick}: Calling pcla.agent_instance.run_step with timestamp={world.get_snapshot().timestamp.elapsed_seconds}")
                        ego_action = pcla.agent_instance.run_step(tick_data, world.get_snapshot().timestamp.elapsed_seconds)
                        logger.debug(f"Tick {tick}: Agent returned action: throttle={ego_action.throttle}, steer={ego_action.steer}, brake={ego_action.brake}")
                        
                        # Steering bias defense
                        if enable_defenses and attack_state.current_attack == 'steering_bias':
                            steering_history.append(ego_action.steer)
                            if len(steering_history) > 50:  # Analyze last 50 ticks
                                steering_history.pop(0)
                                z_scores = stats.zscore(steering_history)
                                if abs(z_scores[-1]) > 3:  # Anomaly detection
                                    ego_action.steer = np.mean(steering_history)  # Revert to mean
                                    logger.debug(f"Steering bias defense: Corrected steer from {ego_action.steer} to {np.mean(steering_history)}")
                        
                        writers['control'].writerow([world.get_snapshot().timestamp.elapsed_seconds, 
                                                    ego_action.throttle, ego_action.steer, ego_action.brake])
                        
                        ego_vehicle.apply_control(carla.VehicleControl(
                            throttle=ego_action.throttle,
                            steer=ego_action.steer,
                            brake=ego_action.brake
                        ))
                        logger.debug(f"Tick {tick}: Applied control to vehicle: throttle={ego_action.throttle}, steer={ego_action.steer}, brake={ego_action.brake}")

                    except Exception as e:
                        logger.error(f"Error at tick {tick}: {e}\n{traceback.format_exc()}")
                        tick_data = {
                            'rgb': (0, np.zeros((300, 400, 3), dtype=np.uint8)),
                            'rgb_front': (0, np.zeros((600, 800, 3), dtype=np.uint8)),
                            'depth_front': (0, np.zeros((600, 800), dtype=np.float32)),
                            'gps': (0, np.array([0.0, 0.0, 0.0], dtype=np.float64)),
                            'imu': (0, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)),
                            'speed': (0, {'speed': ego_vehicle.get_velocity().length()})
                        }
                        logger.debug(f"Tick {tick}: Using fallback tick_data with keys={list(tick_data.keys())}")
                        try:
                            logger.debug(f"Tick {tick}: Attempting fallback agent run_step")
                            ego_action = pcla.agent_instance.run_step(tick_data, world.get_snapshot().timestamp.elapsed_seconds)
                            writers['control'].writerow([world.get_snapshot().timestamp.elapsed_seconds, 
                                                        ego_action.throttle, ego_action.steer, ego_action.brake])
                            ego_vehicle.apply_control(carla.VehicleControl(
                                throttle=ego_action.throttle,
                                steer=ego_action.steer,
                                brake=ego_action.brake
                            ))
                            logger.debug(f"Tick {tick}: Applied fallback control to vehicle: throttle={ego_action.throttle}, steer={ego_action.steer}, brake={ego_action.brake}")
                        except Exception as inner_e:
                            logger.error(f"Fallback action failed at tick {tick}: {inner_e}\n{traceback.format_exc()}")
                            continue
                    world.tick()
                
                generate_combined_video(episode_dir, episode + 1)
                for file in files:
                    file.close()
                files = []  # Clear file list to prevent reuse
                # p = Process(target=plot_behavior_async, args=(episode_dir, comparison_plots_dir, episode + 1))
                # p.start()
                # p.join()
                plot_behavior(episode_dir, comparison_plots_dir, episode + 1)
                completed_episodes += 1
                logger.info(f"Episode {episode + 1} completed")
            
            except Exception as e:
                logger.error(f"Episode {episode + 1} failed: {e}\n{traceback.format_exc()}")
                continue
            finally:
                try:
                    import psutil

                    process = psutil.Process(os.getpid())

                    cpu_usage = process.cpu_percent(interval=1)  # CPU usage over 1 second
                    mem_usage_percent = psutil.Process().memory_percent()

                    logger.info(f"Before cleanups - episode {episode + 1}: CPU usage: {cpu_usage}%, Memory usage: {mem_usage_percent}%")


                    all_actors = world.get_actors()
                    sensors = all_actors.filter('sensor.*')
                    logger.info(f"Active sensors before cleanup: {len(sensors)}")
                    
                    # Step 1: Stop and destroy sensors
                    for sensor in sensors:
                        try:
                            sensor.stop()
                            if sensor.destroy():
                                logger.info(f"Destroyed sensor {sensor.type_id} (ID: {sensor.id})")
                            else:
                                logger.warning(f"Failed to destroy sensor {sensor.type_id} (ID: {sensor.id})")
                        except Exception as se:
                            logger.warning(f"Failed to destroy sensor {sensor.type_id}: {se}")

                    # Step 1: Destroy ego vehicle
                    if ego_vehicle:
                        try:
                            if ego_vehicle.destroy():
                                time.sleep(1)
                                logger.info("Destroyed ego vehicle")
                            else:
                                logger.warning("Failed to destroy ego vehicle")
                        except Exception as ee:
                            logger.warning(f"Failed to destroy ego vehicle: {ee}")
                    else:
                        logger.warning("Ego vehicle is not alive or does not exist, skipping destruction")

                    all_actors = world.get_actors()
                    vehicles = all_actors.filter('vehicle.*')
                    logger.info(f"Active vehicles before cleanup: {len(vehicles)}")
                    
                    # Step 3: Destroy traffic vehicles
                    for vehicle in vehicles:
                        try:
                            vehicle.set_autopilot(False)
                            if vehicle.destroy():
                                time.sleep(1)
                                logger.info(f"Destroyed traffic vehicle (ID: {vehicle.id})")
                            else:
                                logger.warning(f"Failed to destroy traffic vehicle (ID: {vehicle.id})")
                        except Exception as ve:
                            logger.warning(f"Failed to destroy traffic vehicle: {ve}")

                    # Step 4: Stop and destroy pedestrians and controllers

                    all_actors = world.get_actors()
                    pedestrians = all_actors.filter('walker.pedestrian.*')
                    logger.info(f"Active pedestrians before cleanup: {len(pedestrians)}")

                    for walker in pedestrians:
                        try:
                            if hasattr(walker, 'controller'):
                                walker.controller.stop()
                                if walker.controller.destroy():
                                    time.sleep(1)
                                    logger.info(f"Destroyed controller for walker (ID: {walker.id})")
                                else:
                                    logger.warning(f"Failed to destroy controller for walker (ID: {walker.id})")
                            if walker.destroy():
                                time.sleep(1)
                                logger.info(f"Destroyed walker (ID: {walker.id})")
                            else:
                                logger.warning(f"Failed to destroy walker (ID: {walker.id})")
                        except Exception as we:
                            logger.warning(f"Failed to destroy walker: {we}")

                    for name in ['sensors','vehicles','pedestrians']:
                        del locals()[name]
                    if ego_vehicle:
                        ego_vehicle = None

                            
                    # Step 5: Check and log active actors before cleanup
                    time.sleep(5)  # Longer delay
                    world.tick()

                    all_actors = world.get_actors()
                    sensors = all_actors.filter('sensor.*')
                    vehicles = all_actors.filter('vehicle.*')
                    pedestrians = all_actors.filter('walker.pedestrian.*')
                    logger.info(f"Active sensors before cleanup: {len(sensors)}")
                    logger.info(f"Active vehicles before cleanup: {len(vehicles)}")
                    logger.info(f"Active pedestrians before cleanup: {len(pedestrians)}")
                    
                    # Step 6: Cleanup PCLA agent if initialized
                    if hasattr(pcla, 'cleanup'):
                        try:
                            pcla.cleanup()
                            time.sleep(5)
                            pcla = None
                            logger.info("Cleaned up PCLA agent")
                        except Exception as pe:
                            logger.warning(f"Failed to clean up PCLA: {pe}")
                    if pcla:
                        del pcla


                    # Step 7: Clear sensor queues
                    if sensor_queues:
                        for q in sensor_queues.values():
                            while not q.empty():
                                try:
                                    q.get_nowait()
                                except queue.Empty:
                                    pass
                        sensor_queues = {}
                        logger.debug("Cleared sensor queues")
                    if sensor_queues:
                        del sensor_queues

                    
                    # Step 8: Close file handles
                    for file in files:
                        try:
                            file.close()
                            logger.debug("Closed file handle")
                        except Exception as fe:
                            logger.warning(f"Failed to close file: {fe}")
                    if files:
                        files = []
                        del files 
                    if writers:
                        writers = {} 
                        del writers

                    if sensor_interface:
                        del sensor_interface     

                    import gc
                    gc.collect()
                    logger.debug("Garbage collection triggered")
                    time.sleep(5)  # Allow time for cleanup

                    snapshot2 = tracemalloc.take_snapshot()
                    for stat in snapshot2.compare_to(snapshot1, 'lineno')[:10]:
                        logger.info(f"Memory leak detected: {stat}")
                    tracemalloc.stop()

                    cpu_usage = process.cpu_percent(interval=1)  # CPU usage over 1 second
                    mem_usage_percent = psutil.Process().memory_percent()

                    logger.info(f"After cleanups - episode {episode + 1}: CPU usage: {cpu_usage}%, Memory usage: {mem_usage_percent}%")
                    
                    # Step 9: Reset spectator
                    try:
                        spectator = world.get_spectator()
                        spectator.set_transform(carla.Transform(carla.Location(x=0, y=0, z=50)))
                        logger.debug("Reset spectator transform")
                    except Exception as spe:
                        logger.warning(f"Failed to reset spectator: {spe}")
                    
                    # Step 10: Reload the world
                    try:
                        client.reload_world()
                        time.sleep(5)  # Increased delay to ensure reload completes
                        logger.info("World reloaded successfully")
                    except Exception as re:
                        logger.error(f"Failed to reload world: {re}")
                    
                    # Step 11: Verify world reload
                    world = client.get_world()
                    all_actors_after = world.get_actors()
                    if len(all_actors_after) > 0:
                        logger.warning(f"Found {len(all_actors_after)} actors after reload: {[actor.type_id for actor in all_actors_after]}")
                    else:
                        logger.info("No actors found after reload - world is clean")
                    
                    # Step 13: Check simulator responsiveness
                    try:
                        spectator = world.get_spectator()
                        transform = spectator.get_transform()
                        logger.info("Simulator is responsive after reload")
                    except Exception as e:
                        logger.error(f"Simulator unresponsive after reload: {e}")
                    
                    # Step 14: Reapply settings for the next episode
                    settings = carla.WorldSettings(
                        synchronous_mode=True,
                        fixed_delta_seconds=0.05
                    )
                    world.apply_settings(settings)
                    traffic_manager.set_synchronous_mode(True)
                    logger.info("Reapplied world settings for next episode")
                    
                    # Step 12: Verify world settings
                    settings = world.get_settings()
                    if not settings.synchronous_mode or settings.fixed_delta_seconds != 0.05:
                        logger.warning(f"World settings not applied correctly: synchronous_mode={settings.synchronous_mode}, fixed_delta_seconds={settings.fixed_delta_seconds}")
                    else:
                        logger.info("World settings verified")
                
                except Exception as cleanup_e:
                    logger.error(f"Cleanup failed for Episode {episode + 1}: {cleanup_e}\n{traceback.format_exc()}")

    finally:
        try:
            settings.synchronous_mode = False
            world.apply_settings(settings)
            logger.info('Simulation cleaned up')
        except Exception as final_e:
            logger.error(f"Final cleanup failed: {final_e}\n{traceback.format_exc()}")
        if args.generate_analytics and completed_episodes > 0:
            logger.info(f"Generating analytics for {completed_episodes} completed episodes")
            generate_aggregate_analytics(eval_dir)
        elif completed_episodes == 0:
            logger.warning("No episodes completed successfully, skipping analytics generation")
    return eval_dir

def generate_aggregate_analytics(eval_dir):
    try:
        import aggregate_analytics
        output_dir = os.path.join(eval_dir, 'aggregate_results')
        os.makedirs(output_dir, exist_ok=True)
        episode_dirs = [root for root, _, files in os.walk(eval_dir) if 'transform.csv' in files and 'control.csv' in files]
        logger.debug(f"Found {len(episode_dirs)} episode directories for analytics")
        if not episode_dirs:
            logger.error(f"No episode directories in {eval_dir}")
            return
        all_episodes_data = []
        for episode_dir in tqdm(episode_dirs, desc="Loading data"):
            try:
                episode_data = aggregate_analytics.load_episode_data(episode_dir)
                episode_data['_source_path'] = episode_dir
                all_episodes_data.append(episode_data)
                logger.debug(f"Loaded analytics data from {episode_dir}")
            except Exception as e:
                logger.error(f"Error loading {episode_dir}: {e}\n{traceback.format_exc()}")
        if not all_episodes_data:
            logger.error("No valid episode data")
            return
        for func, desc in [
            (aggregate_analytics.plot_aggregate_speed_by_attack, "speed plots"),
            (aggregate_analytics.plot_aggregate_deviation_by_attack, "deviation plots"),
            (aggregate_analytics.plot_sensor_reliability_metrics, "sensor plots"),
            (aggregate_analytics.plot_recovery_analysis, "recovery plots"),
            (aggregate_analytics.plot_combined_trajectory_heatmap, "trajectory heatmap"),
            (aggregate_analytics.plot_task_completion_stats, "completion stats"),
            (aggregate_analytics.plot_control_input_analysis, "control analysis"),
            (aggregate_analytics.plot_phase_transition_analysis, "phase analysis"),
            (aggregate_analytics.plot_attack_impact, "attack impact"),
            (aggregate_analytics.plot_distance_to_goal, "distance to goal")
        ]:
            logger.info(f"Generating {desc}")
            func(all_episodes_data, output_dir)
        logger.info("Creating dashboard")
        aggregate_analytics.create_aggregate_dashboard(output_dir)
        logger.info(f"Analytics saved to {output_dir}")
    except Exception as e:
        logger.error(f"Analytics error: {e}\n{traceback.format_exc()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CARLA Simulation with Attacks')
    parser.add_argument('--episodes', type=int, default=4)
    parser.add_argument('--duration', type=float, default=360)
    parser.add_argument('--num_vehicles', type=int, default=40)
    parser.add_argument('--num_pedestrians', type=int, default=40)
    parser.add_argument('--agent_type', type=str, default='neat_aim2ddepth')
    parser.add_argument('--sampling_rate', type=int, default=1)
    parser.add_argument('--generate_analytics', action='store_true')
    parser.add_argument('--analytics_only', action='store_true')
    parser.add_argument('--eval_dir', type=str)
    parser.add_argument('--detailed_logging', action='store_true', help='Enable detailed DEBUG logging to file')
    args = parser.parse_args()
    try:
        setup_logging(args.detailed_logging)
        if args.analytics_only:
            generate_aggregate_analytics(args.eval_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.expanduser(f"~/output_{args.agent_type}_{timestamp}")
            # output_dir = os.path.expanduser(f"~/test_dataset_carla_pcla_anomalib")
            # output_dir = os.path.expanduser(f"~/training_dataset_carla_pcla_anomalib_v2")
            os.makedirs(output_dir, exist_ok=True)
            comparison_plots_dir = os.path.join(output_dir, 'comparison_plots')
            os.makedirs(comparison_plots_dir, exist_ok=True)
            eval_dir = os.path.join(output_dir, 'eval')
            os.makedirs(eval_dir, exist_ok=True)
            logger.debug(f"Starting simulation with output_dir={output_dir}")

            main(args.episodes, args.duration, args.num_vehicles, args.num_pedestrians, args.agent_type, 
                args.sampling_rate, output_dir, comparison_plots_dir, eval_dir)

    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Main execution error: {e}\n{traceback.format_exc()}")
    finally:
        logger.info('Done.')