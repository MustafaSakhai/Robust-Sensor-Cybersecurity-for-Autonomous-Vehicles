import os
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
from anomalib.data import Folder
from anomalib.models import Patchcore, Padim, EfficientAd, Fastflow
from anomalib.engine import Engine
import matplotlib.pyplot as plt
from torchvision import transforms
import gc
import logging
from lightning.pytorch.callbacks import Callback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set memory optimization settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision('medium')

class MetricsCollector(Callback):
    """Custom callback to collect metrics during training and testing."""
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.predictions = []
        self.current_epoch_losses = []
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Collect training metrics."""
        if isinstance(outputs, dict) and 'loss' in outputs:
            self.current_epoch_losses.append(outputs['loss'].item())
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Process training metrics at epoch end."""
        if self.current_epoch_losses:
            epoch_loss = np.mean(self.current_epoch_losses)
            self.train_losses.append(epoch_loss)
            self.current_epoch_losses = []
            logger.info(f"Epoch {trainer.current_epoch}: Training Loss = {epoch_loss:.4f}")
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Collect validation metrics."""
        if isinstance(outputs, dict) and 'loss' in outputs:
            self.val_losses.append(outputs['loss'].item())
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Collect test predictions from ImageBatch."""
        if outputs is not None:
            if hasattr(outputs, 'pred_scores') and isinstance(outputs.pred_scores, torch.Tensor):
                pred_scores = outputs.pred_scores.cpu().numpy()
                # Handle both single and batch predictions
                if pred_scores.ndim > 0:
                    self.predictions.extend(pred_scores.flatten())
                    for score in pred_scores.flatten():
                        logger.info(f"Collected pred_score: {score:.6f}")
                else:
                    self.predictions.append(float(pred_scores))
                    logger.info(f"Collected pred_score: {float(pred_scores):.6f}")
            elif hasattr(outputs, 'pred_score') and isinstance(outputs.pred_score, torch.Tensor):
                pred_score = outputs.pred_score.cpu().numpy()
                if pred_score.ndim > 0:
                    self.predictions.extend(pred_score.flatten())
                    for score in pred_score.flatten():
                        logger.info(f"Collected pred_score: {score:.6f}")
                else:
                    self.predictions.append(float(pred_score))
                    logger.info(f"Collected pred_score: {float(pred_score):.6f}")
            else:
                logger.warning(f"outputs does not have prediction scores or they are not tensors. Type: {type(outputs)}")
        else:
            logger.warning("outputs is None")
            
def visualize_lidar(points, height=600, width=800, range_max=50.0):
    """Convert LiDAR point cloud to a 2D image."""
    if points.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    distances = np.linalg.norm(points[:, :3], axis=1)
    points = points[distances < range_max]
    if points.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    x = points[:, 0]
    y = points[:, 1]
    x_img = np.clip(((x + range_max) / (2 * range_max)) * (width - 1), 0, width - 1).astype(int)
    y_img = np.clip(((y + range_max) / (2 * range_max)) * (height - 1), 0, height - 1).astype(int)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[y_img, x_img] = [255, 255, 255]
    return img

def process_lidar_data(data_dir, output_dir=None):
    """Process LiDAR .npy files to generate 2D images for anomaly detection.
    
    Args:
        data_dir: Directory containing episode directories with LiDAR data
        output_dir: Optional output directory for processed images
    """
    data_dir = Path(data_dir)
    if output_dir is None:
        output_dir = data_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    # Process each episode directory
    episode_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('episode_')]
    if not episode_dirs:
        # If no episode directories found, treat as single episode
        episode_dirs = [data_dir]
    
    for episode_dir in episode_dirs:
        logger.info(f"Processing LiDAR data for {episode_dir.name}")
        
        lidar_dir = episode_dir / 'lidar'
        if not lidar_dir.exists():
            logger.warning(f"LiDAR directory not found at {lidar_dir}")
            continue
        
        lidar_images_dir = episode_dir / 'lidar_images'
        lidar_images_dir.mkdir(exist_ok=True)
        
        logger.info(f"Processing LiDAR data from {lidar_dir} to {lidar_images_dir}")
        lidar_files = list(lidar_dir.glob('*.npy'))
        logger.info(f"Found {len(lidar_files)} LiDAR files")
        
        for lidar_file in lidar_files:
            img_path = lidar_images_dir / lidar_file.name.replace('.npy', '.png')
            if img_path.exists():
                continue
            
            try:
                points = np.load(lidar_file)
                img = visualize_lidar(points)
                cv2.imwrite(str(img_path), img[:, :, ::-1])  # Convert RGB to BGR for OpenCV
            except Exception as e:
                logger.error(f"Error processing {lidar_file}: {e}")
    
    return output_dir

def create_dataset_structure(data_dir, output_dir, sensors, train_ratio=0.7, attack_frame_threshold=None, normal_only=False):
    """Create dataset structure for anomalib training and testing.
    
    Args:
        data_dir: Directory containing sensor data (eval directory with multiple episodes)
        output_dir: Directory to create dataset structure
        sensors: List of sensor names to process
        train_ratio: Ratio of data to use for training (from non-attack frames)
        attack_frame_threshold: Frame number threshold to separate normal/attack frames
                               If None, will use train_ratio to split data
        normal_only: If True, assumes all data is normal and creates synthetic anomalies
    """
    logger.info(f"Creating dataset structure in {output_dir} for sensors: {sensors}")
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get all episode directories
    episode_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('episode_')]
    if not episode_dirs:
        logger.warning(f"No episode directories found in {data_dir}")
        return
    
    logger.info(f"Found {len(episode_dirs)} episode directories")
    
    # Process each episode
    for episode_dir in sorted(episode_dirs):
        logger.info(f"Processing episode: {episode_dir.name}")
        
        # Process LiDAR data first if needed
        if 'lidar' in sensors:
            process_lidar_data(episode_dir)
        
        # Create dataset directories
        for sensor in sensors:
            logger.info(f"Creating dataset structure for {sensor} in {episode_dir.name}")
            
            # Determine source directory name (special case for lidar)
            source_dir_name = 'lidar_images' if sensor == 'lidar' else sensor
            source_dir = episode_dir / source_dir_name
            
            if not source_dir.exists():
                logger.warning(f"Source directory {source_dir} not found, skipping {sensor}")
                continue
            
            # Create dataset directories
            sensor_dataset_dir = output_dir / sensor
            train_good_dir = sensor_dataset_dir / 'train' / 'good'
            test_good_dir = sensor_dataset_dir / 'test' / 'good'
            test_anomaly_dir = sensor_dataset_dir / 'test' / 'anomaly'
            
            train_good_dir.mkdir(parents=True, exist_ok=True)
            test_good_dir.mkdir(parents=True, exist_ok=True)
            test_anomaly_dir.mkdir(parents=True, exist_ok=True)
            # logger.info(f"Created dataset structure {train_good_dir}, {test_good_dir}, {test_anomaly_dir}")
            
            # Get all image files
            image_files = sorted(list(source_dir.glob('*.png')))
            if not image_files:
                logger.warning(f"No images found in {source_dir}")
                continue
            
            logger.info(f"Found {len(image_files)} images for {sensor} in {episode_dir.name}")
            
            # Split data based on frame numbers
            if normal_only:
                # When we only have normal data, split it into train and test sets
                split_idx = int(len(image_files) * train_ratio)
                train_frames = image_files[:split_idx]
                test_normal_frames = image_files[split_idx:]
                
                # Create synthetic anomalies from test frames
                test_attack_frames = test_normal_frames.copy()
            else:
                if attack_frame_threshold is not None:
                    # Split based on frame number threshold
                    normal_frames = []
                    attack_frames = []
                    
                    for img_file in image_files:
                        frame_num = int(img_file.stem.split('_')[-1])
                        if frame_num < attack_frame_threshold:
                            normal_frames.append(img_file)
                        else:
                            attack_frames.append(img_file)
                    
                    # Split normal frames into train and test
                    split_idx = int(len(normal_frames) * train_ratio)
                    train_frames = normal_frames[:split_idx]
                    test_normal_frames = normal_frames[split_idx:]
                    test_attack_frames = attack_frames
                else:
                    # Split all frames using train_ratio
                    split_idx = int(len(image_files) * train_ratio)
                    train_frames = image_files[:split_idx]
                    test_frames = image_files[split_idx:]
                    
                    # For testing purposes, consider the second half of test frames as anomalous
                    test_split_idx = len(test_frames) // 2
                    test_normal_frames = test_frames[:test_split_idx]
                    test_attack_frames = test_frames[test_split_idx:]
            
            logger.info(f"{sensor} in {episode_dir.name}: {len(train_frames)} training, "
                       f"{len(test_normal_frames)} test normal, {len(test_attack_frames)} test anomaly images")
            
            # Create symbolic links to the original files with episode prefix
            episode_prefix = episode_dir.name.replace('episode_', '')
            
            for i, img_file in enumerate(train_frames):
                dst = train_good_dir / f"train_ep{episode_prefix}_{i:04d}_{img_file.name}"
                if not dst.exists():
                    os.symlink(img_file.absolute(), dst)
            
            for i, img_file in enumerate(test_normal_frames):
                dst = test_good_dir / f"test_normal_ep{episode_prefix}_{i:04d}_{img_file.name}"
                if not dst.exists():
                    os.symlink(img_file.absolute(), dst)
            
            for i, img_file in enumerate(test_attack_frames):
                dst = test_anomaly_dir / f"test_anomaly_ep{episode_prefix}_{i:04d}_{img_file.name}"
                if not dst.exists():
                    os.symlink(img_file.absolute(), dst)

def create_synthetic_anomalies(image_path):
    """Create synthetic anomalies by applying various transformations to normal images."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply random transformations
    anomaly_type = np.random.choice(['noise', 'blur', 'intensity'])
    
    if anomaly_type == 'noise':
        # Add random noise
        noise = np.random.normal(0, 50, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
    elif anomaly_type == 'blur':
        # Apply strong blur
        kernel_size = np.random.choice([7, 9, 11])
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    else:  # intensity
        # Modify intensity
        factor = np.random.uniform(0.5, 1.5)
        img = cv2.convertScaleAbs(img, alpha=factor, beta=0)
    
    # Convert back to BGR for saving
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def train_anomaly_models(dataset_dir, results_dir, sensors, backbone="resnet18", epochs=10, normal_only=False):
    """Train anomaly detection models for each sensor."""
    dataset_dir = Path(dataset_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    for sensor in sensors:
        sensor_dataset_dir = dataset_dir / sensor
        os.makedirs(sensor_dataset_dir, exist_ok=True)
        if not sensor_dataset_dir.exists():
            logger.warning(f"Dataset directory for {sensor} not found, skipping")
            continue
        
        logger.info(f"Training model for {sensor}")
        
        # Create datamodule
        datamodule = Folder(
            name=sensor,
            root=str(sensor_dataset_dir),
            normal_dir='train/good',
            abnormal_dir='test/anomaly',
            normal_test_dir='test/good',
            train_batch_size=1,
            eval_batch_size=1,
            num_workers=16,
        )
        datamodule.train_transforms = train_transform
        datamodule.test_transforms = eval_transform
        
        # Create model
        # model = Patchcore(backbone=backbone, coreset_sampling_ratio=0.01)
        # model = Padim(backbone=backbone)
        model = EfficientAd()
        # model = Fastflow(backbone=backbone)
        
        # Create metrics collector callback
        metrics_collector = MetricsCollector()
        
        # Create engine with validation
        engine = Engine(
            max_epochs=epochs,
            default_root_dir=str(results_dir / sensor),
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            strategy='auto',
            devices=1,
            callbacks=[metrics_collector],
            accumulate_grad_batches=4,
            check_val_every_n_epoch=1,  # Enable validation every epoch
            val_check_interval=0.5,     # Validate twice per epoch
        )
        
        # Train and test
        try:
            logger.info(f"Starting training for {sensor}")
            engine.fit(datamodule=datamodule, model=model)
            
            logger.info(f"Testing model for {sensor}")
            test_results = engine.test(datamodule=datamodule, model=model)
            logger.info(f"Test results for {sensor}: {test_results}")
            
            # Visualize results with training progress
            visualize_results(sensor, metrics_collector, results_dir)
        except Exception as e:
            logger.error(f"Error training model for {sensor}: {e}")
        finally:
            # Clean up
            torch.cuda.empty_cache()
            del datamodule, model, engine
            torch.cuda.empty_cache()
            gc.collect()

def visualize_results(sensor, metrics_collector, output_dir):
    """Visualize training progress and anomaly detection results."""
    output_dir = Path(output_dir) / sensor
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot training and validation losses
    if metrics_collector.train_losses:
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(metrics_collector.train_losses) + 1)
        plt.plot(epochs, metrics_collector.train_losses, 'b-', label='Training Loss', linewidth=2)
        if metrics_collector.val_losses:
            val_epochs = np.linspace(1, len(epochs), len(metrics_collector.val_losses))
            plt.plot(val_epochs, metrics_collector.val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.title(f'Training Progress for {sensor.capitalize()} Sensor', fontsize=14, pad=20)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot anomaly scores
    if not metrics_collector.predictions:
        logger.error(f"No predictions available for {sensor}.")
        return

    try:
        scores = np.array(metrics_collector.predictions)
        if scores.size == 0:
            logger.error(f"Empty predictions array for {sensor}.")
            return
            
        timestamps = np.arange(len(scores))
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
        
        # Plot anomaly scores
        ax1.plot(timestamps, scores, 'b-', label='Anomaly Score', linewidth=2)
        mid_point = len(scores) // 2
        ax1.axvline(x=mid_point, color='r', linestyle='--', label='Normal/Anomaly Split', linewidth=2)
        
        # Add threshold line if we have enough data
        if len(scores) > 1:
            normal_scores = scores[:mid_point] if mid_point > 0 else np.array([])
            if normal_scores.size > 0:
                threshold = np.mean(normal_scores) + 2 * np.std(normal_scores)
                ax1.axhline(y=threshold, color='g', linestyle=':', label='Detection Threshold', linewidth=2)
        
        ax1.set_title(f'Anomaly Detection Results for {sensor.capitalize()} Sensor', fontsize=14, pad=20)
        ax1.set_xlabel('Sample Index', fontsize=12)
        ax1.set_ylabel('Anomaly Score', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Calculate and plot score distribution
        if len(scores) > 1:
            normal_scores = scores[:mid_point] if mid_point > 0 else np.array([])
            anomaly_scores = scores[mid_point:] if mid_point < len(scores) else np.array([])
            
            if normal_scores.size > 0 and anomaly_scores.size > 0:
                # Plot score distributions
                bins = np.linspace(min(scores), max(scores), 50)
                ax2.hist(normal_scores, bins, alpha=0.5, label='Normal', color='blue', density=True)
                ax2.hist(anomaly_scores, bins, alpha=0.5, label='Anomaly', color='red', density=True)
                ax2.set_title('Score Distribution', fontsize=14, pad=20)
                ax2.set_xlabel('Anomaly Score', fontsize=12)
                ax2.set_ylabel('Density', fontsize=12)
                ax2.legend(fontsize=10)
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'anomaly_detection_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate and log statistics
        if len(scores) > 1:
            normal_scores = scores[:mid_point] if mid_point > 0 else np.array([])
            anomaly_scores = scores[mid_point:] if mid_point < len(scores) else np.array([])
            
            if normal_scores.size > 0:
                logger.info(f"{sensor}: Mean normal score: {np.mean(normal_scores):.4f} ± {np.std(normal_scores):.4f}")
            if anomaly_scores.size > 0:
                logger.info(f"{sensor}: Mean anomaly score: {np.mean(anomaly_scores):.4f} ± {np.std(anomaly_scores):.4f}")
            
            # Calculate and log performance metrics
            if normal_scores.size > 0 and anomaly_scores.size > 0:
                threshold = np.mean(normal_scores) + 2 * np.std(normal_scores)
                true_positives = np.sum(anomaly_scores > threshold)
                false_positives = np.sum(normal_scores > threshold)
                true_negatives = np.sum(normal_scores <= threshold)
                false_negatives = np.sum(anomaly_scores <= threshold)
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # Save metrics to a text file
                with open(output_dir / 'performance_metrics.txt', 'w') as f:
                    f.write(f"{sensor} Performance Metrics:\n")
                    f.write(f"Precision: {precision:.4f}\n")
                    f.write(f"Recall: {recall:.4f}\n")
                    f.write(f"F1-Score: {f1_score:.4f}\n")
                    f.write(f"\nThreshold: {threshold:.4f}\n")
                    f.write(f"True Positives: {true_positives}\n")
                    f.write(f"False Positives: {false_positives}\n")
                    f.write(f"True Negatives: {true_negatives}\n")
                    f.write(f"False Negatives: {false_negatives}\n")
                
                logger.info(f"{sensor} Performance Metrics:")
                logger.info(f"  Precision: {precision:.4f}")
                logger.info(f"  Recall: {recall:.4f}")
                logger.info(f"  F1-Score: {f1_score:.4f}")
    except Exception as e:
        logger.error(f"Error visualizing results for {sensor}: {str(e)}")
        plt.close('all')  # Clean up any open figures

def main():
    parser = argparse.ArgumentParser(description='Train anomaly detection models for CARLA sensor data')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing sensor data')
    parser.add_argument('--dataset_dir', type=str, default='datasets', help='Directory to create dataset structure')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--sensors', type=str, nargs='+', 
                        default=['rgb', 'rgb_front', 'dvs_front', 'depth_front', 'lidar'],
                        help='Sensors to process')
    parser.add_argument('--attack_frame', type=int, default=300,
                        help='Frame number threshold to separate normal/attack frames')
    parser.add_argument('--backbone', type=str, default='resnet18', help='Backbone for Patchcore model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--normal_only', action='store_true',
                        help='Use when only normal data is available')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio of data to use for training (from non-attack frames)')
    args = parser.parse_args()
    
    # Process LiDAR data if needed
    if 'lidar' in args.sensors:
        process_lidar_data(args.data_dir)
    
    # Create dataset structure
    create_dataset_structure(
        data_dir=args.data_dir,
        output_dir=args.dataset_dir,
        sensors=args.sensors,
        train_ratio=args.train_ratio,
        attack_frame_threshold=args.attack_frame,
        normal_only=args.normal_only
    )
    
    # If normal_only, create synthetic anomalies
    if args.normal_only:
        for sensor in args.sensors:
            sensor_dataset_dir = Path(args.dataset_dir) / sensor
            test_anomaly_dir = sensor_dataset_dir / 'test' / 'anomaly'
            
            # Process each test file to create synthetic anomalies
            for test_file in test_anomaly_dir.glob('*.png'):
                synthetic_img = create_synthetic_anomalies(test_file)
                if synthetic_img is not None:
                    cv2.imwrite(str(test_file), synthetic_img)
    
    # Train models
    train_anomaly_models(
        args.dataset_dir,
        args.results_dir,
        args.sensors,
        backbone=args.backbone,
        epochs=args.epochs,
        normal_only=args.normal_only
    )
    
if __name__ == '__main__':
    main()