#!/usr/bin/env python3

import os
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import csv
import logging
import gc
from tqdm import tqdm
from torchvision import transforms
from anomalib.data import Folder
from anomalib.models import EfficientAd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set memory optimization settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision('medium')

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

def load_model(model_path, sensor):
    """Load a trained anomaly detection model."""
    logger.info(f"Loading model for {sensor} from {model_path}")
    try:
        # Create model instance
        model_class = EfficientAd
        
        # Load checkpoint - corrected path structure
        checkpoint_path = Path(model_path) / sensor 
        if not checkpoint_path.exists():
            logger.error(f"Model directory not found at {checkpoint_path}")
            return None
            
        # Look for checkpoint files in the lightning directory
        lightning_path = checkpoint_path / 'latest' / 'weights' / 'lightning'
        if not lightning_path.exists():
            logger.error(f"Checkpoint directory not found at {lightning_path}")
            return None
            
        checkpoint_files = list(lightning_path.glob("*.ckpt"))
        if not checkpoint_files:
            logger.error(f"No checkpoint files found in {lightning_path}")
            return None
            
        # Use the latest checkpoint
        checkpoint_file = sorted(checkpoint_files, key=lambda x: x.stat().st_mtime)[-1]
        logger.info(f"Using checkpoint: {checkpoint_file}")
        
        # Load model from checkpoint - use the model class's load_from_checkpoint method
        model = model_class.load_from_checkpoint(str(checkpoint_file))
        model.eval()
        
        return model
    except Exception as e:
        logger.error(f"Error loading model for {sensor}: {e}")
        return None

def get_transform():
    """Get the transform for preprocessing sensor data."""
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def process_lidar_data(episode_dir):
    """Process LiDAR .npy files to generate 2D images for anomaly detection."""
    lidar_dir = Path(episode_dir) / 'lidar'
    lidar_images_dir = Path(episode_dir) / 'lidar_images'
    lidar_images_dir.mkdir(exist_ok=True)
    
    logger.info(f"Processing LiDAR data from {lidar_dir} to {lidar_images_dir}")
    lidar_files = list(lidar_dir.glob('*.npy'))
    logger.info(f"Found {len(lidar_files)} LiDAR files")
    
    for lidar_file in tqdm(lidar_files, desc="Processing LiDAR data"):
        img_path = lidar_images_dir / lidar_file.name.replace('.npy', '.png')
        if img_path.exists():
            continue
        
        try:
            points = np.load(lidar_file)
            img = visualize_lidar(points)
            cv2.imwrite(str(img_path), img[:, :, ::-1])
        except Exception as e:
            logger.error(f"Error processing {lidar_file}: {e}")
    
    return lidar_images_dir

def extract_attack_intervals(episode_dir):
    """Extract attack intervals from transform.csv file."""
    transform_path = Path(episode_dir) / 'transform.csv'
    if not transform_path.exists():
        logger.error(f"Transform data not found at {transform_path}")
        return {}, {}
    
    attack_intervals = {}
    defense_intervals = {}
    current_attack = 'none'
    current_attack_start = 0
    current_defense = False
    current_defense_start = 0
    
    try:
        with open(transform_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            
            for row in reader:
                tick = int(row[0])
                attack = row[17]  # Attack type column
                defense_enabled = bool(int(row[18]))  # Defense enabled column
                
                # Track attack intervals
                if attack != current_attack:
                    if current_attack != 'none':
                        if current_attack not in attack_intervals:
                            attack_intervals[current_attack] = []
                        attack_intervals[current_attack].append((current_attack_start, tick))
                    
                    current_attack = attack
                    current_attack_start = tick
                
                # Track defense intervals
                if defense_enabled != current_defense:
                    if current_defense:
                        if 'defense' not in defense_intervals:
                            defense_intervals['defense'] = []
                        defense_intervals['defense'].append((current_defense_start, tick))
                    
                    current_defense = defense_enabled
                    current_defense_start = tick
        
        # Add the last intervals if they extend to the end
        if current_attack != 'none':
            if current_attack not in attack_intervals:
                attack_intervals[current_attack] = []
            attack_intervals[current_attack].append((current_attack_start, tick))
        
        if current_defense:
            if 'defense' not in defense_intervals:
                defense_intervals['defense'] = []
            defense_intervals['defense'].append((current_defense_start, tick))
        
        logger.info(f"Extracted attack intervals: {attack_intervals}")
        logger.info(f"Extracted defense intervals: {defense_intervals}")
        
        return attack_intervals, defense_intervals
    
    except Exception as e:
        logger.error(f"Error extracting attack intervals: {e}")
        return {}, {}

def analyze_sensor(episode_dir, results_dir, sensor, model):
    """Analyze sensor data and compute anomaly scores."""
    logger.info(f"Analyzing {sensor} data from {episode_dir}")
    
    # Determine source directory name (special case for lidar)
    source_dir_name = 'lidar_images' if sensor == 'lidar' else sensor
    source_dir = Path(episode_dir) / source_dir_name
    
    if not source_dir.exists():
        logger.warning(f"Source directory {source_dir} not found, skipping {sensor}")
        return None
    
    # Get all image files
    image_files = sorted(list(source_dir.glob('*.png')))
    if not image_files:
        logger.warning(f"No images found in {source_dir}")
        return None
    
    logger.info(f"Found {len(image_files)} images for {sensor}")
    
    # Extract frame numbers from filenames
    frame_numbers = []
    for img_file in image_files:
        frame_num = int(img_file.stem.split('_')[-1])
        frame_numbers.append(frame_num)
    
    # Prepare transform
    transform = get_transform()
    
    # Compute anomaly scores
    anomaly_scores = []
    
    for img_file in tqdm(image_files, desc=f"Computing anomaly scores for {sensor}"):
        try:
            # Load and preprocess image
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            
            # Convert to PIL Image for torchvision transforms
            from PIL import Image
            pil_img = Image.fromarray(img)
            
            # Apply transform
            tensor_img = transform(pil_img).unsqueeze(0)  # Add batch dimension
            
            # Move to device
            if torch.cuda.is_available():
                tensor_img = tensor_img.cuda()
            
            # Compute anomaly score
            with torch.no_grad():
                predictions = model(tensor_img)
                score = predictions.pred_score.cpu().numpy()[0]
                # score = 1 - score
            
            anomaly_scores.append(score)
            
        except Exception as e:
            logger.error(f"Error processing {img_file}: {e}")
            anomaly_scores.append(np.nan)  # Use NaN for failed predictions
    
    # Clean up
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        'frame_numbers': frame_numbers,
        'anomaly_scores': anomaly_scores
    }

def visualize_results(episode_dir, sensor, analysis_results, attack_intervals, defense_intervals, model):
    """Visualize anomaly detection results with attack intervals highlighted."""
    if analysis_results is None or 'anomaly_scores' not in analysis_results:
        logger.error(f"No analysis results available for {sensor}")
        return
    
    frame_numbers = analysis_results['frame_numbers']
    anomaly_scores = analysis_results['anomaly_scores']
    
    # Create output directory
    output_dir = Path(episode_dir) / f"{sensor}_{model}_vflip"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot anomaly scores with attack intervals
    plt.figure(figsize=(15, 5))
    
    # Plot anomaly scores
    plt.plot(frame_numbers, anomaly_scores, 'b-', label='Anomaly Score', linewidth=1.5)
    
    # Highlight attack intervals - only for the current sensor's attack type
    sensor_attack_type = {
        'rgb': 'rgb_noise',
        'rgb_front': 'rgb_noise',
        'dvs_front': 'dvs_noise',
        'depth_front': 'depth_tampering',
        'lidar': 'lidar_tampering'
    }.get(sensor)
    
    if sensor_attack_type and sensor_attack_type in attack_intervals:
        for i, (start, end) in enumerate(attack_intervals[sensor_attack_type]):
            label = f'Attack: {sensor_attack_type}' if i == 0 else None
            plt.axvspan(start, end, alpha=0.2, color='r', label=label)
            # Add text annotation for attack type
            mid_point = (start + end) / 2
            plt.text(mid_point, max(anomaly_scores) * 0.9, sensor_attack_type,
                     horizontalalignment='center', verticalalignment='center',
                     bbox=dict(facecolor='white', alpha=0.7))
    
    # Highlight defense intervals
    if 'defense' in defense_intervals:
        for i, (start, end) in enumerate(defense_intervals['defense']):
            label = 'Defense Enabled' if i == 0 else None
            plt.axvspan(start, end, alpha=0.1, color='g', hatch='/', label=label)
    
    # Add labels and title
    plt.title(f'Anomaly Detection Results for {sensor.capitalize()} Sensor', fontsize=14, pad=20)
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('Anomaly Score', fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_dir / f'{sensor}_anomaly_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate statistics for different scenarios
    stats = {}
    
    # Normal operation (no attack)
    normal_scores = []
    for i, frame in enumerate(frame_numbers):
        is_in_attack = False
        for attack_type, intervals in attack_intervals.items():
            if attack_type == 'none':
                continue
            for start, end in intervals:
                if start <= frame <= end:
                    is_in_attack = True
                    break
            if is_in_attack:
                break
        
        if not is_in_attack:
            normal_scores.append(anomaly_scores[i])
    
    if normal_scores:
        stats['normal'] = {
            'mean': np.nanmean(normal_scores),
            'std': np.nanstd(normal_scores),
            'min': np.nanmin(normal_scores),
            'max': np.nanmax(normal_scores),
            'count': len(normal_scores)
        }
    
    # Attack with defense
    for attack_type, intervals in attack_intervals.items():
        if attack_type == 'none':
            continue
            
        attack_with_defense_scores = []
        attack_without_defense_scores = []
        
        for start, end in intervals:
            for i, frame in enumerate(frame_numbers):
                if start <= frame <= end:
                    # Check if defense was enabled
                    has_defense = False
                    for def_start, def_end in defense_intervals.get('defense', []):
                        if def_start <= frame <= def_end:
                            has_defense = True
                            break
                    
                    if has_defense:
                        attack_with_defense_scores.append(anomaly_scores[i])
                    else:
                        attack_without_defense_scores.append(anomaly_scores[i])
        
        if attack_with_defense_scores:
            stats[f'{attack_type}_with_defense'] = {
                'mean': np.nanmean(attack_with_defense_scores),
                'std': np.nanstd(attack_with_defense_scores),
                'min': np.nanmin(attack_with_defense_scores),
                'max': np.nanmax(attack_with_defense_scores),
                'count': len(attack_with_defense_scores)
            }
        
        if attack_without_defense_scores:
            stats[f'{attack_type}_without_defense'] = {
                'mean': np.nanmean(attack_without_defense_scores),
                'std': np.nanstd(attack_without_defense_scores),
                'min': np.nanmin(attack_without_defense_scores),
                'max': np.nanmax(attack_without_defense_scores),
                'count': len(attack_without_defense_scores)
            }
    
    # Save statistics to file
    with open(output_dir / f'{sensor}_statistics.txt', 'w') as f:
        f.write(f"{sensor.capitalize()} Anomaly Detection Statistics\n")
        f.write("=" * 40 + "\n\n")
        
        for scenario, metrics in stats.items():
            f.write(f"{scenario.replace('_', ' ').title()}:\n")
            f.write(f"  Mean: {metrics['mean']:.6f}\n")
            f.write(f"  Std: {metrics['std']:.6f}\n")
            f.write(f"  Min: {metrics['min']:.6f}\n")
            f.write(f"  Max: {metrics['max']:.6f}\n")
            f.write(f"  Count: {metrics['count']}\n\n")
    
    # Create distribution plot
    plt.figure(figsize=(12, 8))
    
    # Plot distributions only for attack scenarios
    for scenario, metrics in stats.items():
        if 'normal' in scenario:
            continue
            
        # Determine color and label based on defense status
        if 'with_defense' in scenario:
            color = 'green'
            attack_type = scenario.split('_with')[0]
            label = f'{attack_type.title()} with Defense'
        elif 'without_defense' in scenario:
            color = 'red'
            attack_type = scenario.split('_without')[0]
            label = f'{attack_type.title()} without Defense'
        else:
            continue
            
        # Extract scores for this attack scenario
        scenario_scores = []
        for start, end in attack_intervals.get(attack_type, []):
            for i, frame in enumerate(frame_numbers):
                if start <= frame <= end:
                    # Check defense status
                    has_defense = any(def_start <= frame <= def_end 
                                    for def_start, def_end in defense_intervals.get('defense', []))
                    if ('with_defense' in scenario and has_defense) or \
                       ('without_defense' in scenario and not has_defense):
                        scenario_scores.append(anomaly_scores[i])
        
        if scenario_scores:            # Convert to numpy array and ensure 1D array
            scenario_scores = np.array(scenario_scores).reshape(-1)
            plt.hist(scenario_scores, bins=30, alpha=0.5, color=color, label=label, density=True)
    
    plt.title(f'Anomaly Score Distribution for {sensor.capitalize()} Sensor', fontsize=14, pad=20)
    plt.xlabel('Anomaly Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save distribution plot
    plt.savefig(output_dir / f'{sensor}_score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats

def create_summary_plot(episode_dir, results_dir, all_stats, sensors):
    """Create a summary plot comparing performance across sensors."""
    output_dir = Path(results_dir) / Path(episode_dir).name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect scenarios across all sensors
    all_scenarios = set()
    for sensor_stats in all_stats.values():
        all_scenarios.update(sensor_stats.keys())
    
    # Filter out empty sensors
    valid_sensors = [sensor for sensor in sensors if sensor in all_stats and all_stats[sensor]]
    
    if not valid_sensors:
        logger.error("No valid sensor data for summary plot")
        return
    
    # Create bar plot for mean anomaly scores
    plt.figure(figsize=(15, 10))
    
    # Define scenario groups and colors
    scenario_groups = {
        'normal': {'color': 'blue', 'hatch': ''},
        'with_defense': {'color': 'green', 'hatch': '/'},
        'without_defense': {'color': 'red', 'hatch': 'x'}
    }
    
    # Set up bar positions
    bar_width = 0.2
    index = np.arange(len(valid_sensors))
    
    # Plot bars for each scenario group
    legend_handles = []
    
    for i, group_key in enumerate(scenario_groups.keys()):
        group_means = []
        group_stds = []
        group_labels = []
        
        for sensor in valid_sensors:
            sensor_stats = all_stats[sensor]
            
            # Find matching scenarios for this group
            matching_scenarios = [s for s in sensor_stats.keys() if group_key in s]
            
            if matching_scenarios:
                # If multiple scenarios match (e.g., different attack types), average them
                means = [sensor_stats[s]['mean'] for s in matching_scenarios]
                stds = [sensor_stats[s]['std'] for s in matching_scenarios]
                group_means.append(np.mean(means))
                group_stds.append(np.mean(stds))
                group_labels.append(', '.join([s.split('_')[0] for s in matching_scenarios]))
            else:
                group_means.append(0)
                group_stds.append(0)
                group_labels.append('')
        
        # Plot bars
        bars = plt.bar(index + i*bar_width - bar_width, group_means, bar_width,
                      yerr=group_stds, capsize=5, 
                      color=scenario_groups[group_key]['color'],
                      hatch=scenario_groups[group_key]['hatch'],
                      label=group_key.replace('_', ' ').title())
        
        legend_handles.append(bars)
    
    # Add labels and title
    plt.xlabel('Sensor', fontsize=14)
    plt.ylabel('Mean Anomaly Score', fontsize=14)
    plt.title('Anomaly Detection Performance Across Sensors', fontsize=16, pad=20)
    plt.xticks(index, [s.capitalize() for s in valid_sensors], fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save summary plot
    plt.savefig(output_dir / 'sensor_comparison_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze sensor anomalies using trained models')
    parser.add_argument('--episode_dir', type=str, required=True, help='Directory containing episode data')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing trained models and for saving results')
    parser.add_argument('--model', type=str, default='EfficientAd', help='Model type to use for analysis')
    parser.add_argument('--sensors', type=str, nargs='+', 
                        default=['rgb_front', 'dvs_front', 'depth_front', 'lidar'],
                        help='Sensors to analyze')
    args = parser.parse_args()
    
    # Process LiDAR data if needed
    if 'lidar' in args.sensors:
        process_lidar_data(args.episode_dir)
    
    # Extract attack and defense intervals
    attack_intervals, defense_intervals = extract_attack_intervals(args.episode_dir)
    
    # Analyze each sensor
    all_stats = {}
    
    for sensor in args.sensors:
        # Load model
        result_dir = Path(args.results_dir) / sensor / args.model
        model_result = load_model(result_dir, sensor)
        if model_result is None:
            logger.error(f"Failed to load model for {sensor}, skipping")
            continue
            
        else:
            model = model_result
            
            # Analyze sensor data
            analysis_results = analyze_sensor(args.episode_dir, args.results_dir, sensor, model)
            
            # Visualize results
            if analysis_results:
                stats = visualize_results(args.episode_dir, sensor, 
                                        analysis_results, attack_intervals, defense_intervals, args.model)
                all_stats[sensor] = stats
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    # Create summary plot
    create_summary_plot(args.episode_dir, args.results_dir, all_stats, args.sensors)
    
    logger.info("Analysis complete")

if __name__ == '__main__':
    main()