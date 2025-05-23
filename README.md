# Cyberattack Resilience of Autonomous Vehicle Sensor Systems: Evaluating RGB vs. Dynamic Vision Sensors in CARLA

## Authors

Mustafa Sakhai, Kaung Sithu, Min Khant Soe Oke, and Maciej Wielgosz

*AGH University of Science and Technology, Krakow, Poland*

## Abstract

Autonomous Vehicles (AVs) rely on a heterogeneous sensor suite RGB cameras, LiDAR, GPS/IMU, and emerging event‐based Dynamic Vision Sensors (DVS) to perceive and navigate complex environments. However, these sensors can be deceived by realistic cyberattacks, undermining safety. In this work, we systematically implement seven attack vectors in the CARLA simulator, salt and pepper noise, event flooding, depth map tampering, LiDAR phantom injection, GPS spoofing, denial of service, and steering bias control, and measure their impact on a state‐of‐the‐art end to end driving agent. We then equip each sensor with tailored defenses (e.g., adaptive median filtering for RGB, spatial clustering for DVS) and integrate a semi-supervised anomaly detector (EfficientAD from anomalib) trained exclusively on benign data. Our detector achieves clear separation between normal and attacked conditions (mean RGB anomaly scores of 0.00 vs. 0.38; DVS: 0.61 vs. 0.76), yielding over 95% detection accuracy with fewer than 5% false positives. Defense evaluations reveal that GPS spoofing is fully mitigated, whereas RGB and depth based attacks still induce 30–45% trajectory drift despite filtering. Notably, DVS sensors exhibit greater intrinsic resilience in high dynamic range scenarios, though their asynchronous output necessitates carefully tuned thresholds. These findings underscore the critical role of multi-modal anomaly detection and demonstrate that possibility of integrating DVS alongside conventional sensors significantly strengthens AV cybersecurity.

**Keywords**: autonomous vehicles, cybersecurity attacks, dynamic vision sensor

---

# CARLA Sensor Anomaly Detection Framework

This repository contains a framework for simulating, detecting, and analyzing sensor anomalies and attacks in autonomous driving systems using the CARLA simulator. The framework includes tools for generating attack scenarios, collecting sensor data, training anomaly detection models, and analyzing the results.

## Overview

The framework consists of the following main components:

1. **Attack Simulation**: Simulates various attacks on autonomous vehicle sensors in CARLA
2. **Data Collection**: Collects sensor data during normal and attack scenarios
3. **Anomaly Detection**: Trains models to detect anomalies in sensor data
4. **Analysis**: Analyzes the effectiveness of anomaly detection models

## Files

### 1. `neat_simplified_attacks_v10_new.py`

This script simulates various attacks on autonomous vehicle sensors in the CARLA simulator.

**Features:**
- Implements multiple attack types (GPS spoofing, RGB noise, depth tampering, DVS noise, LiDAR tampering, DoS, steering bias)
- Configurable attack scenarios based on time intervals
- Collects and saves sensor data during simulation
- Supports various sensors (RGB cameras, depth cameras, DVS cameras, LiDAR, GPS, IMU)

**Usage:**
```bash
python neat_simplified_attacks_v10_new.py [options]
```

### 2. `train_sensor_anomaly_models_v2_recursive.py`

This script trains anomaly detection models for different sensor types using collected data.

**Features:**
- Supports multiple anomaly detection models (EfficientAd, Patchcore, Padim, Fastflow)
- Processes various sensor data types (RGB, depth, DVS, LiDAR)
- Creates dataset structures for training and testing
- Includes custom metrics collection and visualization

**Usage:**
```bash
python train_sensor_anomaly_models_v2_recursive.py --data_dir [path] --output_dir [path] --sensors [sensor_list]
```

### 3. `analyze_sensor_anomalies_new.py`

This script analyzes sensor data using trained anomaly detection models to identify anomalies.

**Features:**
- Loads trained models for each sensor type
- Processes and analyzes sensor data from simulation episodes
- Extracts attack intervals from simulation data
- Computes and visualizes anomaly scores
- Evaluates detection performance

**Usage:**
```bash
python analyze_sensor_anomalies_new.py --data_dir [path] --model_dir [path] --output_dir [path]
```

### 4. `config.json`

Configuration file that defines attack scenarios for the simulation.

**Example:**
```json
{
    "attacks": [
        {"start": 0, "end": 400, "type": "none"},
        {"start": 400, "end": 800, "type": "gps"},
        {"start": 800, "end": 1200, "type": "rgb_noise"},
        ...
    ]
}
```

## Requirements

- CARLA Simulator
- Python 3.7+
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Anomalib

## Installation

1. Install CARLA simulator following the [official documentation](https://carla.readthedocs.io/)
2. Install required Python packages:
   ```bash
   pip install torch torchvision opencv-python numpy matplotlib tqdm anomalib
   ```

## Workflow

1. **Configure Attack Scenarios**:
   - Edit `config.json` to define attack types and time intervals

2. **Run Simulation with Attacks**:
   ```bash
   python neat_simplified_attacks_v10_new.py
   ```

3. **Train Anomaly Detection Models**:
   ```bash
   python train_sensor_anomaly_models_v2_recursive.py --data_dir ./data --output_dir ./models
   ```

4. **Analyze Results**:
   ```bash
   python analyze_sensor_anomalies_new.py --data_dir ./data --model_dir ./models --output_dir ./results
   ```

## Attack Types

The framework supports the following attack types:

- **GPS Spoofing**: Manipulates GPS coordinates
- **RGB Noise**: Adds noise to RGB camera images
- **Depth Tampering**: Manipulates depth camera readings
- **DVS Noise**: Adds noise to Dynamic Vision Sensor data
- **LiDAR Tampering**: Manipulates LiDAR point cloud data
- **DoS (Denial of Service)**: Simulates sensor data unavailability
- **Steering Bias**: Introduces bias in steering control

## Anomaly Detection Models

The framework uses the following anomaly detection models from the Anomalib library:

- **EfficientAd**: Efficient anomaly detection model
- **Patchcore**: Patch-based anomaly detection
- **Padim**: PaDiM anomaly detection model
- **Fastflow**: Flow-based anomaly detection

## Publication

This work is being prepared for publication in an MDPI journal. The full paper titled "Cyberattack Resilience of Autonomous Vehicle Sensor Systems: Evaluating RGB vs. Dynamic Vision Sensors in CARLA" provides detailed methodology, results, and analysis of our experiments.

## License


## Acknowledgements

- [CARLA Simulator](https://carla.org/)
- [Anomalib](https://github.com/openvinotoolkit/anomalib)
- AGH University of Science and Technology, Krakow, Poland