# Real-Time Behavioral Safety Monitor for Geriatric Care (Society 5.0)

## Research Overview

This project presents a modular Computer Vision framework designed to mitigate the risks associated with Kodokushi (lonely deaths) in Japan's aging population. By integrating State-of-the-Art (SOTA) object detection, biometric recognition, and pose estimation, the system provides a robust solution for automated fall detection and identity verification on edge devices.

## System Architecture

The system is engineered using a decoupled four-module pipeline to ensure scalability and independent optimization of neural network performance.

### Module A: Observer (Detection)

- Utilizes YOLOv8n for high-speed spatial localization of human subjects within the frame.

### Module B: Identifier (Recognition)

- Implements the ArcFace (InsightFace) algorithm for biometric verification.
- Computes Euclidean Distances between live embeddings and a registered database to ensure the safety of specific individuals.

### Module C: Analyzer (Behavioral Posture)

- Employs 17-point Human Pose Estimation (HPE).
- Calculates vertical displacement ratios between the nose, hips, and ankles to classify states: Standing, Sitting, or Fallen.

### Module D: Decision (Temporal Logic)

- A critical "Decision Gate" that validates a fall event over a 3-second temporal window to filter out momentary outliers and minimize False Acceptance Rates (FAR).

## Key Research Features

- **Edge-AI Optimized**: Leveraging ONNX Runtime for low-latency execution on CPU-bound environments.
- **Privacy-Preserving**: The system performs localized processing without the need for cloud-based data transmission.
- **Society 5.0 Alignment**: Directly addresses the Japanese government's vision for a human-centric society that balances economic advancement with the resolution of social problems.

## Installation & Usage

### Clone the Repository

```bash
git clone https://github.com/Muhammad044/behavior-monitoring-system.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Database Setup

Place a clear image of the subject in `identifier/database/subject_name.jpg`.

### Run the System

```bash
python main.py
```

## 📊 Performance & Future Work

Current testing indicates high reliability in indoor environments. Future iterations will explore:

- Spatial-Temporal Graph Convolutional Networks (ST-GCN) for more complex action recognition.
- Integration with IoT-based alerting protocols (MQTT) for emergency response.


## 📺 System Demonstration
[![Watch the Demo](https://img.youtube.com/vi/omMpetXWfTE/0.jpg)](https://www.youtube.com/watch?v=omMpetXWfTE)
*Click the image above to view the real-time processing demo.*
