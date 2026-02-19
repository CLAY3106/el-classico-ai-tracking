# ⚽ El Clásico AI Football Tracking & Tactical Analysis

## Overview

This project implements a full computer vision pipeline for automated football match analysis.

Given a broadcast match video, the system:

- Detects players using YOLOv8  
- Tracks identities across frames using ByteTrack  
- Automatically separates teams using color-based clustering  
- Produces structured tracking data  
- Computes team-level tactical metrics (centroids)

The project demons0trates end-to-end system design, feature engineering, unsupervised learning, and analytics architecture.

---

## System Pipeline

Match Video
↓
YOLOv8 Detection
↓
ByteTrack Multi-Object Tracking
↓
Team Classification (2D HSV Clustering)
↓
Structured Tracking Dataset
↓
Team Tactical Metrics


---

## 1️⃣ Player Detection & Tracking

**Model:** YOLOv8 (nano version for efficiency)  
**Tracker:** ByteTrack  

Each detection produces:

- `frame`
- `track_id`
- `team_id`
- `confidence`
- `bounding box (x1, y1, x2, y2)`
- `center point (cx, cy)`

Center coordinates are computed as:

cx = (x1 + x2) / 2
cy = (y1 + y2) / 2


These serve as the positional representation for tactical analysis.

---

## 2️⃣ Automatic Team Classification

### Problem

Hue-only clustering failed for white jerseys because:

- White has very low saturation
- Hue becomes unstable
- Severe class imbalance occurred

### Solution

We upgraded from 1D Hue clustering to 2D color feature clustering using:

- Median Saturation (S)
- Median Value (V)

For each player track:

1. Extract torso region  
2. Convert BGR → HSV  
3. Filter unreliable pixels  
4. Compute median S and V  
5. Accumulate samples per track  
6. Perform 2D K-Means clustering  

### Clustering Details

- Distance metric: squared Euclidean distance  
- Initialization: saturation extremes  
- Stable labeling rule:
  - Team 0 = lower saturation center  
  - Team 1 = higher saturation center  

### Result

Balanced team separation:

- Team 0 detections: 17,765  
- Team 1 detections: 10,318  

This significantly improved performance over 1D Hue clustering.

---

## 3️⃣ Tactical Analytics Layer

### Team Centroid

For each frame:

centroid_x = mean(cx of team players)
centroid_y = mean(cy of team players)


Output format: frame, team_id, centroid_x, centroid_y, n_players


This enables:

- Team block movement tracking  
- Defensive line height estimation  
- Structural compactness analysis  

---

## Project Architecture
```
comp-vision-football/
│
├── track.py
├── tracks.csv
├── team_centroids.csv
│
└── src/
├── classification/
│ └── team_classifier.py
│
└── analytics/
└── compute_centroids.py
```

Design principles:

- Separation of tracking and analytics layers  
- Modular classifier implementation  
- Reusable analytics modules  
- Clean data pipeline structure  

---

## Engineering Highlights

- Manual 2D K-Means implementation (no external clustering libraries)
- Robust median-based feature aggregation
- Stable label ordering to prevent team ID flipping
- Efficient streaming inference to avoid memory overflow
- HSV feature engineering tailored to football broadcast conditions

---

## Current Capabilities

The system can:

- Track multiple players consistently
- Automatically separate two teams without supervision
- Produce structured per-frame positional data
- Compute team-level movement metrics

---

## Limitations

- Pixel coordinate space (no pitch homography yet)
- No ball detection
- No referee filtering
- No speed or distance estimation
- No formation inference

---

## Future Extensions

Planned improvements:

- Homography mapping to pitch coordinates  
- Player speed estimation  
- Team compactness metrics  
- Heatmap visualization  
- Ball tracking  
- Pass network reconstruction  
- Tactical phase detection  

---

## Why This Project Matters

This system demonstrates:

- End-to-end ML system design  
- Feature engineering under real-world constraints  
- Unsupervised learning for team classification  
- Applied computer vision in sports analytics  

It serves as a foundation for advanced football tactical modeling.
