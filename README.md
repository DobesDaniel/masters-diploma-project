# Diploma Project - Recognising Driver's Activities from the Images Captured by an In-Car Camera
This project focuses on detecting and classifying specific driver activities using a skeleton-based analysis of in-car camera footage. The solution uses MediaPipe for body keypoint detection and an LSTM-based neural network for sequence classification.


## Project Overview
The system identifies the following driver activities:
- Fastening the seatbelt
- Unfastening the seatbelt
- Shifting gears
- Using a mobile phone

The workflow consists of extracting skeletal landmarks, feature engineering (distances, angles, triangle areas), and classifying temporal sequences with an LSTM network.

## Architecture
- **Pose Detection**: MediaPipe
- **Feature Extraction**: Custom geometric features (normalized distances, angles, triangle areas)
- **Sequence Modeling**: LSTM, Bi-LSTM, and LSTM with Attention
- **Classification Output**: Driver activity class per video sequence

## Installation
⚠️ **Note:** This project requires **Python 3.11**.  
<!-- 
Other versions (e.g., Python 3.12 or newer) might cause compatibility issues with the MediaPipe library.
 -->

Install the required dependencies with:

```bash
pip install -r requirements.txt


