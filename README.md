# Diploma Project – Recognising Driver's Activities from the Images Captured by an In-Car Camera

This project detects and classifies specific driver activities based on video input from an in-car camera using a skeleton-based approach with MediaPipe and LSTM-based neural networks.

---

## Recognised Activities

The system identifies the following actions:

- Fastening the seatbelt
- Unfastening the seatbelt
- Shifting gears
- Using a mobile phone
- No action (neutral class)

---

## System Architecture

| Component          | Description                                      |
|-------------------|--------------------------------------------------|
| Pose Detection     | Google MediaPipe Pose                            |
| Feature Types      | Distances, angles, triangle areas (Types A–E)    |
| Sequence Models    | LSTM variants: single-directional, bidirectional, convolutional, attention |
| Input              | Keypoints from body skeleton over time           |
| Output             | Predicted activity class for each input sequence |

---

## Installation & Setup

Python version required: 3.11

Using a virtual environment is strongly recommended.

```bash
# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate          # On Linux/macOS
venv\Scripts\activate           # On Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Test Video

The system automatically downloads the required demo video when launching a demo script.

For manual access:  
Google Drive – Test Video (600 MB): https://drive.google.com/uc?id=1ONaGO5p1LoBm60Le9BAwmzfO8EICaD4p

The video will be downloaded to ./test_data/test_video.mp4 if not already present.

---

## Running the Demos

Navigate to the demo folder and execute any of the scripts below. Each script uses a different model architecture and feature type.

```bash
cd SEATBELT_DEMO

python demo1.py   # Simple LSTM (features A, sequence=4)
python demo2.py   # Bidirectional LSTM (features E, sequence=6)
python demo3.py   # Bi-LSTM + Conv (features E, sequence=5)
python demo4.py   # Bi-LSTM + Conv + Attention (features E, sequence=5)
```

Each demo performs the following:

1. Downloads the test video (if missing)
2. Detects body skeleton with MediaPipe
3. Computes features
4. Predicts the current driver activity
5. Displays prediction overlayed on video

---

## Project Structure (partial)

```
.
├── requirements.txt
├── README.md
├── SEATBELT_DEMO/
│   ├── demo1.py
│   ├── demo2.py
│   ├── demo3.py
│   └── demo4.py
├── SEATBELT_TRAIN_TEST/
│   ├── model_basic.py
│   ├── model_bidirectional.py
│   ├── model_convolution.py
│   └── model_attention.py
└── SEATBELT_PREPROCESS/
    └── calculate_features.py
```

---

## Additional Notes

- Each demo runs on CPU (no GPU required)
- Models must be available in ../SEATBELT_TRAIN_TEST/models/
- The prediction threshold is set to 95% confidence
- Feature extraction and model structure align with the thesis documentation

---

## Author

Daniel Dobeš  
Master’s Thesis, VŠB-TUO (2025)  
Department of Computer Science  
Supervisor: doc. Dr. Ing. Eduard Sojka
