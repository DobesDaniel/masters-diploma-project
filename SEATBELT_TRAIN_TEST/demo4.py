"""
    Demo 3
    - model for bidirectional LSTM with convolution layer and attention layer
    - sequence length = 5
    - features types = E

"""

import os
import cv2
import torch
import numpy as np
import mediapipe as mp
import gdown

from hyperparameters import *
from SEATBELT_PREPROCESS.calculate_features import calculate_features

# force using cpu
NO_CUDA = True

MODEL_ARCHITECTURE = ["basic","bidirectional","convolution","attention"]
MODEL_ARCHITECTURE_IDX = 3

lstm_block_size = 5
features_type = "E"
experiment_name = f"size_{lstm_block_size}_features{features_type}"

demo_name = f"Demo - {MODEL_ARCHITECTURE[MODEL_ARCHITECTURE_IDX]} - {experiment_name}"

video_path = '../test_data/test_video.mp4'

print(f"For this demo there is a required video that will be downloaded from google drive!")

video_download_url = "https://drive.google.com/uc?id=1ONaGO5p1LoBm60Le9BAwmzfO8EICaD4p"
output_dir = os.path.dirname(video_path)

if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(video_path):
    gdown.download(video_download_url, video_path,fuzzy=True, quiet=False)


action_scope_window = 60 # 30 fps

if MODEL_ARCHITECTURE[MODEL_ARCHITECTURE_IDX] == "basic":
    from model_basic import ActionRecognitionLSTM
elif MODEL_ARCHITECTURE[MODEL_ARCHITECTURE_IDX] == "bidirectional":
    from model_bidirectional import ActionRecognitionLSTM
elif MODEL_ARCHITECTURE[MODEL_ARCHITECTURE_IDX] == "convolution":
    from model_convolution import ActionRecognitionLSTM
elif MODEL_ARCHITECTURE[MODEL_ARCHITECTURE_IDX] == "attention":
    from model_attention import ActionRecognitionLSTM
else:
    raise Exception("Unknown model")

model_architecture = f"model_{MODEL_ARCHITECTURE[MODEL_ARCHITECTURE_IDX]}"

model_path = f"./models/my_{model_architecture}_{experiment_name}"


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if NO_CUDA: device = "cpu"
model = ActionRecognitionLSTM(input_size, hidden_size, num_classes, num_layers).to(device)
if NO_CUDA: model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
else: model.load_state_dict(torch.load(model_path))
model.eval()


cv2.namedWindow(f"Video {demo_name}", cv2.WINDOW_NORMAL)
# cv2.resizeWindow(f"Video {demo_name}", 1325, 1014)
# cv2.moveWindow(f"Video {demo_name}", 600, 200)
# smaller size
cv2.resizeWindow(f"Video {demo_name}", 800, 500)
cv2.moveWindow(f"Video {demo_name}", 400, 150)

# sequence buffer
sequence = []
sequence_length = lstm_block_size   # 3, 4, 5, 6
offset = action_scope_window / sequence_length

# class names
class_labels = ['fasten', 'gear', 'no_action', 'phone', 'unfasten']
ACTION_THRESHOLD = 0.95  # (0,1)

cap = cv2.VideoCapture(video_path)

predicted_label = ""
offset_counter = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if offset_counter >= offset:

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        if results.pose_landmarks:
            offset_counter = 0
            landmarks = results.pose_landmarks.landmark

            # target landmarks
            nose_x, nose_y = landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y
            left_shoulder_x, left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_shoulder_x, right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            left_elbow_x, left_elbow_y = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y
            right_elbow_x, right_elbow_y = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y
            left_wrist_x, left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
            right_wrist_x, right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y

            # calculate training features
            features = calculate_features(features_type ,[nose_x, nose_y,
                                              left_shoulder_x, left_shoulder_y,
                                              right_shoulder_x, right_shoulder_y,
                                              left_elbow_x, left_elbow_y,
                                              right_elbow_x, right_elbow_y,
                                              left_wrist_x, left_wrist_y,
                                              right_wrist_x, right_wrist_y])

            # add to sequence
            sequence.append(features)

            # if sequence is long enough test for action
            if len(sequence) >= sequence_length:
                # sequence to npy to tensor
                sequence_np = np.array(sequence)
                sequence_tensor = torch.tensor(sequence_np, dtype=torch.float32).unsqueeze(0).to(device)

                # action detection
                with torch.no_grad():
                    outputs = model(sequence_tensor)
                    softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(softmax_outputs, 1)

                    # thresholding
                    if confidence.item() >= ACTION_THRESHOLD:
                        predicted_label = class_labels[predicted.item()]
                        print(f"Action prediction: {predicted_label:<10} | Confidence: {confidence.item():.5f}")
                    else:
                        predicted_label = "no_action"

                sequence.pop(0)

    offset_counter += 1

    cv2.putText(frame, f"Action: {predicted_label}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 9)
    cv2.putText(frame, f"Action: {predicted_label}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.imshow(f"Video {demo_name}", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
