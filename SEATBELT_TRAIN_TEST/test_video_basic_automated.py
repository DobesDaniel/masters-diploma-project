"""
    Testing models on videos
        - Single sequence of frames for model that has behaviour similar to queue (FIFO)
            - oldest frame from sequence when is at max length is being dropped
        - detection is being run after every new frame that is completing sequence to its max len

    Set MODEL_ARCHITECTURE_IDX = 0,1,2,3
        (row) 42
    Set count of frames that is used for detection = sequence length
        lstm_block_size = 3,4,5,6
        (row) 44
    Set features_type = A,B,C,D,E
        (row) 45

    Set video_path - video that is being run
        (row) 62
    Set action_scope_window - 60/120
        (row) 65

    Set output_csv_dir - "./predictions/"
        (row) 68

"""

import os
import cv2
import torch
import numpy as np
import mediapipe as mp
import csv
from pathlib import Path

from collate_fn import collate_fn
import logging
from hyperparameters import *
from SEATBELT_PREPROCESS.calculate_features import calculate_features

# force using cpu
NO_CUDA = True

MODEL_ARCHITECTURE = ["basic","bidirectional","convolution","attention"]
MODEL_ARCHITECTURE_IDX = 3

lstm_block_size = 5
features_type = "E"

# count of frames during action can happen - 2 seconds -> 2xFPS = ...
# this could technically be automated but not really...
# there is an option to choose from default idea is
# use action_scope_window = 120 for 60 fps
# and action_scope_window =  60 for 30 fps
# but! it could be changed
# - it will basically lower the offset between frames that are being picked
# of course models are pretrained with 2 seconds per action

# pick video
# video_path = f"../resources/SB_test/GX010020.MP4"     # 60 fps
# video_path = f"../resources/SB_test/GX010023.MP4"     # 60 fps
# video_path = f"../resources/SB_test/GX010037.MP4"     # 60 fps
# video_path = f"../resources/SB_test/GX010047.MP4"     # 60 fps
# video_path = f"../resources/SB_test/GX010062.MP4"     # 60 fps
video_path = f"../resources/MY_SB_test/guvc-4.avi"     # 30 fps

# action_scope_window = 120         # 60 fps
action_scope_window = 60           # 30 fps

# directory where will all predictions be saved
output_csv_dir = "../resources/final_predictions/"


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

experiment_name = f"size_{lstm_block_size}_features{features_type}"
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


cv2.namedWindow("Video Test", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Video Test", 1325, 1014)
# cv2.moveWindow("Video Test", 600, 200)
# smaller size
cv2.resizeWindow("Video Test", 800, 500)
cv2.moveWindow("Video Test", 400, 150)

# sequence buffer
sequence = []
sequence_length = lstm_block_size   # 3, 4, 5, 6
offset = action_scope_window / sequence_length

# class names
class_labels = ['fasten', 'gear', 'no_action', 'phone', 'unfasten']
ACTION_THRESHOLD = 0.95  # (0,1)

# this is just for creating results into .csv
video_filename = os.path.basename(video_path)
video_name = os.path.splitext(video_filename)[0]
output_dir = output_csv_dir
os.makedirs(output_dir, exist_ok=True)
output_csv_path = os.path.join(
    output_dir,
    f"{model_architecture}_{experiment_name}_pred.csv"
)

def append_prediction_to_csv(csv_path, video_name, architecture, model_id, frame_number, action_class, confidence):
    file_exists = Path(csv_path).is_file()
    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["video", "architecture", "model_type", "frame", "class", "confidence"])
        writer.writerow([video_name, architecture, model_id, frame_number, action_class, f"{confidence:.4f}"])


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
                        print(f"Action prediction: {predicted_label} | Confidence: {confidence.item()}")
                    else:
                        predicted_label = "no_action"

                    # this part is for a way to test models by comparing predictions for each frame with GT
                    append_prediction_to_csv(
                        output_csv_path,
                        video_name,
                        model_architecture,
                        experiment_name,
                        int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                        predicted_label,
                        confidence.item()
                    )
                sequence.pop(0)

    offset_counter += 1

    cv2.putText(frame, f"Action: {predicted_label}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.imshow("Video Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
