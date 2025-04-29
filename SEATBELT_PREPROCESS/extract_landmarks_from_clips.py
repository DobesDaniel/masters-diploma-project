"""

    Save LANDMARKS/KEYPOINTS for pose from clips of target actions

    LANDMARKS:
    - NOSE
    - LEFT_SHOULDER
    - RIGHT_SHOULDER
    - LEFT_ELBOW
    - RIGHT_ELBOW
    - LEFT_WRIST
    - RIGHT_WRIST

    creates txt (csv) file with same name as video
    each line contains 7 coords for each landmark

"""

import os
import glob
import cv2
import mediapipe as mp
import pandas as pd

def extract_landmarks(video_path, output_directory):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)

    landmarks_data = []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            frame_landmarks = {
                'frame': frame_idx,
                'nose_x': landmarks[mp_pose.PoseLandmark.NOSE].x,
                'nose_y': landmarks[mp_pose.PoseLandmark.NOSE].y,
                'left_shoulder_x': landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                'left_shoulder_y': landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                'right_shoulder_x': landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                'right_shoulder_y': landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                'left_elbow_x': landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                'left_elbow_y': landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y,
                'right_elbow_x': landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                'right_elbow_y': landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
                'left_wrist_x': landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                'left_wrist_y': landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y,
                'right_wrist_x': landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                'right_wrist_y': landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
            }
            landmarks_data.append(frame_landmarks)

        frame_idx += 1

    cap.release()

    output_filename = os.path.join(output_directory, video_path.split('/')[-1].split('.')[0] + '.csv')

    df = pd.DataFrame(landmarks_data)
    df.to_csv(output_filename, index=False)

    print(f"Landmarks saved to {output_filename}")


def extract_landmarks_from_clips(input_path, output_path):

    os.makedirs(output_path, exist_ok=True)
    video_files = glob.glob(input_path)
    for video_path in video_files:
        extract_landmarks(video_path, output_path)
        # break     # run only for first video - just testing
    pass

def main():
    # input_path = "../resources/SB_cuts/fasten/*.mp4"
    # output_path = "../resources/SB_cuts_landmarks/fasten/"
    # extract_landmarks_from_clips(input_path, output_path)
    #
    # input_path = "../resources/SB_cuts/unfasten/*.mp4"
    # output_path = "../resources/SB_cuts_landmarks/unfasten/"
    # extract_landmarks_from_clips(input_path, output_path)
    #
    # input_path = "../resources/SB_cuts/no_action/*.mp4"
    # output_path = "../resources/SB_cuts_landmarks/no_action/"
    # extract_landmarks_from_clips(input_path, output_path)

    # input_path = "../resources/SB_cuts/no_action/GX010057_2188_2308.mp4"
    # output_path = "../resources/SB_cuts_landmarks/no_action/"
    # extract_landmarks_from_clips(input_path, output_path)


    input_path = "../resources/MY_SB_cuts/no_action/*.avi"
    output_path = "../resources/MY_SB_cuts_landmarks/no_action/"
    extract_landmarks_from_clips(input_path, output_path)

    input_path = "../resources/MY_SB_cuts/fasten/*.avi"
    output_path = "../resources/MY_SB_cuts_landmarks/fasten/"
    extract_landmarks_from_clips(input_path, output_path)

    input_path = "../resources/MY_SB_cuts/unfasten/*.avi"
    output_path = "../resources/MY_SB_cuts_landmarks/unfasten/"
    extract_landmarks_from_clips(input_path, output_path)

    input_path = "../resources/MY_SB_cuts/phone/*.avi"
    output_path = "../resources/MY_SB_cuts_landmarks/phone/"
    extract_landmarks_from_clips(input_path, output_path)

    input_path = "../resources/MY_SB_cuts/gear/*.avi"
    output_path = "../resources/MY_SB_cuts_landmarks/gear/"
    extract_landmarks_from_clips(input_path, output_path)

    pass

if __name__ == "__main__":
    main()
    # extract_landmarks_from_clips()
    pass
