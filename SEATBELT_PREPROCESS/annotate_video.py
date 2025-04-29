"""

    Program that plays videos from directory, and allows to annotate action within
    - play and pause, back and forward few frames
    - end of annotated action adding class of that action
    - start of the action is exactly 2 seconds before the end (120 frames subtracted from end)
    - output text file per video with each line contains class of action, frame where the action starts and ends

"""

import cv2
import os
import glob


# CONTROLS
ctrl_quit = 'q'
ctrl_play_pause = ' '
ctrl_skip_back = 'a'
ctrl_skip_forward = 'd'
ctrl_action_end = 'f'

FRAME_JUMP = 100  # number of frames to jump forward or backward
ACTION_DURATION_FRAMES = 120  # start of action is 2 seconds before end (assuming 60 FPS)

actions = []
start_frame = None
current_action = None

def annotate_video(video_path, output_directory):

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_directory, f"{video_name}.txt")

    # empty annotations
    global actions
    actions = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    cv2.namedWindow("Video Annotator", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Video Annotator", 1325, 1014)
    # cv2.moveWindow("Video Annotator", 600, 200)

    # smaller window screen
    cv2.resizeWindow("Video Annotator", 800, 500)  # Set default size to 800x600
    cv2.moveWindow("Video Annotator", 400, 150)  # Move window to (x=100, y=100)


    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_number = 0
    playing = True

    # print("Running: ", video_path)
    print(f"Fps: {fps}")

    action_end = None

    while cap.isOpened():
        if playing:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1

        # if frame_number % 2 == 1: continue

        # playing at normal speed using openCV (4x faster)
        # if frame_number % 2 == 1: continue
        # if frame_number % 4 == 1: continue
        # if frame_number % 4 == 2: continue
        # if frame_number % 4 == 3: continue

        cv2.putText(frame, f"Frame: {frame_number}/{total_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Video Annotator", frame)
        key = cv2.waitKey(30 if playing else 0) & 0xFF

        if key == ord(ctrl_quit):       # close
            break
        elif key == ord(ctrl_play_pause):  # pause
            cv2.waitKey(0)
        elif key == ord(ctrl_skip_forward):  # move forward
            frame_number = min(frame_number + FRAME_JUMP, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        elif key == ord(ctrl_skip_back):  # move backward
            frame_number = max(frame_number - FRAME_JUMP, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        elif key == ord(ctrl_action_end):  # mark end of action
            current_action = input("Enter action class: ")
            start_frame = frame_number - ACTION_DURATION_FRAMES
            actions.append({"class": current_action, "start": start_frame, "end": frame_number})
            print(f"Action [{current_action}] marked at frame:", frame_number)

    cap.release()
    cv2.destroyAllWindows()
    save_annotations(output_path)


def save_annotations(output_path):
    # txt_file = os.path.splitext(video_path)[0] + ".txt"
    with open(output_path, "a") as f:
        for action in actions:
            f.write(f"{action['class']} {action['start']} {action['end']}\n")
    print(f"Annotations saved to {output_path}")

def process_videos_in_directory(directory, annotation_directory):
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    videos = [f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in video_extensions]

    for i, video in enumerate(videos):
        video_path = os.path.join(directory, video)
        print(f"Processing [{i+1}/{len(videos)}]: {video_path}")
        annotate_video(video_path, annotation_directory)

if __name__ == "__main__":
    video_dir = "../resources/MY_SB_train/"
    annotations_dir = "../resources/MY_SB_annotation"
    if os.path.isdir(video_dir):
        process_videos_in_directory(video_dir, annotations_dir)
    else:
        print("Invalid directory path.")