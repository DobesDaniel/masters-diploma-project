"""

    Program that plays videos from directory, and allows to annotate action within
    - play and pause, back and forward few frames
    - start and end of annotated action adding class of that action
    - output text file per video with each line contains class of action, frame where the action starts and ends

    - basically almost same as annotate_video.py

"""

import cv2
import os
import glob
import csv

# CONTROLS
ctrl_quit = 'q'
ctrl_play_pause = ' '
ctrl_skip_back = 'a'
ctrl_skip_forward = 'd'
ctrl_action_start = 's'
ctrl_action_end = 'f'


FRAME_JUMP = 100  # number of frames to jump forward or backward

actions = []
start_frame = None
current_action = None

def annotate_video(video_path, output_directory):

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_directory, f"{video_name}.txt")

    global actions
    actions = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    cv2.namedWindow("Video Annotator", cv2.WINDOW_NORMAL)  # Allows resizing
    # cv2.resizeWindow("Video Annotator", 1325, 1014)  # Set default size to 800x600
    # cv2.moveWindow("Video Annotator", 600, 200)  # Move window to (x=100, y=100)

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
    start_frame = None
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

        if key == ord(ctrl_quit):  # quit
            break
        elif key == ord(ctrl_play_pause):  # pause
            cv2.waitKey(0)
        elif key == ord(ctrl_skip_forward):  # move forward
            frame_number = min(frame_number + FRAME_JUMP, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        elif key == ord(ctrl_skip_back):  # move backward
            frame_number = max(frame_number - FRAME_JUMP, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        elif key == ord(ctrl_action_start):
            start_frame = frame_number
            print(f"Action start at frame: {start_frame}")
        elif key == ord(ctrl_action_end):  # mark end of action
            if start_frame == None:
                print(f"There is not marked action start")
                continue
            current_action = input("Enter action class: ")
            actions.append({"class": current_action, "start": start_frame, "end": frame_number})
            print(f"Action [{current_action}] marked from {start_frame} to {frame_number}")
            start_frame = None

    cap.release()
    cv2.destroyAllWindows()
    save_annotations(video_name, output_directory)



def save_annotations(video_name, output_directory):
    output_csv = os.path.join(output_directory, f"{video_name}_gt.csv")
    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["class", "start_frame", "end_frame"])
        for action in actions:
            writer.writerow([action["class"], action["start"], action["end"]])
    print(f"Annotations saved to {output_csv}")


def process_videos_in_directory(directory, annotation_directory):
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    videos = [f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in video_extensions]

    for i, video in enumerate(videos):
        video_path = os.path.join(directory, video)
        print(f"Processing [{i+1}/{len(videos)}]: {video_path}")
        annotate_video(video_path, annotation_directory)

if __name__ == "__main__":
    video_dir = "../resources/MY_SB_test/"
    # video_dir = "../resources/SB_test/"
    annotations_dir = "../resources/SB_tmp"
    # annotations_dir = "../resources/SB_video_ground_truth"
    if os.path.isdir(video_dir):
        process_videos_in_directory(video_dir, annotations_dir)
    else:
        print("Invalid directory path.")