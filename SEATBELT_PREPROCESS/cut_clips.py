"""

    Cut clips from a videos in ../resources/SB_rotated/
    based on frames annotated in ../resources/SB_annotation_2
    where video name is same as annotation file just with .txt
    annotations are on each line and there are 3 parts,
    first is class {fasten|unfasten|...} and starting frame for clip and ending frame for clip
    [class name] [starting frame] [ending frame]
    Create these clips into output directory ../resources/SB_cuts/{class name}

"""

import os
import cv2


def create_video_clip(video_path, start_frame, end_frame, output_path):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')      # for .mp4
    fourcc = cv2.VideoWriter_fourcc(*'XVID')        # for .avi

    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1

    cap.release()
    out.release()


def process_annotations(annotation_dir, video_dir, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(annotation_dir):
        if filename.endswith(".txt"):
            annotation_file_path = os.path.join(annotation_dir, filename)
            video_name = os.path.splitext(filename)[0]
            # video_file_path = os.path.join(video_dir, f"{video_name}.MP4")  # expected uppercase .MP4
            video_file_path = os.path.join(video_dir, f"{video_name}.avi")

            if not os.path.exists(video_file_path):
                # video_file_path = os.path.join(video_dir, f"{video_name}.mp4")
                video_file_path = os.path.join(video_dir, f"{video_name}.avi")

            if not os.path.exists(video_file_path):
                print(f"Video file {video_file_path} does not exist. Skipping.")
                continue

            with open(annotation_file_path, 'r') as file:
                lines = file.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) == 3:
                    class_name, start_frame, end_frame = parts
                    start_frame = int(start_frame)
                    end_frame = int(end_frame)

                    class_output_dir = os.path.join(output_dir, class_name)
                    if not os.path.exists(class_output_dir):
                        os.makedirs(class_output_dir)

                    # output_clip_path = os.path.join(class_output_dir, f"{video_name}_{start_frame}_{end_frame}.mp4")
                    output_clip_path = os.path.join(class_output_dir, f"{video_name}_{start_frame}_{end_frame}.avi")

                    print(f"Creating clip for {video_name} from frame {start_frame} to {end_frame}")
                    create_video_clip(video_file_path, start_frame, end_frame, output_clip_path)

            print(f"Processed annotation file: {filename}")


def main():
    # annotation_dir = "../resources/SB_annotation"
    # video_dir = "../resources/SB_train"
    # output_dir = "../resources/SB_cuts"

    annotation_dir = "../resources/MY_SB_annotation"
    video_dir = "../resources/MY_SB_train"
    output_dir = "../resources/MY_SB_cuts"

    process_annotations(annotation_dir, video_dir, output_dir)

    pass

if __name__ == "__main__":
    main()
    pass

