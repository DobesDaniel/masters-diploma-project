"""
    This wil just flip the video into normal orientation

"""

import os
import subprocess


def flip_video_ffmpeg(input_path, output_path):

    # command = [
    #     "ffmpeg",
    #     "-i", input_path,
    #     "-vf", "hflip",  # Flip horizontally
    #     "-c:a", "copy",  # Copy audio without re-encoding
    #     output_path
    # ]
    command = [
        "ffmpeg",
        "-i", input_path,
        "-vf", "hflip",      # Horizontal flip
        "-c:v", "libx264",   # Use H.264 codec
        "-crf", "18",        # Constant Rate Factor: lower is better quality (18–23 is typical)
        "-preset", "veryfast",  # Speed vs. compression trade-off
        "-c:a", "copy",     # Copy audio
        output_path
    ]

    try:
        print(f"Processing {input_path}...")
        subprocess.run(command, check=True)
        print(f"Saved flipped video to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {input_path}: {e}")


if __name__ == "__main__":
    in_directory = "../resources/MY_SB_original"
    out_directory = "../resources/MY_SB_flipped"

    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    videos = [f for f in os.listdir(in_directory) if os.path.splitext(f)[1].lower() in video_extensions]

    if not videos:
        print(f"No videos found in {in_directory}")

    for video in videos:
        video_path = os.path.join(in_directory, video)
        output_path = os.path.join(out_directory, video)
        flip_video_ffmpeg(video_path, output_path)
