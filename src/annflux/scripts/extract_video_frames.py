from __future__ import annotations

import datetime
import json
import os.path
from pathlib import Path

import cv2
import argparse
import re

# Function to extract frames
import ffmpeg
import pytz


def get_datetime(path: str) -> datetime.datetime | None:
    # Define the regex pattern
    match = re.match(r"^\d+_(\d{10})\.mp4$", path)
    if match:
        timestamp = match.group(1)
        result = datetime.datetime.utcfromtimestamp(int(timestamp))
        result = result.replace(tzinfo=pytz.utc)
        return result
    else:
        raise ValueError


def frame_capture(paths, annflux_folder, subfolder="images"):
    annflux_folder = Path(annflux_folder)
    video_meta_path = annflux_folder / "video_meta.json"
    annflux_folder = annflux_folder / subfolder
    annflux_folder.mkdir(exist_ok=True)
    count = 0
    frame_rate = int(43 / 3)  # TODO: from file
    if os.path.exists(video_meta_path):
        video_meta = json.load(open(video_meta_path))
    else:
        video_meta = {"videos": []}
    for path in paths:
        basename = os.path.basename(path)
        tokens = os.path.split(path)[0].split(os.sep)
        parent_folder = tokens[-1]
        num_frames = ffmpeg.probe(path.strip())["streams"][0]["nb_frames"]
        date_time = get_datetime(basename)
        basename = f"{parent_folder}_{basename}"
        video_meta["videos"].append(
            {
                "path": path,
                "basename": basename,
                "created": date_time.isoformat(),
            }
        )
        with open(video_meta_path, "w") as f:
            json.dump(
                video_meta,
                f,
                indent=2,
            )
        print(f"estimated seconds {int(num_frames) / frame_rate}")
        vid_obj = cv2.VideoCapture(path.strip())

        # checks whether frames were extracted
        success = 1
        max_width = 1820

        while success:
            success, image = vid_obj.read()
            if count % frame_rate == 0:
                maxsize = (
                    max_width,
                    int(max_width * (2160.0 / 3840)),
                )  # TODO: based on actual frame size
                try:
                    image = cv2.resize(image, maxsize)
                except:
                    print(path, count, "failed")
                    continue

                # Saves the frames with frame-count
                cv2.imwrite(
                    os.path.join(
                        annflux_folder,
                        f"{basename.replace('.', '_')}_t_{int(count / frame_rate)}s.jpg",
                    ),
                    image,
                )

            count += 1
            print(path, count / frame_rate)


# Driver Code
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Capture frames from video files.")

    # Adding arguments for the output directory and video paths
    parser.add_argument(
        "video_paths",
        type=str,
        nargs="+",
        help="Paths to video files, separated by newlines.",
    )

    parser.add_argument(
        "out_folder",
        type=str,
        help="The directory where frames will be saved.",
    )

    args = parser.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)

    frame_capture(args.video_paths, args.out_folder)
