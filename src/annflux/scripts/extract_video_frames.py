from __future__ import annotations

import datetime
import glob
import os.path
from pathlib import Path

import cv2
import argparse
import re

# Function to extract frames
import ffmpeg
import pandas
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


def frame_capture(
    input_folder: str, output_folder: str, existing_original_paths: set[str]
) -> pandas.DataFrame:
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    count = 0
    frame_rate = int(43 / 3)  # TODO: from file
    rows = []
    for video_path in (
        set(
            glob.glob(f"{str(input_folder)}/**/*.mp4")
            + glob.glob(f"{str(input_folder)}/*.mp4")
        )
        - existing_original_paths
    ):
        basename = os.path.basename(video_path)
        tokens = os.path.split(video_path)[0].split(os.sep)
        parent_folder = tokens[-1]
        num_frames = ffmpeg.probe(video_path.strip())["streams"][0]["nb_frames"]
        date_time = get_datetime(basename)
        basename = f"{parent_folder}_{basename}"

        print(f"estimated seconds {int(num_frames) / frame_rate}")
        vid_obj = cv2.VideoCapture(video_path.strip())

        # checks whether frames were extracted
        success = 1
        max_width = 1820  # TODO: configurable

        while success:
            # TODO: use tqdm
            success, image = vid_obj.read()
            if count % frame_rate == 0:
                maxsize = (
                    max_width,
                    int(max_width * (2160.0 / 3840)),
                )  # TODO: based on actual frame size
                try:
                    image = cv2.resize(image, maxsize)
                except:  # noqa
                    print(video_path, count, "failed")
                    continue

                # Saves the frames with frame-count
                time_s = int(count / frame_rate)
                out_path = os.path.join(
                    output_folder,
                    f"{basename.replace('.', '_')}_t_{time_s}s.jpg",
                )
                cv2.imwrite(
                    out_path,
                    image,
                )

                rows.append(
                    {
                        "path": out_path,
                        "original_path": video_path,
                        "original_id": basename,
                        "video_created": date_time.isoformat(),
                        "datetime": (
                            date_time + datetime.timedelta(seconds=time_s)
                        ).isoformat(),
                        "time_offset_s": time_s,
                    }
                )

            count += 1
            print(video_path, count / frame_rate)
    return pandas.DataFrame(data=rows)


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
