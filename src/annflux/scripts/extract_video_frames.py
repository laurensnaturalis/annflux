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
from tqdm import tqdm


def get_datetime(path: str) -> datetime.datetime | None:
    # Define the regex pattern
    match = re.match(r"^\d+_(\d{10})\.mp4$", path)
    if match:
        timestamp = match.group(1)
        result = datetime.datetime.fromtimestamp(int(timestamp))
        result = result.replace(tzinfo=pytz.utc)
        return result
    else:
        raise ValueError


def frame_capture(
    input_folder: str | Path, output_folder: str | Path, existing_original_paths: set[str]
) -> pandas.DataFrame:
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    frame_rate = int(43 / 3)  # TODO: from file
    rows = []
    for video_path in (
        set(
            glob.glob(f"{str(input_folder)}/**/*.mp4")
            + glob.glob(f"{str(input_folder)}/*.mp4")
        )
        - existing_original_paths
    ):
        count = 0
        basename = os.path.basename(video_path)
        tokens = os.path.split(video_path)[0].split(os.sep)
        parent_folder = tokens[-1]
        num_frames = ffmpeg.probe(video_path.strip())["streams"][0]["nb_frames"]
        date_time = get_datetime(basename)
        if date_time is None:
            raise RuntimeError()
        basename = f"{parent_folder}_{basename}"

        num_extracted_frames = int(num_frames) / frame_rate
        print(f"estimated seconds {num_extracted_frames}, {num_frames=}")
        vid_obj = cv2.VideoCapture(video_path.strip())

        # checks whether frames were extracted
        max_width = 1820  # TODO: configurable
        success = True
        with tqdm(
            total=int(num_extracted_frames), desc=f"extracting frames from {video_path}"
        ) as pbar:
            while success:
                if count % frame_rate == 0:
                    # Saves the frames with frame-count
                    time_s = int(count / frame_rate)
                    out_path = os.path.join(
                        output_folder,
                        f"{basename.replace('.', '_')}_t_{time_s}s.jpg",
                    )
                    if not os.path.exists(out_path):
                        success, image = vid_obj.read()

                        maxsize = (
                            max_width,
                            int(max_width * (2160.0 / 3840)),
                        )  # TODO: based on actual frame size
                        try:
                            image = cv2.resize(image, maxsize)
                        except:  # noqa
                            print(video_path, count, "failed")
                            continue

                        cv2.imwrite(
                            out_path,
                            image,
                        )
                    else:
                        success = vid_obj.grab()

                    # print(count, success, count / frame_rate)

                    if not success:
                        break

                    pbar.update(1)
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
                else:
                    success = vid_obj.grab()
                count += 1
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

    frame_capture(args.video_paths, args.out_folder, set())
