

import os
import random

def remove_90_percent_files(directory):
    # List all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Calculate 90% of the files
    num_to_remove = int(0.9 * len(files))

    # Randomly select files to remove
    files_to_remove = random.sample(files, num_to_remove)

    # Remove the selected files
    for file in files_to_remove:
        os.remove(os.path.join(directory, file))
        print(f"Removed: {file}")

# Example usage:
directory = "/mnt/big/indeed/video_hazehorst_bandb/images"
remove_90_percent_files(directory)
