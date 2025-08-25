import glob
from time import sleep

import pandas
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os
from concurrent.futures import ThreadPoolExecutor

sleep_time = 0.1
images_path = "/mnt/big/indeed/lepisea/images"
original_images_path = "/mnt/big/indeed/lepisea/original_images"
def m():
    tables = [pandas.read_csv(fn) for fn in glob.glob("/mnt/big/indeed/lepisea/source_data/*.csv")]

    table = pandas.concat(tables)

    # https://medialib.naturalis.nl/file/id/ZMA.INS.1531801_1/format/large
    print(table.columns)

    # Directory to save images
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(original_images_path, exist_ok=True)

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=32) as executor:
        executor.map(download_and_resize, [row for _, row in table.iterrows()])


def download_and_resize(row):
    global sleep_time
    reg_nr = row['Registration nr']
    url = f"https://medialib.naturalis.nl/file/id/{reg_nr}_1/format/medium"
    resized_path = f"{images_path}/{reg_nr}.jpg"

    if os.path.exists(resized_path):
        return

    try:
        # Download the image
        response = requests.get(url)
        if response.status_code == 429:
            sleep_time *= 1.1
            print(f"{sleep_time=}")
        else:
            sleep_time *= 0.99
        response.raise_for_status()

        # Save original image
        # original_path = f"{original_images_path}/{reg_nr}.jpg"
        # with open(original_path, 'wb') as f:
        #     f.write(response.content)

        # Resize the image
        img = Image.open(BytesIO(response.content))
        width, height = img.size

        # Calculate new dimensions to maintain aspect ratio
        if width > height:
            new_width = 512
            new_height = int(512 * height / width)
        else:
            new_height = 512
            new_width = int(512 * width / height)

        # Resize the image
        img_resized = img.resize((new_width, new_height))
        # Save resized image
        img_resized.save(resized_path)

        print(f"Processed {reg_nr}")
    except Exception as e:
        print(f"Error processing {reg_nr}: {e}")
    sleep(sleep_time)



print("All images processed.")



if __name__ == '__main__':
    m()