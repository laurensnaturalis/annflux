import logging
import os
import shutil
import threading
import time

from PIL import Image
import pandas
import requests
from filetype import filetype

logger = logging.getLogger("remote_io")

def fetch_images(
    media_list: pandas.DataFrame,
    image_save_path,
    uid_column="image_id",
    url_column="image_url",
    max_width=None,
    max_num_concurrent_threads=32,
sleep_seconds:float=0.1
) -> (list[str], list[str]):
    failed_uids = []
    photo_paths = []
    download_threads = []
    for i, row in media_list.iterrows():
        image_id = row[uid_column]
        image_url = row[url_column]

        photo_save_path = os.path.join(image_save_path, f"{image_id}.jpg")

        time.sleep(sleep_seconds)

        # download and store the photo
        if not os.path.exists(photo_save_path):
            thread = threading.Thread(
                target=store_image,
                args=(image_url, photo_save_path, max_width),
            )
            # start up to max_num_concurrent_threads
            thread.start()
            download_threads.append(thread)

        # wait for all downloads to complete when number of downloading threads exceeds maximum
        if len(download_threads) > max_num_concurrent_threads:
            for thread in download_threads:
                thread.join()
            download_threads = []

        if not os.path.exists(photo_save_path):
            failed_uids.append(image_id)
        photo_paths.append(photo_save_path)
    return photo_paths, failed_uids


def store_image(
    image_url_or_path: str, photo_save_path: str, max_image_width: int = None
) -> bool:
    """
    Copies or downloads images and saves them
    :param image_url_or_path: source path, can be local file or URL
    :param photo_save_path: target path
    :param max_image_width: if not None, specifies the image width the image will be resized to if wider
    :return: True if the operation succeeded, otherwise False
    """
    success = True
    is_local_file = (
        "http://" not in image_url_or_path and "https://" not in image_url_or_path
    )
    print(image_url_or_path)
    if is_local_file:
        shutil.copy(image_url_or_path, photo_save_path)
    else:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/39.0.2171.95 Safari/537.36"
            }
            r = requests.get(image_url_or_path, allow_redirects=True, headers=headers)
            if r.status_code < 400:
                open(photo_save_path, "wb").write(r.content)
                true_extension = filetype.guess(photo_save_path).extension
                valid_extensions = ["jpeg", "gif", "png", "bmp", "jpg"]

                if true_extension not in ["jpg",]:
                    if true_extension in valid_extensions:
                        logger.warning(
                            f"Saving {image_url_or_path} with extension {true_extension} as JPEG"
                        )
                        im = Image.open(photo_save_path)
                        im.convert("RGB").save(photo_save_path, quality=95)
                    else:
                        logger.error(
                            f"Failed to download {image_url_or_path}, extension not in {valid_extensions}, header = {open(photo_save_path, 'rb').read(10)}"
                        )
                        os.remove(photo_save_path)
                        success = False

                # resize image if specified
                if max_image_width is not None:
                    im = Image.open(photo_save_path)
                    if im.width > max_image_width:
                        im.thumbnail(
                            (
                                max_image_width,
                                im.height * max_image_width / float(im.width),
                            )
                        )
                        im.save(photo_save_path, quality=95)
            else:
                logger.error(
                    f"Failed to download {image_url_or_path}, status_code= {r.status_code}"
                )
                success = False
        except requests.RequestException:
            logger.error(
                "Failed to download {}".format(image_url_or_path), exc_info=True
            )
            success = False

    return success
