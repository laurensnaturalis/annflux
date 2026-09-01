"""Download and resize missing images from CSV URLs."""
import os
import shutil
import time
import pandas
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# Rate limiting config
MIN_REQUEST_INTERVAL = 0.5  # seconds between requests
last_request_time = 0


def normalize_filename(filename: str) -> str:
    """Replace dots with underscores in filename (keep extension)."""
    name, ext = os.path.splitext(filename)
    return name.replace(".", "_") + ext


def rate_limited_request(url: str, max_retries: int = 3, timeout: int = 30) -> tuple[requests.Response | None, int]:
    """Make a rate-limited request with retry logic for 429 errors.
    
    Returns (response, status_code). On success status_code is 200.
    On failure response is None and status_code indicates the error.
    """
    global last_request_time
    last_status = 0
    
    for attempt in range(max_retries):
        # Enforce minimum interval between requests
        elapsed = time.time() - last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        
        try:
            last_request_time = time.time()
            response = requests.get(url, timeout=timeout)
            last_status = response.status_code
            
            if response.status_code == 429:
                # Too many requests - back off and retry
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                time.sleep(wait_time)
                continue
            
            if response.status_code != 200:
                return None, response.status_code
            
            return response, 200
            
        except requests.exceptions.Timeout:
            last_status = -1
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
        except requests.exceptions.RequestException:
            last_status = -2
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
    
    return None, last_status


def download_and_resize(url: str, output_path: str, size: int = 512) -> tuple[bool, int]:
    """
    Download image from URL, resize to size x size (keep aspect ratio), and save.
    
    Returns (success, status_code).
    """
    try:
        # Download with rate limiting and retries
        response, status_code = rate_limited_request(url)
        if response is None:
            print(f"Failed ({status_code}): {url}")
            return False, status_code
        
        # Open image
        img = Image.open(BytesIO(response.content))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize keeping aspect ratio, then center crop to square
        width, height = img.size
        
        if width != height:
            # Resize so smallest dimension equals target size
            if width < height:
                new_width = size
                new_height = int(height * size / width)
            else:
                new_height = size
                new_width = int(width * size / height)
            img = img.resize((new_width, new_height), Image.LANCZOS)
        else:
            # Already square, just resize
            img = img.resize((size, size), Image.LANCZOS)
        
        # Save
        img.save(output_path, "JPEG", quality=95)
        return True, 200
        
    except Exception as e:
        print(f"Error downloading/resizing {url}: {e}")
        return False, -3


def load_failures(failures_path: str) -> set:
    """Load previously failed URLs from CSV."""
    if os.path.exists(failures_path):
        df = pandas.read_csv(failures_path)
        return set(df["url"].tolist())
    return set()


def log_failure(failures_path: str, url: str, status_code: int):
    """Append a failure entry to the failures CSV."""
    write_header = not os.path.exists(failures_path)
    with open(failures_path, "a") as f:
        if write_header:
            f.write("url,status_code\n")
        f.write(f'"{url}",{status_code}\n')


def main():
    csv_path = "/home/lhogeweg/Documents/annflux_ln/src/annflux/projects/lepisea_june/Papillot_And_Malaysian_Undersides_croppaths.csv"
    dest_dir = "/mnt/big/indeed/lepisea2/images"
    fallback_dir = "/mnt/big/indeed/lepisea/images"
    failures_path = "/home/lhogeweg/Documents/annflux_ln/src/annflux/projects/lepisea_june/download_failures.csv"
    
    # Ensure destination exists
    os.makedirs(dest_dir, exist_ok=True)
    
    # Load previously failed URLs to skip
    known_failures = load_failures(failures_path)
    
    # Read CSV
    table = pandas.read_csv(csv_path, encoding="latin-1", low_memory=False)
    
    downloaded = 0
    existing = 0
    copied_from_fallback = 0
    skipped_known_failure = 0
    failed = 0
    
    for _, row in tqdm(table.iterrows(), total=len(table), desc="Processing"):
        image_filename = row["image_filename"]
        image_url = row["image_url"]
        
        # Skip rows with missing data
        if pandas.isna(image_filename) or pandas.isna(image_url):
            failed += 1
            continue
        
        image_url = str(image_url)
        
        # Skip previously failed URLs
        if image_url in known_failures:
            skipped_known_failure += 1
            continue
        
        # Normalize filename (replace dots with underscores, keep extension)
        target_filename = normalize_filename(str(image_filename))
        output_path = os.path.join(dest_dir, target_filename)
        
        # Check if file already exists in dest
        if os.path.exists(output_path):
            existing += 1
            continue
        
        # Check if file exists in fallback dir
        fallback_path = os.path.join(fallback_dir, target_filename)
        if os.path.exists(fallback_path):
            shutil.copy2(fallback_path, output_path)
            copied_from_fallback += 1
            continue
        
        # Download and resize
        success, status_code = download_and_resize(image_url, output_path, size=512)
        if success:
            downloaded += 1
        else:
            log_failure(failures_path, image_url, status_code)
            failed += 1
    
    print(f"\nSummary: {downloaded} downloaded, {copied_from_fallback} copied from fallback, "
          f"{existing} already existed, {skipped_known_failure} skipped (known failures), {failed} failed")


if __name__ == "__main__":
    main()
