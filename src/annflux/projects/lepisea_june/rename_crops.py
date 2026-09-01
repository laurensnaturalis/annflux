"""Copy cropped images from source paths to destination with proper filenames."""
import os
import shutil
import pandas


def transform_path(source_dir, resized_path):
    r"""
    Transform Windows CSV path to Linux path.
    
    CSV paths are like:
      E:\All_Lepidoptera_features\processed_224\Naturalis_Papillot_project\000\filename.jpg
    
    We need to map to:
      /mnt/big/indeed/lepisea2/Naturalis_Papillot_project_crops/Naturalis_Papillot_project/000/filename.jpg
    
    Strategy: Remove the prefix up to and including "Naturalis_Papillot_project\",
    then join with source_dir + "Naturalis_Papillot_project".
    """
    # Handle NaN/None values
    if pandas.isna(resized_path):
        return None
    
    # Convert to string and normalize separators
    path = str(resized_path).replace("\\", "/")
    
    # Find the position of "Naturalis_Papillot_project/" and keep everything after it
    marker = "Naturalis_Papillot_project/"
    pos = path.find(marker)
    if pos != -1:
        # Get the relative path starting from the subfolder (e.g., "000/filename.jpg")
        relative = path[pos + len(marker):]
    else:
        # Fallback: just use the filename
        relative = os.path.basename(path)
    
    # Build the target path: source_dir/Naturalis_Papillot_project/relative
    base_project_dir = os.path.join(source_dir, "Naturalis_Papillot_project")
    full_path = os.path.join(base_project_dir, relative)
    
    return full_path


def find_file(source_dir, resized_path):
    """Transform path and check if file exists."""
    full_path = transform_path(source_dir, resized_path)
    if full_path and os.path.exists(full_path):
        return full_path
    return None


def main():
    csv_path = "/home/lhogeweg/Documents/annflux_ln/src/annflux/projects/lepisea_june/Papillot_And_Malaysian_Undersides_croppaths.csv"
    source_dir = "/mnt/big/indeed/lepisea2/Naturalis_Papillot_project_crops"
    dest_dir = "/mnt/big/indeed/lepisea2/images"

    # Ensure destination exists
    os.makedirs(dest_dir, exist_ok=True)

    # Read CSV
    table = pandas.read_csv(csv_path, encoding="latin-1")

    copied = 0
    missing = 0

    for _, row in table.iterrows():
        resized_path = row["resized_image_path"]
        target_filename = row["image_filename"]

        # Skip rows with missing data
        if pandas.isna(resized_path) or pandas.isna(target_filename):
            missing += 1
            continue

        # Convert to strings and normalize filename (replace . with _, but keep extension)
        target_filename = str(target_filename)
        name, ext = os.path.splitext(target_filename)
        target_filename = name.replace(".", "_") + ext

        # Find source file using path transformation
        src_path = find_file(source_dir, resized_path)

        # Build full destination path
        dest_path = os.path.join(dest_dir, target_filename)

        # Copy if found
        if src_path:
            shutil.copy2(src_path, dest_path)
            print(f"Copied: {src_path} -> {dest_path}")
            copied += 1
        else:
            print(f"Missing: {resized_path}")
            missing += 1

    print(f"\nSummary: {copied} copied, {missing} missing")


if __name__ == "__main__":
    main()
