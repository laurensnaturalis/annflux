
import argparse
import os

def rename_files_in_folder(folder_path, apply=False):
    for filename in os.listdir(folder_path):
        if '-' in filename:
            new_filename = filename.replace('-', '_')
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            if apply:
                os.rename(old_path, new_path)
            print(f'{"Renamed" if apply else "Would rename"}: {filename} -> {new_filename}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace dashes with underscores in filenames")
    parser.add_argument("directory", help="path to directory")
    parser.add_argument("--apply", action="store_true", help="actually rename files (default is dry-run)")
    args = parser.parse_args()

    rename_files_in_folder(args.directory, apply=args.apply)
