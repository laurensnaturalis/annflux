
import os

def rename_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if '-' in filename:
            new_filename = filename.replace('-', '_')
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)
            print(f'Renamed: {filename} -> {new_filename}')

# Example usage:
folder_path = '/mnt/big/indeed/plantstream/images'  # Replace with your folder path
rename_files_in_folder(folder_path)
