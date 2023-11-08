import os

def rename_files(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Sort the files to ensure a consistent order
    files.sort()

    # Set the starting number for renaming
    start_number = 28

    # Iterate through each file in the folder
    for file_name in files:
        # Build the full paths to the old and new files
        old_path = os.path.join(folder_path, file_name)
        new_path = os.path.join(folder_path, f"{start_number}.jpg")

        # Rename the file
        os.rename(old_path, new_path)

        # Increment the start number for the next iteration
        start_number += 1

# Specify the path to your folder
folder_path = r"dataset\images\blight"

# Call the function to rename files in the folder
rename_files(folder_path)
