import os
import csv
import concurrent.futures
from tqdm import tqdm

# Path to the CSV file that contains the folders in the "path" column.
CSV_FILE = "/home/haicu/sophia.wagner/projects/dinov2/Beluga_may_combined_test_excluded.csv"

# Output text file that will list all image paths.
OUTPUT_TXT = "beluga_march_april_may_combined_train.txt"

# Define the image file extensions you wish to capture (in lowercase).
IMAGE_EXTENSIONS = {".tif"}

def get_image_files_from_folder(folder):
    """
    Given a folder path, returns a list of full paths for image files
    whose extensions match one of the allowed IMAGE_EXTENSIONS.
    """
    image_files = []
    try:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                ext = os.path.splitext(filename)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    image_files.append(file_path)
    except Exception as e:
        print(f"Error accessing folder {folder}: {e}")
    return image_files

def read_folders_from_csv(csv_file):
    """
    Reads the CSV file and extracts folder paths from the 'path' column.
    Returns a list of folder paths.
    """
    folders = []
    try:
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                folder = row.get("path")
                if folder:
                    folders.append(folder)
                else:
                    print("Warning: Missing 'path' value in CSV row.")
    except FileNotFoundError:
        print(f"CSV file not found: {csv_file}")
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
    return folders

def main():
    # Step 1: Read folder paths from CSV.
    folders = read_folders_from_csv(CSV_FILE)
    if not folders:
        print("No folders to process. Exiting.")
        return

    # Filter out non-existing directories and warn about them.
    valid_folders = []
    for folder in folders:
        if os.path.isdir(folder):
            valid_folders.append(folder)
        else:
            print(f"Warning: The folder '{folder}' does not exist or is not a directory.")

    if not valid_folders:
        print("No valid directories found. Exiting.")
        return

    # Step 2: Process folders in parallel to gather image file paths.
    all_image_paths = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit a task for each valid folder.
        future_to_folder = {
            executor.submit(get_image_files_from_folder, folder): folder
            for folder in valid_folders
        }
        # Wrap the as_completed iterator with tqdm to display a progress bar.
        for future in tqdm(concurrent.futures.as_completed(future_to_folder), total=len(future_to_folder),
                           desc="Processing folders"):
            folder = future_to_folder[future]
            try:
                image_paths = future.result()
                if image_paths:
                    all_image_paths.extend(image_paths)
                else:
                    print(f"No image files found in: {folder}")
            except Exception as exc:
                print(f"Folder {folder} generated an exception: {exc}")

    # Step 3: Write the collected image paths to the output text file.
    try:
        with open(OUTPUT_TXT, "w") as out_f:
            for image_path in all_image_paths:
                out_f.write(image_path + "\n")
        print(f"Successfully wrote {len(all_image_paths)} image paths to '{OUTPUT_TXT}'")
    except Exception as e:
        print(f"An error occurred while writing to '{OUTPUT_TXT}': {e}")

if __name__ == "__main__":
    main()
