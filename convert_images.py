import pandas as pd
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

# Function to convert a single TIFF image to PNG
def convert_tiff_to_png(args):
    input_path, output_path = args
    try:
        img = Image.open(input_path)
        img.save(output_path, "PNG")
        # return f"Converted: {os.path.basename(input_path)} -> {os.path.basename(output_path)}"
    except Exception as e:
        return f"Failed to convert {os.path.basename(input_path)}: {e}"

# Function to prepare the list of tasks
def prepare_conversion_tasks(df, tiff_folder, output_folder):
    tasks = []
    for _, row in tqdm(df.iterrows()):
        tiff_filename = row[0]
        input_path = os.path.join(tiff_folder, tiff_filename)
        output_filename = os.path.splitext(tiff_filename)[0] + ".png"
        output_path = os.path.join(output_folder, output_filename)
        # Ensure the output folder exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        tasks.append((input_path, output_path))
    return tasks

# Main function to process the CSV file
def process_csv_and_convert_images(csv_file, tiff_folder, output_folder):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Prepare tasks
    tasks = prepare_conversion_tasks(df, tiff_folder, output_folder)
    
    # Use a multiprocessing Pool to parallelize the conversion
    with Pool(processes=20) as pool:
        for result in tqdm(pool.imap(convert_tiff_to_png, tasks), total=len(tasks)):
            print(result)

# Example usage
if __name__ == "__main__":
    csv_file = "mll_beluga_labels.csv"          # Path to the CSV file
    tiff_folder = "/lustre/groups/labs/marr/qscd01/datasets/230824_MLL_BELUGA/RawImages"  # Folder containing TIFF images
    output_folder = "/lustre/groups/shared/histology_data/hematology_data/mll_beluga_unlabeled"    # Folder where PNGs will be saved
    
    process_csv_and_convert_images(csv_file, tiff_folder, output_folder)
