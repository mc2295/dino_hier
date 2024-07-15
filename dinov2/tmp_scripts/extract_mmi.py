import os
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import label, find_objects, binary_dilation
from tqdm import tqdm

# Define the padding function to expand the purple regions
def pad_mask(mask, distance=2):
    structure = np.ones((2 * distance + 1, 2 * distance + 1), dtype=int)
    padded_mask = binary_dilation(mask, structure=structure).astype(mask.dtype)
    return padded_mask

# Define the function to count pixels in a connected component
def count_pixels(label_array, obj_slice):
    obj = label_array[obj_slice]
    return np.sum(obj > 0)

# Process a single image to extract patches
def process_image(image_path, target_folder, lower_purple_rgb, upper_purple_rgb):
    # Load the image
    image = Image.open(image_path)
    if image is None:
        return
    
    # Convert the image to RGB
    image_rgb = image.convert('RGB')
    image_np = np.array(image_rgb)
    
    # Threshold the image to get only purple colors
    mask = np.all(np.logical_and(image_np >= lower_purple_rgb, image_np <= upper_purple_rgb), axis=-1).astype(np.uint8) * 255
    
    # Pad the mask to consider pixels within a distance of 2
    padded_mask = pad_mask(mask, distance=2)
    
    # Find connected components with the padded mask
    labeled_array_padded, num_features_padded = label(padded_mask)
    
    # Find objects larger than 1000 pixels with the padded mask
    objects_slices_padded = find_objects(labeled_array_padded)
    
    # Calculate the center points of the filtered components
    centers = []
    for obj_slice in objects_slices_padded:
        pixel_count = count_pixels(labeled_array_padded, obj_slice)
        if pixel_count > 1000:
            y_center = (obj_slice[0].start + obj_slice[0].stop) // 2
            x_center = (obj_slice[1].start + obj_slice[1].stop) // 2
            centers.append((x_center, y_center))
    
    # Extract 256x256 images centered on these points
    image_offset=int(image_size/2)
    for i, center in enumerate(centers):
        x_center, y_center = center
        x_start = max(x_center - image_offset, 0)
        x_end = min(x_center + image_offset, image_np.shape[1])
        y_start = max(y_center - image_offset, 0)
        y_end = min(y_center + image_offset, image_np.shape[0])
        
        # Extract the patch
        patch = image_np[y_start:y_end, x_start:x_end]
        patch_image = Image.fromarray(patch)
        
        # Resize to 256x256 if needed
        patch_image = ImageOps.fit(patch_image, (image_size, image_size), method=Image.LANCZOS)
        
        # Save the patch
        patch_filename = os.path.join(target_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_patch_{i}.png")
        patch_image.save(patch_filename)

# Main function to process all images in a directory
def process_directory(source_folder, target_folder, lower_purple_rgb, upper_purple_rgb):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder, mode=0o777)
        os.chmod(target_folder, 0o777)
    image_files = [f for f in os.listdir(source_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    for filename in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(source_folder, filename)
        process_image(image_path, target_folder, lower_purple_rgb, upper_purple_rgb)

# Parameters
source_folder = '/ictstr01/groups/shared/histology_data/MiMM_SBILab/MiMM-SBILab'  # Change this to your source folder
target_folder = '/ictstr01/groups/shared/histology_data/MiMM_SBILab/patched'  # Change this to your target folder
lower_purple_rgb = np.array([75, 40, 135])
upper_purple_rgb = np.array([120, 80, 150])
image_size=1024
# Process the directory
process_directory(source_folder, target_folder, lower_purple_rgb, upper_purple_rgb)
