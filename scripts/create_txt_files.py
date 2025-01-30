import os
import argparse

def list_files(directory, extension, output_file):
    """
    Recursively finds all files with the given extension in the specified directory
    and writes their paths to the output file.
    
    :param directory: The root directory to search.
    :param extension: The file extension to look for.
    :param output_file: The output file to store the results.
    """
    with open(output_file, 'w') as f:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(extension):
                    f.write(os.path.join(root, file) + '\n')
    print(f"File paths written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List files with a specific extension and write to a file.")
    parser.add_argument("--directory", type=str, help="The directory to search.")
    parser.add_argument("--ext", type=str, help="The file extension to look for (e.g., .png).")
    parser.add_argument("--output_file", type=str, help="The output text file.")
    
    args = parser.parse_args()
    list_files(args.directory, args.ext, args.output_file)

# /lustre/groups/labs/marr/qscd01/datasets/papsmear_dataset_processed/Cervix93 png
# python scripts/create_txt_files.py --directory /lustre/groups/labs/marr/qscd01/datasets/papsmear_dataset_processed/Cervix93 --ext .png --output_file /lustre/groups/shared/users/peng_marr/DinoBloomv2/cytology_patches/cervix93.txt

# /lustre/groups/labs/marr/qscd01/datasets/papsmear_dataset_processed/LBC jpg
# python scripts/create_txt_files.py --directory /lustre/groups/labs/marr/qscd01/datasets/papsmear_dataset_processed/LBC --ext .jpg --output_file /lustre/groups/shared/users/peng_marr/DinoBloomv2/cytology_patches/lbc.txt

# /lustre/groups/labs/marr/qscd01/datasets/papsmear_dataset_processed/CCEDD png
# python scripts/create_txt_files.py --directory /lustre/groups/labs/marr/qscd01/datasets/papsmear_dataset_processed/CCEDD --ext .png --output_file /lustre/groups/shared/users/peng_marr/DinoBloomv2/cytology_patches/ccedd.txt

# /lustre/groups/labs/marr/qscd01/datasets/papsmear_dataset_processed/ISBI2014 png
# python scripts/create_txt_files.py --directory /lustre/groups/labs/marr/qscd01/datasets/papsmear_dataset_processed/ISBI2014 --ext .png --output_file /lustre/groups/shared/users/peng_marr/DinoBloomv2/cytology_patches/isbi2014.txt

# /lustre/groups/labs/marr/qscd01/datasets/papsmear_dataset_processed/Herlev_new BMP
# python scripts/create_txt_files.py --directory /lustre/groups/labs/marr/qscd01/datasets/papsmear_dataset_processed/Herlev_new --ext .BMP --output_file /lustre/groups/shared/users/peng_marr/DinoBloomv2/cytology_patches/herlev_new.txt

# /lustre/groups/labs/marr/qscd01/datasets/papsmear_dataset_processed/CRIC png
# python scripts/create_txt_files.py --directory /lustre/groups/labs/marr/qscd01/datasets/papsmear_dataset_processed/CRIC --ext .png --output_file /lustre/groups/shared/users/peng_marr/DinoBloomv2/cytology_patches/cric.txt

# /lustre/groups/labs/marr/qscd01/datasets/papsmear_dataset_processed/BTTFA png
# python scripts/create_txt_files.py --directory /lustre/groups/labs/marr/qscd01/datasets/papsmear_dataset_processed/BTTFA --ext .png --output_file /lustre/groups/shared/users/peng_marr/DinoBloomv2/cytology_patches/bttfa.txt

# /lustre/groups/labs/marr/qscd01/datasets/papsmear_dataset_processed/RepoMedUNM jpg
# python scripts/create_txt_files.py --directory /lustre/groups/labs/marr/qscd01/datasets/papsmear_dataset_processed/RepoMedUNM --ext .jpg --output_file /lustre/groups/shared/users/peng_marr/DinoBloomv2/cytology_patches/repomedunm.txt

# /lustre/groups/labs/marr/qscd01/datasets/papsmear_dataset_processed/BMT JPG
# python scripts/create_txt_files.py --directory /lustre/groups/labs/marr/qscd01/datasets/papsmear_dataset_processed/BMT --ext .JPG --output_file /lustre/groups/shared/users/peng_marr/DinoBloomv2/cytology_patches/bmt.txt