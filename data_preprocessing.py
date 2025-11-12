import os
import gzip
import shutil
import pandas as pd
from scipy.io import mmread

def extract_raw_patient_data(input_dir):
    """
    Extracts the raw patient data from the downloaded archives.

    Args:
         input_dir: The root directory where the raw data archives have been downloaded.
    """
    # Define the directory containing the files
    output_dir = os.path.join(input_dir, "processed_data_csv")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # File patterns to look for
    file_patterns = ["processed_barcodes.tsv", "processed_genes.tsv", "processed_metadata.tsv", "processed_matrix.mtx"]

    # Loop through files in the input directory
    for filename in os.listdir(input_dir):
        # Check if the file matches any of the desired patterns
        if any(pattern in filename for pattern in file_patterns):
            file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, filename.replace(".tsv", ".csv").replace(".mtx", ".csv"))

            # Extract and convert gzipped files if necessary
            if filename.endswith(".gz"):
                with gzip.open(file_path, 'rt') as gz_file, open(output_file_path, 'w') as out_file:
                    shutil.copyfileobj(gz_file, out_file)
            else:
                # For uncompressed files, copy them to the new location with a .csv extension
                shutil.copyfile(file_path, output_file_path)

    print(f"Processed data has been extracted and saved to {output_dir}")

def process_sparse_matrices(input_dir):
    """
    Extracts raw data, create dense DataFrames from sparse matrices and save to CVS files.

    Args:
         input_dir: The root directory where the raw data archives have been downloaded.
    """
    # Define output directories
    output_dir = os.path.join(input_dir, "processed_matrices_csv")
    os.makedirs(output_dir, exist_ok=True)

    # Process only the *_processed_matrix.mtx.gz files
    for filename in os.listdir(input_dir):
        if filename.endswith("_processed_matrix.mtx.gz"):
            # Define file paths
            gz_file_path = os.path.join(input_dir, filename)
            decompressed_path = gz_file_path.replace(".gz", "")
            output_csv_path = os.path.join(output_dir, filename.replace(".mtx.gz", ".csv"))

            try:
                # Decompress the .gz file
                with gzip.open(gz_file_path, 'rt') as gz_file:
                    with open(decompressed_path, 'w') as decompressed_file:
                        decompressed_file.write(gz_file.read())

                # Read the sparse matrix
                sparse_matrix = mmread(decompressed_path)
                # Convert to a dense DataFrame
                dense_matrix = pd.DataFrame(sparse_matrix.toarray())
                # Save the dense matrix as a CSV
                dense_matrix.to_csv(output_csv_path, index=False)
                print(f"Processed matrix saved: {output_csv_path}")

                # Clean up: Remove the temporary decompressed file
                os.remove(decompressed_path)

            except Exception as e:
                print(f"Error processing {gz_file_path}: {e}")

    print(f"All processed matrices have been saved to {output_dir}")