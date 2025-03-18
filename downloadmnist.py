import os
import urllib.request
import gzip

# Base URL for downloading MNIST dataset files
base_url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'

# List of files to download
files = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz'
]

# Get current working directory and define target directory
cwd = os.getcwd()
target_dir = os.path.join(cwd, 'datasets', 'mnist')

# Create target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Process each file
for file in files:
    # Construct full URL and file paths
    url = base_url + file
    gz_file = file
    decompressed_file = os.path.splitext(gz_file)[0]
    decompressed_path = os.path.join(target_dir, decompressed_file)
    
    # Skip if decompressed file already exists
    if os.path.exists(decompressed_path):
        print(f"{decompressed_file} already exists, skipping.")
        continue
    
    gz_path = os.path.join(target_dir, gz_file)
    
    # Download the compressed file
    print(f"Downloading {gz_file}...")
    try:
        urllib.request.urlretrieve(url, gz_path)
    except Exception as e:
        # Remove partially downloaded file if download fails
        if os.path.exists(gz_path):
            os.remove(gz_path)
        raise e
    
    # Decompress the file
    print(f"Decompressing to {decompressed_file}...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(decompressed_path, 'wb') as f_out:
            f_out.write(f_in.read())
    
    # Remove the compressed file after decompression
    os.remove(gz_path)

# Completion message
print("MNIST dataset downloaded and decompressed to datasets/mnist/")