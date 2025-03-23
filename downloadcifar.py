import urllib.request
import tarfile
import os

import urllib.request
import tarfile
import os

def download_and_extract(url, download_path, extract_path):
    # Download the file
    print("Downloading file from:", url)
    urllib.request.urlretrieve(url, download_path)
    print("Download complete.")
    
    # Extract the tar.gz file
    print("Extracting files to:", extract_path)
    with tarfile.open(download_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    print("Extraction complete.\n")

if __name__ == "__main__":
    # CIFAR-10
    cifar10_url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    cifar10_download_path = "datasets/cifar-10/cifar-10-binary.tar.gz"
    cifar10_extract_path = "datasets/cifar-10/cifar-10-binary"
    
    if not os.path.exists(cifar10_extract_path):
        os.makedirs(cifar10_extract_path)
    
    download_and_extract(cifar10_url, cifar10_download_path, cifar10_extract_path)
    
    # CIFAR-100
    cifar100_url = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"
    cifar100_download_path = "datasets/cifar-100/cifar-100-binary.tar.gz"
    cifar100_extract_path = "datasets/cifar-100/cifar-100-binary"
    
    if not os.path.exists(cifar100_extract_path):
        os.makedirs(cifar100_extract_path)
    
    download_and_extract(cifar100_url, cifar100_download_path, cifar100_extract_path)

