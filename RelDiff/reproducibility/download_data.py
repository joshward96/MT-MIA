import os
import zipfile

import gdown

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(current_dir, ".."))


def download_and_extract(url, filename):
    # Path to the directory where the file will be extracted
    extract_dir = os.path.join(project_dir, "data/original")

    # Create the download directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)

    file_path = os.path.join(extract_dir, filename)
    gdown.download(url, file_path, quiet=False)

    # Extract the file
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    # Clean up the downloaded zip file
    os.remove(file_path)


# URL of the file to download
orig_url = "https://drive.google.com/uc?id=1FIBnmdQSVUK4xi5uFpzb_vFseK_KLQUG"

download_and_extract(orig_url, "original.zip")

print("Files downloaded and extracted successfully!")
