import os
import zipfile
import requests
from tqdm import tqdm
import urllib3
import ssl

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_icdar2015():
    url = 'https://datasets.cvc.uab.es/text/icdar2015/ch4_training_images.zip'
    url_gt = 'https://datasets.cvc.uab.es/text/icdar2015/ch4_training_localization_transcription_gt.zip'

    os.makedirs('icdar2015', exist_ok=True)

    def download(url, filename):
        # Create a session with SSL verification disabled
        session = requests.Session()
        session.verify = False
        
        try:
            with session.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                
                with open(filename, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=f'Downloading {os.path.basename(filename)}') as pbar:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                print(f"Successfully downloaded {filename}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename}: {e}")
            return False
        return True

    # Download images
    if not os.path.exists('icdar2015/images.zip'):
        print("Downloading training images...")
        if not download(url, 'icdar2015/images.zip'):
            return
    else:
        print("Training images already exist, skipping download.")

    # Download ground truth
    if not os.path.exists('icdar2015/gt.zip'):
        print("Downloading ground truth data...")
        if not download(url_gt, 'icdar2015/gt.zip'):
            return
    else:
        print("Ground truth data already exists, skipping download.")

    # Extract files
    print("Extracting files...")
    try:
        if os.path.exists('icdar2015/images.zip'):
            with zipfile.ZipFile('icdar2015/images.zip', 'r') as zip_ref:
                zip_ref.extractall('icdar2015/images')
            print("Images extracted successfully.")

        if os.path.exists('icdar2015/gt.zip'):
            with zipfile.ZipFile('icdar2015/gt.zip', 'r') as zip_ref:
                zip_ref.extractall('icdar2015/gt')
            print("Ground truth data extracted successfully.")
            
        print("Dataset download and extraction completed!")
        
    except zipfile.BadZipFile as e:
        print(f"Error extracting files: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    download_icdar2015()
