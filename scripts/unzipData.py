import zipfile
import os
from tqdm import tqdm

def extract_with_progress(zip_path, extract_path):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            total_size = sum(file.file_size for file in zip_ref.infolist())
            extracted_size = 0
            
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Extracting") as pbar:
                for file in zip_ref.infolist():
                    zip_ref.extract(file, extract_path)
                    extracted_size += file.file_size
                    pbar.update(file.file_size)
        
        print("Extraction complete.")
    except zipfile.BadZipFile:
        print("The ZIP file is invalid or corrupted.")
    except NotImplementedError:
        print("The ZIP uses an unsupported compression method.")

zip_path = r"C:\Users\Ben Gur\src\src\wubai\data\fma_large.zip"
extract_path = r"C:\Users\Ben Gur\src\src\wubai\data"
extract_with_progress(zip_path, extract_path)
