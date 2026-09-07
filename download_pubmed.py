"""
Download SciFact corpus + claims.
Uses the official S3 tarball (GitHub raw links are dead).
"""

import os
import json
import requests
import tarfile
import shutil
from tqdm import tqdm
from config import *

def download_file(url, filepath, desc=None):
    """Download a file with progress bar."""
    if os.path.exists(filepath):
        print(f"Already exists: {os.path.basename(filepath)}")
        return True
    
    d = desc or f"Downloading {os.path.basename(filepath)}"
    print(d)
    try:
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print(f"  Saved: {filepath} ({os.path.getsize(filepath)} bytes)")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def download_scifact():
    """
    Download SciFact dataset from official S3 tarball.
    GitHub raw links are 404 — AllenAI hosts data on S3.
    """
    print("=" * 70)
    print("DOWNLOADING SCIFACT DATASET")
    print("=" * 70)
    
    # Official S3 URL from AllenAI
    data_url = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
    tarball_path = os.path.join(DATA_DIR, "scifact_data.tar.gz")
    
    # If corpus already exists, skip everything
    if os.path.exists(SCIFACT_CORPUS):
        print("SciFact corpus already exists. Skipping download.")
        return verify_scifact_files()
    
    # Download tarball
    if not os.path.exists(tarball_path):
        success = download_file(data_url, tarball_path, "Downloading SciFact data.tar.gz from S3...")
        if not success:
            print("S3 download failed. Trying alternative sources...")
            return download_scifact_alternative()
    
    # Extract tarball
    print(f"\nExtracting {tarball_path}...")
    try:
        with tarfile.open(tarball_path, 'r:gz') as tar:
            tar.extractall(DATA_DIR)
        print("Extraction complete.")
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False
    
    # Move files from extracted 'data/' folder to 'data/scifact/'
    extracted_dir = os.path.join(DATA_DIR, "data")
    scifact_dir = os.path.join(DATA_DIR, "scifact")
    
    if os.path.exists(extracted_dir):
        os.makedirs(scifact_dir, exist_ok=True)
        for filename in ["corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl"]:
            src = os.path.join(extracted_dir, filename)
            dst = os.path.join(scifact_dir, filename)
            if os.path.exists(src):
                shutil.move(src, dst)
                print(f"  Moved: {filename}")
        
        # Clean up
        shutil.rmtree(extracted_dir)
        os.remove(tarball_path)
        print("Cleaned up temporary files.")
    
    return verify_scifact_files()

def download_scifact_alternative():
    """
    Fallback: try to download individual files from known working URLs.
    """
    print("=" * 70)
    print("ALTERNATIVE DOWNLOAD: Individual files")
    print("=" * 70)
    
    # Try S3 direct links for individual files
    base_urls = [
        "https://scifact.s3-us-west-2.amazonaws.com/release/latest",
    ]
    
    files = ["corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl"]
    scifact_dir = os.path.join(DATA_DIR, "scifact")
    os.makedirs(scifact_dir, exist_ok=True)
    
    all_ok = True
    for filename in files:
        filepath = os.path.join(scifact_dir, filename)
        if os.path.exists(filepath):
            print(f"Already exists: {filename}")
            continue
        
        downloaded = False
        for base_url in base_urls:
            url = f"{base_url}/{filename}"
            if download_file(url, filepath):
                downloaded = True
                break
        
        if not downloaded:
            print(f"  Could not download {filename}")
            all_ok = False
    
    return verify_scifact_files() if all_ok else False

def verify_scifact_files():
    """Check that all required files exist and print counts."""
    files = ["corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl"]
    scifact_dir = os.path.join(DATA_DIR, "scifact")
    
    print("\nSciFact dataset status:")
    all_ok = True
    for filename in files:
        filepath = os.path.join(scifact_dir, filename)
        if os.path.exists(filepath):
            count = sum(1 for _ in open(filepath, 'r', encoding='utf-8'))
            print(f"  {filename}: {count} lines  ✓")
        else:
            print(f"  {filename}: MISSING  ✗")
            all_ok = False
    
    return all_ok

def load_scifact_abstracts():
    """Load SciFact abstracts into list of dicts."""
    if not os.path.exists(SCIFACT_CORPUS):
        print(f"ERROR: {SCIFACT_CORPUS} not found. Run download first.")
        return []
    
    abstracts = []
    with open(SCIFACT_CORPUS, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            text = doc.get('title', '') + ' ' + ' '.join(doc.get('abstract', []))
            abstracts.append({
                'id': f"scifact_{doc['doc_id']}",
                'title': doc.get('title', ''),
                'text': text.strip(),
                'source': 'scifact'
            })
    return abstracts

def load_all_abstracts():
    """Load SciFact + PubMed abstracts."""
    all_abstracts = []
    
    if os.path.exists(SCIFACT_CORPUS):
        scifact_abs = load_scifact_abstracts()
        all_abstracts.extend(scifact_abs)
        print(f"Loaded {len(scifact_abs)} SciFact abstracts")
    else:
        print("WARNING: SciFact corpus not found.")
    
    pubmed_file = os.path.join(DATA_DIR, "pubmed_abstracts", "pubmed_sample.jsonl")
    if os.path.exists(pubmed_file):
        with open(pubmed_file, 'r', encoding='utf-8') as f:
            for line in f:
                all_abstracts.append(json.loads(line))
        print(f"Loaded PubMed abstracts")
    
    print(f"Total abstracts: {len(all_abstracts)}")
    return all_abstracts

if __name__ == "__main__":
    success = download_scifact()
    
    if success:
        abs_list = load_all_abstracts()
        if len(abs_list) > 0:
            print(f"\nSample abstract:")
            print(f"ID: {abs_list[0]['id']}")
            print(f"Title: {abs_list[0]['title'][:100]}...")
            print(f"Text length: {len(abs_list[0]['text'])} chars")
        else:
            print("\nNo abstracts loaded.")
    else:
        print("\nDownload failed. Please check your internet connection.")