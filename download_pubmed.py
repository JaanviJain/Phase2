"""
Download SciFact corpus + claims, and load abstracts for FAISS indexing.
Uses the official S3 tarball (GitHub raw links are dead).
Primary source: SciFact corpus.jsonl
Optional: Additional PubMed abstracts from data/pubmed_abstracts/
"""

import os
import json
import requests
import tarfile
import shutil
from tqdm import tqdm

# Import paths from your config
try:
    from config import DATA_DIR, SCIFACT_CORPUS
except ImportError:
    # Fallback if running directly without config in path
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from config import DATA_DIR, SCIFACT_CORPUS


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
        if os.path.exists(tarball_path):
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
    
    base_urls = ["https://scifact.s3-us-west-2.amazonaws.com/release/latest"]
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
    """
    Load SciFact abstracts into list of dicts.
    Handles both string and list-of-sentences formats for the 'abstract' field.
    """
    if not os.path.exists(SCIFACT_CORPUS):
        print(f"ERROR: {SCIFACT_CORPUS} not found. Run download first.")
        return []
    
    abstracts = []
    with open(SCIFACT_CORPUS, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            title = doc.get('title', '')
            abstract_field = doc.get('abstract', [])
            
            # SciFact abstracts are often lists of sentences. Join them safely.
            if isinstance(abstract_field, list):
                abstract_text = ' '.join(abstract_field)
            else:
                abstract_text = str(abstract_field)
                
            full_text = f"{title} {abstract_text}".strip()
            
            abstracts.append({
                'id': str(doc['doc_id']),
                'title': title,
                'text': full_text,
                'source': 'scifact'
            })
    return abstracts


def load_all_abstracts():
    """
    Load all abstracts from available sources for FAISS indexing.
    Returns list of dicts: [{'id': ..., 'title': ..., 'text': ..., 'source': ...}]
    """
    abstracts = []
    
    # 1. Load SciFact corpus (primary)
    if os.path.exists(SCIFACT_CORPUS):
        print(f"Loading SciFact corpus from {SCIFACT_CORPUS}...")
        scifact_abs = load_scifact_abstracts()
        abstracts.extend(scifact_abs)
        print(f"  Loaded {len(scifact_abs)} abstracts from SciFact.")
    else:
        print(f"  WARNING: {SCIFACT_CORPUS} not found.")
    
    # 2. Optional: Load additional PubMed abstracts
    pubmed_dir = os.path.join(DATA_DIR, "pubmed_abstracts")
    if os.path.exists(pubmed_dir):
        extra_files = [f for f in os.listdir(pubmed_dir) if f.endswith('.jsonl')]
        for fname in extra_files:
            fpath = os.path.join(pubmed_dir, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                count = 0
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    doc = json.loads(line)
                    text = doc.get('text', doc.get('abstract', '')).strip()
                    
                    # Handle list-of-sentences just in case
                    if isinstance(text, list):
                        text = ' '.join(text)
                        
                    if text:
                        abstracts.append({
                            'id': str(doc.get('pmid', doc.get('id', doc.get('doc_id', f"{fname}_{count}")))),
                            'title': doc.get('title', ''),
                            'text': text,
                            'source': 'pubmed'
                        })
                        count += 1
                print(f"  Loaded {count} abstracts from {fname}.")
    
    if not abstracts:
        raise FileNotFoundError(
            "No abstracts found. Place corpus.jsonl in data/scifact/ "
            "or PubMed JSONL files in data/pubmed_abstracts/"
        )
    
    print(f"Total abstracts loaded: {len(abstracts)}")
    return abstracts


if __name__ == "__main__":
    print("=" * 70)
    print("SCIFACT DOWNLOADER & LOADER")
    print("=" * 70)
    
    # Step 1: Ensure data is downloaded
    success = verify_scifact_files()
    if not success:
        print("\nFiles missing. Initiating download...")
        success = download_scifact()
    
    # Step 2: Load and verify
    if success:
        print("\n" + "=" * 70)
        print("LOADING ABSTRACTS FOR INDEXING")
        print("=" * 70)
        abs_list = load_all_abstracts()
        
        if len(abs_list) > 0:
            print(f"\n✅ SUCCESS! Sample abstract:")
            print(f"   ID: {abs_list[0]['id']}")
            print(f"   Source: {abs_list[0]['source']}")
            print(f"   Title: {abs_list[0]['title'][:80]}...")
            print(f"   Text length: {len(abs_list[0]['text'])} chars")
        else:
            print("\n❌ No abstracts loaded. Check your data directories.")
    else:
        print("\n❌ Download failed. Please check your internet connection.")
