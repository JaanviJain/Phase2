"""
Loads abstracts for FAISS indexing.
Primary source: SciFact corpus.jsonl
Optional: Additional PubMed abstracts from data/pubmed_abstracts/
"""

import os
import json
from config import SCIFACT_CORPUS, DATA_DIR


def load_all_abstracts():
    """
    Load all abstracts from available sources.
    Returns list of dicts: [{'id': ..., 'title': ..., 'text': ..., 'source': ...}]
    """
    abstracts = []
    
    # 1. Load SciFact corpus (primary)
    if os.path.exists(SCIFACT_CORPUS):
        print(f"Loading SciFact corpus from {SCIFACT_CORPUS}...")
        with open(SCIFACT_CORPUS, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line.strip())
                title = doc.get('title', '')
                abstract_text = doc.get('abstract', '')
                full_text = f"{title} {abstract_text}".strip()
                
                abstracts.append({
                    'id': str(doc['doc_id']),
                    'title': title,
                    'text': full_text,
                    'source': 'scifact'
                })
        print(f"  Loaded {len(abstracts)} abstracts from SciFact.")
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
                    doc = json.loads(line.strip())
                    text = doc.get('text', doc.get('abstract', '')).strip()
                    if text:
                        abstracts.append({
                            'id': doc.get('pmid', doc.get('id', f"{fname}_{count}")),
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
    abs_list = load_all_abstracts()
    print(f"\nSample abstract: {abs_list[0]['id']} | {abs_list[0]['title'][:60]}...")
