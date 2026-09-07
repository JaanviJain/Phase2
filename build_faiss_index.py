"""
Build FAISS index from abstracts.
Uses transformers BertModel + BertTokenizer directly.
"""

import os
import pickle
import numpy as np
import faiss
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from config import *
from download_pubmed import load_all_abstracts


class BioBERTEncoder:
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1", device="cpu"):
        self.device = device
        print(f"Loading BioBERT: {model_name}")
        
        # EXPLICIT BertTokenizer — avoids AutoTokenizer fast/slow bug
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(device)
        self.model.eval()
        print(f"BioBERT loaded on {device}")
    
    def encode(self, texts, batch_size=32):
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i:i+batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Mean pooling
            token_embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings).astype("float32")


def build_faiss_index():
    print("=" * 70)
    print("BUILDING FAISS INDEX")
    print("=" * 70)
    
    abstracts = load_all_abstracts()
    if len(abstracts) == 0:
        raise ValueError("No abstracts found. Run download_pubmed.py first.")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"PyTorch device: {device}")
    
    encoder = BioBERTEncoder(BIOMODEL, device=device)
    
    texts = [abs_dict['text'] for abs_dict in abstracts]
    ids = [abs_dict['id'] for abs_dict in abstracts]
    metadata = {'ids': ids, 'abstracts': abstracts}
    
    print(f"Encoding {len(texts)} abstracts...")
    embeddings = encoder.encode(texts, batch_size=32)
    print(f"Embeddings shape: {embeddings.shape}")
    
    faiss.normalize_L2(embeddings)
    
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"Index size: {index.ntotal}")
    
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"Saved FAISS index: {FAISS_INDEX_PATH}")
    
    with open(FAISS_META_PATH, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata: {FAISS_META_PATH}")
    
    # Test search
    print("\nTesting index with sample query...")
    test_query = "Aspirin reduces risk of heart attack"
    test_emb = encoder.encode([test_query])
    faiss.normalize_L2(test_emb)
    distances, indices = index.search(test_emb, k=3)
    
    print(f"\nTop 3 results for: '{test_query}'")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        print(f"  {i+1}. [{ids[idx]}] Score: {dist:.4f}")
        print(f"      {abstracts[idx]['title'][:80]}...")
    
    return index, metadata


def load_faiss_index():
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"Index not found at {FAISS_INDEX_PATH}. Run build_faiss_index.py first.")
    
    print(f"Loading FAISS index from {FAISS_INDEX_PATH}")
    index = faiss.read_index(FAISS_INDEX_PATH)
    
    with open(FAISS_META_PATH, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Loaded index with {index.ntotal} vectors")
    return index, metadata


if __name__ == "__main__":
    build_faiss_index()

