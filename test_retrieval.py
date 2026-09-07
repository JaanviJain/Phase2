"""
Test retrieval without LLM (faster, for debugging).
Uses transformers AutoModel directly (NOT sentence-transformers).
"""

import faiss
import torch
from build_faiss_index import load_faiss_index, BioBERTEncoder
from config import *


def test_basic_retrieval():
    print("=" * 70)
    print("TESTING BASIC RETRIEVAL (NO LLM)")
    print("=" * 70)
    
    index, metadata = load_faiss_index()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = BioBERTEncoder(BIOMODEL, device=device)
    
    queries = [
        "Aspirin reduces heart attack risk",
        "Vaccines cause autism",
        "Vitamin C prevents common cold",
        "Statins cause memory loss",
        "Exercise reduces depression"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        query_emb = encoder.encode([query])
        faiss.normalize_L2(query_emb)
        distances, indices = index.search(query_emb, k=3)
        
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            abs_dict = metadata['abstracts'][idx]
            print(f"  {i+1}. [{abs_dict['source']}] Score: {dist:.4f}")
            print(f"      {abs_dict['title'][:70]}...")
