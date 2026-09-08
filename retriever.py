"""
Pure retrieval classes. No LLM. No ReAct.
These are your scientific baselines.
"""

import os
import numpy as np
import faiss
import torch
from rank_bm25 import BM25Okapi

from config import BIOMODEL, FAISS_INDEX_PATH, FAISS_META_PATH, TOP_K_RETRIEVE
from build_faiss_index import BioBERTEncoder


class BioBERTRetriever:
    """
    Dense retriever using BioBERT + FAISS.
    Loads model ONCE and reuses it.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        # Singleton pattern: never load the model twice
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, device=None):
        if self._initialized:
            return
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print("Initializing BioBERT Retriever (singleton)...")
        self.encoder = BioBERTEncoder(BIOMODEL, device=device)
        self.index, self.metadata = self._load_faiss()
        self._initialized = True
        print("Retriever ready.")
    
    def _load_faiss(self):
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}")
        
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_META_PATH, 'rb') as f:
            metadata = pickle.load(f)
        return index, metadata
    
    def search(self, query: str, k: int = TOP_K_RETRIEVE):
        """
        Search local FAISS index.
        Returns list of dicts with id, title, text, score.
        """
        q_emb = self.encoder.encode([query])
        distances, indices = self.index.search(q_emb, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata['abstracts']):
                continue
            doc = self.metadata['abstracts'][idx]
            results.append({
                'id': doc['id'],
                'title': doc['title'],
                'text': doc['text'],
                'source': doc.get('source', 'unknown'),
                'score': float(dist)
            })
        return results


class BM25Retriever:
    """
    BM25 baseline for comparison.
    """
    
    def __init__(self):
        from build_faiss_index import load_faiss_index
        _, metadata = load_faiss_index()
        self.docs = metadata['abstracts']
        self.corpus_texts = [d['text'] for d in self.docs]
        self.tokenized_corpus = [doc.lower().split() for doc in self.corpus_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"BM25 retriever ready with {len(self.docs)} documents.")
    
    def search(self, query: str, k: int = TOP_K_RETRIEVE):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            doc = self.docs[idx]
            results.append({
                'id': doc['id'],
                'title': doc['title'],
                'text': doc['text'],
                'source': doc.get('source', 'unknown'),
                'score': float(scores[idx])
            })
        return results
