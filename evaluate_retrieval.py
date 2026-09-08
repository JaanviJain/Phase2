"""
Evaluate pure retrieval (BioBERT-FAISS and BM25) against SciFact gold evidence.
Produces: Recall@1, Recall@5, Recall@10, MRR
THIS IS YOUR SCIENTIFIC BASELINE.
"""

import json
import argparse
from collections import defaultdict

from retriever import BioBERTRetriever, BM25Retriever
from config import SCIFACT_CLAIMS_DEV, SCIFACT_CLAIMS_TEST, RETRIEVAL_EVAL_PATH


def load_scifact_claims(path):
    """Load SciFact claims and extract gold doc_ids."""
    claims = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            gold_docs = set()
            
            # SciFact evidence format: {doc_id: [sentences]}
            if "evidence" in item and item["evidence"]:
                gold_docs.update(str(k) for k in item["evidence"].keys())
            
            # Fallback
            if "cited_doc_ids" in item and item["cited_doc_ids"]:
                gold_docs.update(str(k) for k in item["cited_doc_ids"])
            
            claims.append({
                "id": str(item["id"]),
                "claim": item["claim"],
                "gold_docs": gold_docs,
                "label": item.get("label", "NEI")
            })
    return claims


def evaluate_retriever(retriever, claims, k_values=[1, 5, 10]):
    """
    Compute Recall@K and MRR.
    """
    metrics = {f"recall@{k}": 0.0 for k in k_values}
    rr_sum = 0.0
    total = len(claims)
    
    for claim in claims:
        results = retriever.search(claim["claim"], k=max(k_values))
        retrieved_ids = [r["id"] for r in results]
        
        # Find rank of first relevant document
        first_hit = None
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in claim["gold_docs"]:
                first_hit = rank
                break
        
        # MRR
        rr_sum += 1.0 / first_hit if first_hit else 0.0
        
        # Recall@K
        retrieved_set = set(retrieved_ids)
        for k in k_values:
            if retrieved_set & claim["gold_docs"]:
                metrics[f"recall@{k}"] += 1.0
    
    # Normalize
    for k in k_values:
        metrics[f"recall@{k}"] /= total
    metrics["mrr"] = rr_sum / total
    metrics["num_claims"] = total
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    args = parser.parse_args()
    
    claims_path = SCIFACT_CLAIMS_DEV if args.split == "dev" else SCIFACT_CLAIMS_TEST
    print(f"Evaluating retrieval on {claims_path}...")
    
    claims = load_scifact_claims(claims_path)
    print(f"Loaded {len(claims)} claims with gold evidence.")
    
    # BioBERT-FAISS
    print("\n=== BioBERT + FAISS ===")
    biobert = BioBERTRetriever()
    biobert_metrics = evaluate_retriever(biobert, claims)
    print(f"Recall@1:  {biobert_metrics['recall@1']:.3f}")
    print(f"Recall@5:  {biobert_metrics['recall@5']:.3f}")
    print(f"Recall@10: {biobert_metrics['recall@10']:.3f}")
    print(f"MRR:       {biobert_metrics['mrr']:.3f}")
    
    # BM25
    print("\n=== BM25 ===")
    bm25 = BM25Retriever()
    bm25_metrics = evaluate_retriever(bm25, claims)
    print(f"Recall@1:  {bm25_metrics['recall@1']:.3f}")
    print(f"Recall@5:  {bm25_metrics['recall@5']:.3f}")
    print(f"Recall@10: {bm25_metrics['recall@10']:.3f}")
    print(f"MRR:       {bm25_metrics['mrr']:.3f}")
    
    # Save
    results = {
        "split": args.split,
        "num_claims": len(claims),
        "biobert_faiss": biobert_metrics,
        "bm25": bm25_metrics
    }
    
    with open(RETRIEVAL_EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Saved to {RETRIEVAL_EVAL_PATH}")


if __name__ == "__main__":
    main()
