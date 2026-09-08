"""
Evaluate ReAct agent vs direct retrieval baseline.
Measures: gold hit rate, avg hops, evidence quality.
"""

import json
import argparse
from collections import defaultdict

from react_agent import ReActAgent
from retriever import BioBERTRetriever
from config import SCIFACT_CLAIMS_DEV, SCIFACT_CLAIMS_TEST, REACT_EVAL_PATH


def load_scifact_claims(path):
    claims = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            gold_docs = set()
            if "evidence" in item and item["evidence"]:
                gold_docs.update(str(k) for k in item["evidence"].keys())
            if "cited_doc_ids" in item and item["cited_doc_ids"]:
                gold_docs.update(str(k) for k in item["cited_doc_ids"])
            claims.append({
                "id": str(item["id"]),
                "claim": item["claim"],
                "gold_docs": gold_docs,
                "label": item.get("label", "NEI")
            })
    return claims


def evaluate_react(claims, max_claims=None):
    # Singleton retriever shared across all evaluations
    retriever = BioBERTRetriever()
    agent = ReActAgent(retriever=retriever)
    baseline = BioBERTRetriever()  # Same singleton instance returned
    
    if max_claims:
        claims = claims[:max_claims]
    
    stats = {
        "total_claims": len(claims),
        "avg_hops": 0.0,
        "evidence_found_rate": 0.0,
        "agent_gold_hit_rate": 0.0,
        "baseline_gold_hit_rate": 0.0,
        "avg_evidence_per_claim": 0.0,
        "stop_reasons": defaultdict(int)
    }
    
    detailed = []
    
    for i, claim in enumerate(claims):
        print(f"\n[{i+1}/{len(claims)}] {claim['claim'][:70]}...")
        
        # ReAct agent
        state = agent.investigate_claim(claim["claim"], claim_id=claim["id"], true_label=None)
        
        # Baseline direct search
        baseline_results = baseline.search(claim["claim"], k=10)
        
        # Metrics
        agent_ids = [e["id"] for e in state.evidence_collected]
        baseline_ids = [r["id"] for r in baseline_results]
        
        agent_hit = bool(set(agent_ids) & claim["gold_docs"])
        baseline_hit = bool(set(baseline_ids) & claim["gold_docs"])
        
        if state.evidence_collected:
            stats["evidence_found_rate"] += 1
        if agent_hit:
            stats["agent_gold_hit_rate"] += 1
        if baseline_hit:
            stats["baseline_gold_hit_rate"] += 1
        
        stats["avg_hops"] += state.hop_count
        stats["avg_evidence_per_claim"] += len(state.evidence_collected)
        
        # Track stop reason from last scratchpad entry
        if state.scratchpad:
            last = state.scratchpad[-1]
            reason = last.split("Action: stop(")[-1].rstrip(")") if "stop(" in last else "max_hops"
            stats["stop_reasons"][reason[:80]] += 1
        
        detailed.append({
            "claim_id": claim["id"],
            "agent_hit": agent_hit,
            "baseline_hit": baseline_hit,
            "num_hops": state.hop_count,
            "num_evidence": len(state.evidence_collected),
            "gold_docs": list(claim["gold_docs"]),
            "agent_retrieved": agent_ids,
            "baseline_retrieved": baseline_ids
        })
    
    # Normalize
    n = len(claims)
    for key in ["evidence_found_rate", "agent_gold_hit_rate", "baseline_gold_hit_rate"]:
        stats[key] /= n
    stats["avg_hops"] /= n
    stats["avg_evidence_per_claim"] /= n
    stats["stop_reasons"] = dict(stats["stop_reasons"])
    
    print(f"\n{'='*60}")
    print("REACT AGENT EVALUATION")
    print(f"{'='*60}")
    print(f"Claims: {n}")
    print(f"Avg hops: {stats['avg_hops']:.2f}")
    print(f"Evidence found: {stats['evidence_found_rate']:.3f}")
    print(f"Gold hit (Agent): {stats['agent_gold_hit_rate']:.3f}")
    print(f"Gold hit (Baseline): {stats['baseline_gold_hit_rate']:.3f}")
    print(f"Avg evidence/claim: {stats['avg_evidence_per_claim']:.2f}")
    
    output = {"summary": stats, "detailed": detailed}
    with open(REACT_EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Saved to {REACT_EVAL_PATH}")
    
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--max_claims", type=int, default=None)
    args = parser.parse_args()
    
    claims_path = SCIFACT_CLAIMS_DEV if args.split == "dev" else SCIFACT_CLAIMS_TEST
    claims = load_scifact_claims(claims_path)
    evaluate_react(claims, max_claims=args.max_claims)


if __name__ == "__main__":
    main()
