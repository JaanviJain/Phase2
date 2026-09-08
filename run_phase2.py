"""
MAIN PHASE 2 PIPELINE
This is the ONLY script you run to generate Phase 2 output for Phase 3.
"""

import json
import argparse

from react_agent import ReActAgent, save_phase2_output
from retriever import BioBERTRetriever
from config import SCIFACT_CLAIMS_DEV, SCIFACT_CLAIMS_TEST, MAX_REACT_HOPS


def load_scifact_claims(path, max_claims=None):
    claims = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_claims and i >= max_claims:
                break
            doc = json.loads(line.strip())
            
            # Label mapping
            label_map = {"REFUTES": 0, "NOT ENOUGH INFO": 1, "SUPPORTS": 2}
            label = label_map.get(doc.get("label"), 1)
            
            claims.append({
                "claim_id": f"scifact_{doc['id']}",
                "claim": doc["claim"],
                "label": label
            })
    return claims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test", "train"], default="dev")
    parser.add_argument("--max_claims", type=int, default=None,
                        help="Limit claims for quick testing")
    args = parser.parse_args()
    
    # Select claims file
    if args.split == "dev":
        claims_path = SCIFACT_CLAIMS_DEV
    elif args.split == "test":
        claims_path = SCIFACT_CLAIMS_TEST
    else:
        claims_path = SCIFACT_CLAIMS_DEV
    
    print(f"Loading claims from {claims_path}...")
    claims = load_scifact_claims(claims_path, max_claims=args.max_claims)
    print(f"Total claims to process: {len(claims)}")
    print(f"Max ReAct hops: {MAX_REACT_HOPS}")
    print("Make sure Ollama is running: ollama serve")
    
    # CRITICAL: Initialize ONE retriever and share it with the agent
    # This prevents reloading BioBERT 200+ times
    print("\nInitializing retriever (this loads BioBERT once)...")
    retriever = BioBERTRetriever()
    
    print("Initializing ReAct agent...")
    agent = ReActAgent(retriever=retriever)
    
    all_states = []
    
    for i, claim in enumerate(claims):
        print(f"\n{'='*70}")
        print(f"CLAIM {i+1}/{len(claims)} | ID: {claim['claim_id']}")
        print(f"TEXT: {claim['claim'][:100]}...")
        print(f"{'='*70}")
        
        state = agent.investigate_claim(
            claim=claim["claim"],
            claim_id=claim["claim_id"],
            true_label=claim["label"]
        )
        
        all_states.append(state)
        print(f"✅ Retrieved {len(state.evidence_collected)} unique evidence items.")
        
        # Save trace for debugging
        agent.save_trace(state)
    
    # Save final output
    save_phase2_output(all_states)
    
    print(f"\n{'='*70}")
    print("PHASE 2 PIPELINE COMPLETE")
    print(f"{'='*70}")
    print("Next steps:")
    print("  1. Check data/phase2_evidence/retrieved_evidence.json")
    print("  2. Run: python evaluate_retrieval.py --split dev")
    print("  3. Run: python evaluate_react.py --split dev --max_claims 50")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
