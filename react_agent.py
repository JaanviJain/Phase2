"""
================================================================================
PHASE 2 REACT AGENT: Local FAISS ONLY. No live API during evaluation.
OLLAMA VERSION — Singleton retriever, deduplication, robust parsing.
================================================================================
"""

import os
import re
import json
import time
import requests
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from config import (
    BASE_DIR, OLLAMA_MODEL, OLLAMA_URL, OLLAMA_TAGS_URL,
    TRACES_DIR, MAX_REACT_HOPS, PHASE2_OUTPUT_PATH
)
from retriever import BioBERTRetriever


# ==============================================================================
# AGENT STATE
# ==============================================================================
@dataclass
class AgentState:
    claim: str
    claim_id: str
    true_label: int = None
    scratchpad: List[str] = field(default_factory=list)
    evidence_collected: List[dict] = field(default_factory=list)
    hop_count: int = 0
    max_hops: int = MAX_REACT_HOPS


# ==============================================================================
# OLLAMA LLM
# ==============================================================================
class OllamaLLM:
    def __init__(self, model_name: str = OLLAMA_MODEL):
        self.model_name = model_name
        self.url = OLLAMA_URL
        self.tags_url = OLLAMA_TAGS_URL
        
        print(f"Connecting to Ollama ({model_name})...")
        try:
            r = requests.get(self.tags_url, timeout=5)
            if r.status_code == 200:
                models = [m['name'] for m in r.json().get('models', [])]
                print(f"  Ollama running. Available: {models}")
                if not any(model_name in m for m in models):
                    print(f"  WARNING: {model_name} not found. Run: ollama pull {model_name}")
            else:
                print(f"  WARNING: Ollama status {r.status_code}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Ollama not running at localhost:11434. Run 'ollama serve'.")
        except Exception as e:
            print(f"  WARNING: Could not verify Ollama: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 400, temperature: float = 0.1) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "stop": ["Observation:", "Continue from here."]
            }
        }
        try:
            r = requests.post(self.url, json=payload, timeout=120)
            r.raise_for_status()
            return r.json().get("response", "")
        except requests.exceptions.Timeout:
            print("  ERROR: Ollama timeout.")
            return "Thought: Timeout occurred.\nAction: stop('Timeout')"
        except Exception as e:
            print(f"  ERROR: Ollama failed: {e}")
            return f"Thought: Error occurred.\nAction: stop('Error: {e}')"


# ==============================================================================
# REACT AGENT
# ==============================================================================
class ReActAgent:
    def __init__(self, retriever: BioBERTRetriever = None):
        """
        Args:
            retriever: Pre-initialized BioBERTRetriever (singleton).
                      If None, creates one (but you should pass one for efficiency).
        """
        self.llm = OllamaLLM()
        self.retriever = retriever if retriever is not None else BioBERTRetriever()
        
        with open("react_system_prompt.txt", "r", encoding="utf-8") as f:
            self.base_prompt = f.read()
    
    def parse_action(self, text: str) -> Optional[Dict]:
        """
        Robust parsing of Thought/Action from LLM output.
        Returns: {"thought": "...", "tool": "search_pubmed"|"stop", "argument": "..."}
        """
        # Extract thought
        thought_match = re.search(
            r'Thought:\s*(.*?)(?=\n\s*Action:|$)', 
            text, 
            re.DOTALL | re.IGNORECASE
        )
        thought = thought_match.group(1).strip() if thought_match else "No thought parsed."
        
        # Extract action
        action_match = re.search(
            r'Action:\s*(.*?)(?=\n|$)', 
            text, 
            re.DOTALL | re.IGNORECASE
        )
        action_str = action_match.group(1).strip() if action_match else "STOP"
        
        # Parse tool and argument
        tool = "stop"
        argument = action_str
        
        # Match search_pubmed("...") with various quote styles
        pm_match = re.search(
            r'search_pubmed\(\s*["\']?(.*?)["\']?\s*\)', 
            action_str, 
            re.IGNORECASE
        )
        if pm_match:
            tool = "search_pubmed"
            argument = pm_match.group(1).strip()
        elif "stop" in action_str.lower():
            tool = "stop"
            # Extract reason if in parentheses
            reason_match = re.search(r'stop\(\s*["\']?(.*?)["\']?\s*\)', action_str, re.IGNORECASE)
            argument = reason_match.group(1).strip() if reason_match else action_str
        
        return {
            "thought": thought,
            "tool": tool,
            "argument": argument
        }
    
    def deduplicate_evidence(self, evidence_list: List[Dict]) -> List[Dict]:
        """Remove duplicate evidence by ID."""
        seen_ids = set()
        unique = []
        for ev in evidence_list:
            eid = ev.get('id', ev.get('pmid', ''))
            if eid and eid not in seen_ids:
                seen_ids.add(eid)
                unique.append(ev)
        return unique
    
    def search_local(self, query: str, k: int = 3) -> List[Dict]:
        """Search local FAISS only. NO live API."""
        return self.retriever.search(query, k=k)
    
    def investigate_claim(self, claim: str, claim_id: str, true_label: int = None) -> AgentState:
        """
        Run ReAct loop for one claim.
        Returns AgentState with collected evidence and trace.
        """
        state = AgentState(
            claim=claim, 
            claim_id=claim_id, 
            true_label=true_label, 
            max_hops=MAX_REACT_HOPS
        )
        
        prompt = self.base_prompt.format(claim=claim, max_hops=MAX_REACT_HOPS)
        conversation = prompt + "\n\n"
        
        print(f"\n{'='*70}")
        print(f"CLAIM: {claim[:80]}...")
        print(f"{'='*70}")
        
        while state.hop_count < state.max_hops:
            print(f"\n--- Hop {state.hop_count + 1}/{state.max_hops} ---")
            
            # Build scratchpad context
            scratchpad_text = "\n\n".join(state.scratchpad) if state.scratchpad else "No previous actions."
            full_prompt = conversation + f"\nHere is what has happened so far:\n{scratchpad_text}\n\nContinue from here."
            
            response = self.llm.generate(full_prompt, max_tokens=400, temperature=0.1)
            print(f"LLM: {response[:300]}...")
            
            parsed = self.parse_action(response)
            print(f"Parsed -> Thought: {parsed['thought'][:80]}...")
            print(f"Parsed -> Action: {parsed['tool']}('{parsed['argument'][:60]}...')")
            
            state.scratchpad.append(f"Thought: {parsed['thought']}\nAction: {parsed['tool']}('{parsed['argument']}')")
            
            if parsed['tool'] == 'stop':
                print("Agent stopped.")
                break
            
            if parsed['tool'] == 'search_pubmed':
                results = self.search_local(parsed['argument'], k=3)
                
                # Format observation for LLM
                obs_lines = []
                for i, res in enumerate(results[:3], 1):
                    obs_lines.append(f"Result {i} [{res['source']}] (ID: {res['id']}): {res['title']}")
                    obs_lines.append(f"  {res['text'][:250]}...")
                
                observation = "\n".join(obs_lines) if obs_lines else "No results found."
                
                # Collect evidence
                for res in results:
                    state.evidence_collected.append({
                        'id': res['id'],
                        'title': res['title'],
                        'text': res['text'],
                        'score': res['score'],
                        'source': res['source'],
                        'query': parsed['argument']
                    })
                
                state.scratchpad.append(f"Observation: {observation}")
            else:
                print(f"Unknown tool '{parsed['tool']}'. Stopping.")
                break
            
            state.hop_count += 1
            time.sleep(0.2)
        
        # Deduplicate
        state.evidence_collected = self.deduplicate_evidence(state.evidence_collected)
        
        # Sort by relevance score
        state.evidence_collected.sort(key=lambda x: x['score'], reverse=True)
        
        return state
    
    def save_trace(self, state: AgentState, output_dir: str = None):
        """Save human-readable trace file."""
        if output_dir is None:
            output_dir = TRACES_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        safe_id = re.sub(r'[^\w]', '_', state.claim_id)[:50]
        filepath = os.path.join(output_dir, f"{safe_id}.txt")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Claim ID: {state.claim_id}\n")
            f.write(f"Claim: {state.claim}\n")
            if state.true_label is not None:
                label_names = {0: "REFUTED", 1: "NEI", 2: "SUPPORTED"}
                f.write(f"True Label: {label_names.get(state.true_label, 'UNKNOWN')}\n")
            f.write(f"Hops used: {state.hop_count}/{state.max_hops}\n")
            f.write(f"Evidence collected: {len(state.evidence_collected)}\n")
            f.write("="*70 + "\n\n")
            for entry in state.scratchpad:
                f.write(entry + "\n\n")
        
        print(f"Trace saved: {filepath}")


# ==============================================================================
# STANDALONE: Save Phase 2 output in correct structure
# ==============================================================================
def save_phase2_output(all_states: List[AgentState], output_path: str = None):
    """
    Save structured output for Phase 3.
    CRITICAL FIX: Explicit claim association with metadata header.
    """
    if output_path is None:
        output_path = PHASE2_OUTPUT_PATH
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    label_map_reverse = {0: "REFUTED", 1: "NEI", 2: "SUPPORTED"}
    
    claims_output = []
    for state in all_states:
        evidence_list = []
        for ev in state.evidence_collected:
            evidence_list.append({
                "id": ev['id'],
                "title": ev.get('title', ''),
                "text": ev.get('text', '')[:1000],
                "score": float(ev.get('score', 0.0)),
                "source": ev.get('source', 'unknown'),
                "retrieval_query": ev.get('query', '')
            })
        
        claims_output.append({
            "claim_id": state.claim_id,
            "claim": state.claim,
            "true_label": state.true_label,
            "true_label_name": label_map_reverse.get(state.true_label, "UNKNOWN"),
            "num_evidence": len(evidence_list),
            "evidence": evidence_list,
            "agent_metadata": {
                "num_hops": state.hop_count,
                "max_hops": state.max_hops
            }
        })
    
    final_output = {
        "metadata": {
            "total_claims": len(claims_output),
            "total_evidence": sum(c["num_evidence"] for c in claims_output),
            "model": OLLAMA_MODEL,
            "corpus": "scifact_local_faiss",
            "max_hops": MAX_REACT_HOPS
        },
        "claims": claims_output
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("PHASE 2 OUTPUT SAVED")
    print(f"{'='*70}")
    print(f"File: {output_path}")
    print(f"Claims: {final_output['metadata']['total_claims']}")
    print(f"Total evidence: {final_output['metadata']['total_evidence']}")
    
    # BUG FIX VERIFICATION
    if final_output['metadata']['total_claims'] == 0:
        print("🚨 CRITICAL: Output contains 0 claims. DO NOT proceed to Phase 3.")
    else:
        print("✅ Output structure verified. Safe for Phase 3.")
    print(f"{'='*70}")
    
    return output_path
