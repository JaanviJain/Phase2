"""
================================================================================
PHASE 2 REACT AGENT: Local FAISS + Live PubMed API + Wikipedia Fallback
OLLAMA VERSION — No transformers LLM loading. No API keys. No GPU VRAM issues.
================================================================================
"""

import os
import re
import json
import time
import pickle
import requests
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

import numpy as np
import faiss
import torch
from transformers import BertTokenizer, BertModel

from config import (
    BASE_DIR, BIOMODEL, FAISS_INDEX_PATH, FAISS_META_PATH,
    OLLAMA_MODEL, OLLAMA_URL, OLLAMA_TAGS_URL, TRACES_DIR
)

# ==============================================================================
# AGENT STATE
# ==============================================================================
@dataclass
class AgentState:
    claim: str
    scratchpad: List[str] = field(default_factory=list)
    evidence_collected: List[dict] = field(default_factory=list)
    hop_count: int = 0
    max_hops: int = 3
    true_label: int = None

# ==============================================================================
# OLLAMA LLM — REPLACES THE OLD TRANSFORMERS LLMInterface
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
                print(f"  Ollama running. Available models: {models}")
                if not any(model_name in m for m in models):
                    print(f"  WARNING: {model_name} not found. Run: ollama pull {model_name}")
            else:
                print(f"  WARNING: Ollama returned status {r.status_code}")
        except requests.exceptions.ConnectionError:
            print("  ERROR: Cannot connect to Ollama at localhost:11434")
            print("  Make sure Ollama is installed and running (background service).")
            raise RuntimeError("Ollama not running")
        except Exception as e:
            print(f"  WARNING: Could not verify Ollama status: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 400, temperature: float = 0.7) -> str:
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
            print("  ERROR: Ollama request timed out (120s).")
            return "Thought: I need to stop due to timeout.\nAction: STOP"
        except Exception as e:
            print(f"  ERROR: Ollama generation failed: {e}")
            return "Thought: I encountered an error.\nAction: STOP"

# ==============================================================================
# LOAD LABELED CLAIMS FROM SCIFACT
# ==============================================================================
def load_scifact_claims(split="test", max_claims=75):
    """
    Load real labeled claims from SciFact.
    label: 0=REFUTES, 1=NOT ENOUGH INFO, 2=SUPPORTS
    """
    claims_file = os.path.join(BASE_DIR, "data", "scifact", f"claims_{split}.jsonl")
    
    if not os.path.exists(claims_file):
        print(f"WARNING: {claims_file} not found. Using dummy claims.")
        return None
    
    claims = []
    missing_label_count = 0
    
    with open(claims_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_claims:
                break
            doc = json.loads(line)
            
            # SciFact uses 'label', 'verdict', or implicit evidence structure
            label_map = {"REFUTES": 0, "NOT ENOUGH INFO": 1, "SUPPORTS": 2}
            
            if 'label' in doc:
                label = label_map.get(doc['label'], 1)
            elif 'verdict' in doc:
                label = label_map.get(doc['verdict'], 1)
            else:
                # Infer from evidence: no evidence = NEI
                evidence = doc.get('evidence', {})
                if not evidence:
                    label = 1  # NEI
                else:
                    # Evidence exists but we can't determine SUPPORTS vs REFUTES
                    # without corpus lookup. Default to NEI for safety.
                    label = 1
                missing_label_count += 1
            
            claims.append({
                'claim_id': f"scifact_{doc['id']}",
                'claim': doc['claim'],
                'label': label
            })
    
    if missing_label_count > 0:
        print(f"  Note: {missing_label_count} claims had no explicit label (defaulted to NEI).")
    
    print(f"Loaded {len(claims)} labeled claims from SciFact {split}")
    return claims

# ==============================================================================
# LOCAL FAISS RETRIEVER — FIXED: Uses BertTokenizer/BertModel directly
# ==============================================================================
class BioBERTEncoder:
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1", device="cpu"):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(device)
        self.model.eval()
    
    def encode(self, texts, batch_size=32):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
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
            
            token_embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings).astype("float32")


class LocalRetriever:
    def __init__(self):
        print(f"Loading FAISS index from {FAISS_INDEX_PATH}")
        self.encoder = BioBERTEncoder(BIOMODEL)
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        
        with open(FAISS_META_PATH, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Index loaded: {self.index.ntotal} vectors")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        q_emb = self.encoder.encode([query])
        faiss.normalize_L2(q_emb)
        distances, indices = self.index.search(q_emb, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            abs_dict = self.metadata['abstracts'][idx]
            results.append({
                'pmid': abs_dict['id'],
                'title': abs_dict['title'],
                'text': abs_dict['text'][:1000],
                'source': abs_dict.get('source', 'faiss'),
                'score': float(dist)
            })
        return results

# ==============================================================================
# LIVE PUBMED API
# ==============================================================================
def search_pubmed_live(query: str, max_results: int = 5) -> List[Dict]:
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
    
    try:
        r = requests.get(esearch_url, params=params, timeout=10)
        ids = r.json()["esearchresult"]["idlist"]
    except Exception as e:
        print(f"  PubMed search failed: {e}")
        return []
    
    if not ids:
        return []
    
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": ",".join(ids), "rettype": "abstract", "retmode": "text"}
    
    try:
        raw = requests.get(efetch_url, params=params, timeout=15).text
        sections = raw.split("\n\n")
        results = []
        for i, pmid in enumerate(ids):
            text = sections[i] if i < len(sections) else ""
            lines = text.strip().split('\n')
            title = lines[0] if lines else f"PubMed {pmid}"
            results.append({
                'pmid': pmid,
                'title': title[:100],
                'text': text[:1000],
                'source': 'pubmed_live',
                'score': 1.0
            })
        return results
    except Exception as e:
        print(f"  PubMed fetch failed: {e}")
        return []

# ==============================================================================
# WIKIPEDIA FALLBACK
# ==============================================================================
def search_wikipedia(query: str) -> str:
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 1}
    try:
        r = requests.get(url, params=params, timeout=10)
        results = r.json()["query"]["search"]
        return results[0]["snippet"] if results else ""
    except Exception as e:
        print(f"  Wikipedia search failed: {e}")
        return ""

# ==============================================================================
# UNIFIED SEARCH
# ==============================================================================
def search_evidence(query: str, use_local: bool = True, k: int = 5) -> List[Dict]:
    if use_local and os.path.exists(FAISS_INDEX_PATH):
        try:
            retriever = LocalRetriever()
            results = retriever.search(query, k)
            if results:
                print(f"  Local FAISS: {len(results)} results")
                return results
        except Exception as e:
            print(f"  Local search failed: {e}")
    
    print(f"  Falling back to live PubMed API...")
    results = search_pubmed_live(query, k)
    if results:
        print(f"  Live PubMed: {len(results)} results")
        return results
    
    print(f"  Falling back to Wikipedia...")
    wiki = search_wikipedia(query)
    if wiki:
        print(f"  Wikipedia: 1 result")
        return [{'pmid': 'wiki', 'title': 'Wikipedia', 'text': wiki, 'source': 'wikipedia', 'score': 0.5}]
    
    print(f"  No results from any source.")
    return []

# ==============================================================================
# PROMPT TEMPLATE
# ==============================================================================
SYSTEM_PROMPT = """You are a medical fact-checking research agent. Your job is to gather evidence to evaluate this claim:

CLAIM: "{claim}"

You have two tools:
- search_pubmed(query): searches for clinical/research evidence
- search_wikipedia(query): searches for general background

You may search up to {max_hops} times total. After each search, decide whether you have enough evidence to stop, or need to refine your query.

If PubMed returns no results, try Wikipedia for general knowledge.

Respond in this exact format each turn:
Thought: <your reasoning about what you know and what you still need>
Action: search_pubmed("...") OR search_wikipedia("...") OR STOP

Here is what has happened so far:
{scratchpad}

Continue from here."""

# ==============================================================================
# PARSING FUNCTIONS
# ==============================================================================
def parse_response(response: str) -> Tuple[str, str]:
    thought_match = re.search(r'Thought:\s*(.*?)(?=\nAction:|$)', response, re.DOTALL | re.IGNORECASE)
    action_match = re.search(r'Action:\s*(.*?)(?=\n|$)', response, re.DOTALL | re.IGNORECASE)
    
    thought = thought_match.group(1).strip() if thought_match else "No thought provided."
    action = action_match.group(1).strip() if action_match else "STOP"
    
    return thought, action

def parse_action(action: str) -> Tuple[str, str]:
    action = action.strip()
    if action.upper() == "STOP":
        return "STOP", ""
    
    match = re.search(r'search_(\w+)\(["\']?(.+?)["\']?\)', action, re.IGNORECASE)
    if match:
        return f"search_{match.group(1)}", match.group(2)
    
    if "pubmed" in action.lower():
        query = re.sub(r'.*pubmed', '', action, flags=re.IGNORECASE).strip('(" )\'')
        return "search_pubmed", query
    if "wikipedia" in action.lower():
        query = re.sub(r'.*wikipedia', '', action, flags=re.IGNORECASE).strip('(" )\'')
        return "search_wikipedia", query
    
    return "STOP", ""

def format_observation(results: List[Dict]) -> str:
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results[:3], 1):
        lines.append(f"Result {i} [{r['source']}] (ID: {r['pmid']}): {r['title']}")
        lines.append(f"  {r['text'][:300]}...")
    return "\n".join(lines)

# ==============================================================================
# MAIN REACT LOOP
# ==============================================================================
def run_react_agent(claim: str, max_hops: int = 3, use_local: bool = True,
                    model_name: str = OLLAMA_MODEL, true_label: int = None) -> AgentState:
    state = AgentState(claim=claim, max_hops=max_hops, true_label=true_label)
    llm = OllamaLLM(model_name=model_name)
    
    print(f"\n{'='*70}")
    print(f"INVESTIGATING: {claim}")
    if true_label is not None:
        label_names = {0: "REFUTED", 1: "NEI", 2: "SUPPORTED"}
        print(f"TRUE LABEL: {label_names.get(true_label, 'UNKNOWN')}")
    print(f"{'='*70}")
    
    while state.hop_count < state.max_hops:
        scratchpad_text = "\n\n".join(state.scratchpad) if state.scratchpad else "No previous actions."
        prompt = SYSTEM_PROMPT.format(claim=claim, max_hops=max_hops, scratchpad=scratchpad_text)
        
        print(f"\n--- Hop {state.hop_count + 1}/{max_hops} ---")
        response = llm.generate(prompt, max_tokens=400, temperature=0.7)
        print(f"LLM:\n{response[:500]}...")
        
        thought, action_str = parse_response(response)
        state.scratchpad.append(f"Thought: {thought}\nAction: {action_str}")
        
        if action_str.upper() == "STOP":
            print("Agent decided to STOP.")
            break
        
        tool_name, query = parse_action(action_str)
        print(f"Tool: {tool_name} | Query: '{query}'")
        
        if tool_name == "search_pubmed":
            results = search_evidence(query, use_local=use_local, k=3)
        elif tool_name == "search_wikipedia":
            wiki_text = search_wikipedia(query)
            results = [{'pmid': 'wiki', 'title': 'Wikipedia', 'text': wiki_text, 'source': 'wikipedia', 'score': 0.5}] if wiki_text else []
        else:
            print(f"Unknown tool '{tool_name}'. Stopping.")
            break
        
        state.evidence_collected.extend(results)
        observation = format_observation(results)
        state.scratchpad.append(f"Observation: {observation}")
        
        state.hop_count += 1
        time.sleep(0.3)
    
    if state.hop_count >= state.max_hops:
        print("\nMax hops reached. Forcing STOP.")
    
    save_trace(state)
    return state

def save_trace(state: AgentState, output_dir: str = None):
    if output_dir is None:
        output_dir = TRACES_DIR
    os.makedirs(output_dir, exist_ok=True)
    safe_claim = re.sub(r'[^\w]', '_', state.claim)[:50]
    filepath = os.path.join(output_dir, f"{safe_claim}.txt")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"Claim: {state.claim}\n")
        if state.true_label is not None:
            label_names = {0: "REFUTED", 1: "NEI", 2: "SUPPORTED"}
            f.write(f"True Label: {label_names.get(state.true_label)}\n")
        f.write(f"Hops used: {state.hop_count}/{state.max_hops}\n")
        f.write(f"Evidence collected: {len(state.evidence_collected)}\n")
        f.write("="*70 + "\n\n")
        for entry in state.scratchpad:
            f.write(entry + "\n\n")
    
    print(f"Trace saved: {filepath}")

# ==============================================================================
# SAVE PHASE 2 OUTPUT FOR PHASE 3
# ==============================================================================
def save_phase2_output(all_states, output_path=None):
    if output_path is None:
        output_dir = os.path.join(BASE_DIR, "data", "phase2_evidence")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "retrieved_evidence.json")
    
    all_results = {}
    label_map_reverse = {0: "REFUTED", 1: "NEI", 2: "SUPPORTED"}
    
    for claim_dict, state in all_states:
        claim_id = claim_dict['claim_id']
        evidence_list = []
        for ev in state.evidence_collected:
            evidence_list.append({
                "id": ev.get('pmid', ev.get('id', 'unknown')),
                "title": ev.get('title', 'No title'),
                "text": ev.get('text', '')[:1000],
                "score": float(ev.get('score', 0.0)),
                "source": ev.get('source', 'unknown')
            })
        
        all_results[claim_id] = {
            "claim_text": state.claim,
            "true_label": state.true_label,
            "true_label_name": label_map_reverse.get(state.true_label, "UNKNOWN"),
            "evidence": evidence_list
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"PHASE 2 OUTPUT SAVED FOR PHASE 3")
    print(f"{'='*70}")
    print(f"File: {output_path}")
    print(f"Claims: {len(all_results)}")
    print(f"Total evidence: {sum(len(v['evidence']) for v in all_results.values())}")
    print(f"{'='*70}")
    return output_path

# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================
def create_agent(model_type: str = "qwen"):
    """Return Ollama model name."""
    models = {
        "qwen": "qwen2.5:7b",
        "deepseek": "deepseek-r1:7b",
        "mistral": "mistral:7b",
        "llama": "llama3.1:8b"
    }
    return models.get(model_type, models["qwen"])

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 2: REACT AGENT WITH OLLAMA (LOCAL LLM)")
    print("=" * 70)
    
    labeled_claims = load_scifact_claims(split="test", max_claims=75)
    
    if labeled_claims is None:
        print("Using dummy claims for testing...")
        labeled_claims = [
            {'claim_id': 'dummy_001', 'claim': 'Aspirin reduces the risk of heart attack.', 'label': 2},
            {'claim_id': 'dummy_002', 'claim': 'Vaccines cause autism.', 'label': 0},
            {'claim_id': 'dummy_003', 'claim': 'Drinking lemon juice cures cancer.', 'label': 0},
        ]
    
    print(f"\nRunning Phase 2 on {len(labeled_claims)} claims...")
    print(f"Ollama model: {create_agent('qwen')}")
    print("Make sure Ollama is running in the background!")
    
    all_states = []
    for i, claim_dict in enumerate(labeled_claims, 1):
        print(f"\n{'='*70}")
        print(f"CLAIM {i}/{len(labeled_claims)}")
        print(f"{'='*70}")
        
        state = run_react_agent(
            claim=claim_dict['claim'],
            max_hops=3,
            use_local=True,
            model_name=create_agent("qwen"),
            true_label=claim_dict.get('label')
        )
        all_states.append((claim_dict, state))
    
    save_phase2_output(all_states)
    
    print(f"\n{'='*70}")
    print("PHASE 2 COMPLETE")
    print(f"{'='*70}")
    print("Next: Run Phase 3 to classify evidence hierarchy.")
    print("Reminder: Phase 2 does NOT produce accuracy %. It retrieves evidence.")