"""Utility functions for VLM Decision Lab.

Provides:
- Frame pair loading (camera + map view) from `sequence_samples/`
- Prompt builders (baseline, multiview, RAG-augmented)
- VLM invocation helpers (lazy Hugging Face model loading with graceful fallbacks)
- CLIP-based rule retrieval integration with `rag_database_lab`
- Evaluation metrics (accuracy, flip rate, streak length)

Enhancements vs initial version:
- Adds output directory helpers (`ensure_dir`, `save_text`)
- Caches rule embeddings to avoid re-embedding per frame (significant speed-up for RAG)
- More robust JSON parsing tolerant of fenced code blocks or markdown artifacts
- Dynamic model loader fallback for models without Vision2Seq architecture
- Optional temperature parameter for generation

Design goals: keep dependencies light, defer heavy imports until needed, allow notebook use.
"""
from __future__ import annotations
import os, json, re, sys, math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
from PIL import Image

# Add rag_database_lab to path for rule retrieval
LAB_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(LAB_ROOT)
RAG_LAB = os.path.join(REPO_ROOT, 'rag_database_lab')
if RAG_LAB not in sys.path:
    sys.path.insert(0, RAG_LAB)

try:  # runtime import (typing fallback used below)
    from clip_rule_retrieval import CLIPRuleImageRetriever  # type: ignore
except Exception:  # if unavailable, use a stub type
    class CLIPRuleImageRetriever:  # type: ignore
        pass

"""Single-model policy:
This lab now standardizes exclusively on Qwen/Qwen2.5-VL Instruct.
Historical multi-model support (phi, tiny, qwen2) has been removed.
Public API keeps the `model_key` parameter for backward compatibility but it is ignored.
"""

# Ensure repository root on sys.path for sibling lab import
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from prompt_engineering_lab.live_vlm_test.vlm_runners import VLMRunnerQwen25  # type: ignore
except Exception as e:  # fail fast: user must have prompt_engineering_lab intact
    raise ImportError(
        "Failed to import VLMRunnerQwen25 from prompt_engineering_lab.live_vlm_test.vlm_runners. "
        "Ensure that directory has __init__.py files (now added) and dependencies installed. Original error: "
        f"{e}"
    )

_qwen_runner: Optional[VLMRunnerQwen25] = None

def _get_qwen_runner() -> VLMRunnerQwen25:
    global _qwen_runner
    if _qwen_runner is not None:
        return _qwen_runner
    import torch
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    _qwen_runner = VLMRunnerQwen25(device=device)
    return _qwen_runner

# ---------------- Generic FS helpers ----------------
def ensure_dir(path: str) -> str:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path

def save_text(path: str, content: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def _deprecated_model_loader(_: str):  # kept so older external calls won't break if imported
    raise RuntimeError("Multi-model loading removed. Qwen2.5-VL is now the sole model.")


def generate_action(
    model_key: str,  # ignored; retained for backward compatibility
    images: List[Image.Image],
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.2,
    do_sample: bool = True,
) -> str:
    """Generate an action rationale using the standardized Qwen2.5-VL model.

    Parameters mirror the previous multi-model API; `model_key` is ignored.
    Only the first image is passed to the model (if multiview provided, encode map context inside `prompt`).
    """
    runner = _get_qwen_runner()
    primary = images[0] if images else Image.new('RGB', (256,256), 'black')
    # Adjust sampling arguments: temperature applies only if do_sample True
    res = runner.generate(
        image=primary,
        user_text=prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 0.0,
    )
    return res.strip()


def list_frame_pairs(folder_name : Optional[str] = None) -> List[Tuple[str, str]]:
    """Return sorted list of (cam_path, map_path) pairs based on naming convention frameXX_cam.png/map.
    Skips if either side missing."""
    folder = os.path.join(LAB_ROOT, folder_name) if folder_name is not None else os.path.join(LAB_ROOT, 'example_edge_samples')
    files = sorted(f for f in os.listdir(folder) if f.endswith('.png'))
    pairs: List[Tuple[str, str]] = []
    by_prefix: Dict[str, Dict[str, str]] = {}
    for f in files:
        m = re.match(r'(frame\d\d)_(cam|map)\.png', f)
        if not m:
            continue
        prefix, kind = m.group(1), m.group(2)
        by_prefix.setdefault(prefix, {})[kind] = os.path.join(folder, f)
    for prefix, d in sorted(by_prefix.items()):
        if 'cam' in d and 'map' in d:
            pairs.append((d['cam'], d['map']))
    return pairs


# ---------------- Prompt Builders ----------------
BASE_ACTION_INSTRUCTION = (
    "You are a driving assistant of robot in simulation environment. Observe the camera view and provide a decision what to do in current situation. "
    "OUTPUT ONLY JSON with keys: "
    "{'action': 'FORWARD|LEFT|RIGHT|STOP|SLOW', 'rationale': '<short reason>'}. "
    "Use only visible evidence."
)

MULTIVIEW_PREFIX = (
    "Context: Two views provided. View[1]=Front Camera. View[2]=Bird's-eye map with red dot for robot position. "
    "Fuse them before deciding."
)

RAG_RULE_HEADER = "Relevant driving rules (ranked):\n"


def build_single_view_prompt(extra: str = "") -> str:
    return BASE_ACTION_INSTRUCTION + (" " + extra if extra else "")


def build_multiview_prompt(extra: str = "") -> str:
    return MULTIVIEW_PREFIX + " " + BASE_ACTION_INSTRUCTION + (" " + extra if extra else "")


def build_rag_prompt(base_prompt: str, ranked_rules: List[Dict], max_rules: int = 3) -> str:
    lines = [RAG_RULE_HEADER]
    for r in ranked_rules[:max_rules]:
        lines.append(f"- {r['rule_id']}: {r['rule_text']}")
    lines.append("\nApply the most relevant rules strictly when choosing action.\n")
    return "\n".join(lines) + base_prompt


# ---------------- RAG Retrieval ----------------
_rule_cache: Optional[List[Dict]] = None
_retriever: Optional[CLIPRuleImageRetriever] = None


def load_rules(path: Optional[str] = None) -> List[Dict]:
    global _rule_cache
    if _rule_cache is not None:
        return _rule_cache
    path = path or os.path.join(RAG_LAB, 'rules.json')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Normalize keys
    rules = []
    for item in data:
        rules.append({'id': item['id'], 'text': item['rule_text']})
    _rule_cache = rules
    return rules


def ensure_retriever() -> Optional[CLIPRuleImageRetriever]:
    global _retriever
    if CLIPRuleImageRetriever is None:
        return None
    if _retriever is None:
        _retriever = CLIPRuleImageRetriever()
    return _retriever


_rule_emb_cache: Optional[Tuple[Any, List[Any], List[str]]] = None

def retrieve_rules_for_image(image: Image.Image, top_k: int = 5, refresh_cache: bool = False) -> List[Dict]:
    """Return top-k rules for an image using cached rule embeddings (unless refresh requested)."""
    retriever = ensure_retriever()
    if retriever is None:
        return []
    global _rule_emb_cache
    if _rule_emb_cache is None or refresh_cache:
        rules = load_rules()
        _rule_emb_cache = retriever.embed_rules(rules)  # (embeddings, ids, texts)
    rule_embs, ids, texts = _rule_emb_cache
    img_emb = retriever.embed_image(image.convert('RGB'))
    ranked = retriever.rank_rules(img_emb, rule_embs, ids, texts, top_k=top_k)
    return [
        {
            'rule_id': r['rule_id'],
            'rule_text': r['rule_text'],
            'similarity': r['similarity'],
        }
        for r in ranked
    ]


# ---------------- Evaluation Helpers ----------------
VALID_ACTIONS = {"FORWARD", "LEFT", "RIGHT", "STOP", "SLOW"}

def parse_action_json(output: str) -> Optional[Dict]:
    """Attempt to parse JSON object containing 'action' & 'rationale'. Robust to stray text.

    Handling improvements:
    - Strips markdown fences ```json ... ``` if present.
    - Truncates before first unmatched closing brace to reduce hallucinated tail text.
    - Accepts lowercase actions by normalizing then validating.
    """
    import json
    text = output.strip()
    # remove code fences
    if text.startswith('```'):
        text = re.sub(r'^```[a-zA-Z]*\n', '', text)
        text = text.split('```')[0]
    # locate first JSON object
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if not m:
        return None
    snippet = m.group(0)
    try:
        obj = json.loads(snippet)
        if 'action' in obj and 'rationale' in obj:
            act = str(obj['action']).upper()
            if act in VALID_ACTIONS:
                obj['action'] = act
                obj['rationale'] = str(obj['rationale']).strip()
                return obj
    except Exception:
        return None
    return None


def action_accuracy(outputs: List[Dict], ground_truth: List[str]) -> float:
    if not outputs or not ground_truth:
        return 0.0
    correct = 0
    for o, gt in zip(outputs, ground_truth):
        if o.get('action') == gt:
            correct += 1
    return correct / len(ground_truth)


def flip_rate(actions: List[str]) -> float:
    if len(actions) < 2:
        return 0.0
    flips = sum(1 for a,b in zip(actions, actions[1:]) if a != b)
    return flips / (len(actions) - 1)


def longest_streak(actions: List[str]) -> int:
    best = cur = 0
    prev = None
    for a in actions:
        if a == prev:
            cur += 1
        else:
            cur = 1
            prev = a
        best = max(best, cur)
    return best

__all__ = [
    'list_frame_pairs', 'build_single_view_prompt', 'build_multiview_prompt', 'build_rag_prompt',
    'generate_action', 'retrieve_rules_for_image', 'parse_action_json', 'action_accuracy',
    'flip_rate', 'longest_streak', 'ensure_dir', 'save_text'
]
