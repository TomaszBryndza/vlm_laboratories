# VLM Decision Lab

Laboratory for analyzing how a Vision-Language Models (Qwen2.5-VL Instruct) suggests driving actions for a simulated Duckietown robot under varying information regimes:

1. Single front camera view.
2. Multi-view fusion (front camera + bird's-eye map).
3. Retrieval-Augmented Generation (RAG) with traffic/behavior rules from `rag_database_lab`.
4. Temporal sequence stability (edge set: 6 frames; longer sequence: 10 frames).
5. Prompt design / ablation (structured schema vs concise vs minimal).

## Goals
- Align suggested action with hand-authored ground truth (when available).
- Measure impact of map context on action quality.
- Assess benefit of top-k rule retrieval for reasoning quality.
- Quantify temporal stability (flip rate, longest streak).
- Provide reproducible notebooks + lightweight helpers.

## Directory Layout (Corrected)
```
vlm_decision_lab/
  README.md
  requirements.txt
  utils.py
  evaluation.py
  experiment1_single_view_baseline.ipynb
  experiment2_multiview_fusion.ipynb
  experiment3_rag_augmented_actions.ipynb
  experiment4_comparative_metrics.ipynb
  experiment5_prompt_ablation.ipynb
  experiment6_sequence_consistency.ipynb
  example_edge_samples/          # 6 paired frames + ground truth actions
  example_sequence_samples/      # 10 paired frames + ground truth actions
```
Notes:
- No root-level `ground_truth_actions.json`; each sample directory contains its own file.
- Earlier drafts referenced `sequence_samples/`; correct name is `example_sequence_samples/`.
- Folders `outputs/`, `metrics/`, `rag_cache/` appear when generated.

## Core Concepts
### Action JSON Schema
```json
{
  "action": "FORWARD|LEFT|RIGHT|STOP|SLOW",
  "rationale": "Short justification based only on visible evidence"
}
```
### Data / Frames
Two datasets:
- Edge cases: `example_edge_samples/` (frames 01–06)
- Longer sequence: `example_sequence_samples/` (frames 01–10)
Naming convention: `frameXX_cam.png`, `frameXX_map.png`. Add new pairs by extending numbering.

### Multi-View Fusion
Either pass both images (if runner supports it) or encode map context in prompt text. Current helper sends a single image; incorporate map semantics manually in prompt until multi-image support extended.

### RAG Augmentation
1. Embed rule texts from `rag_database_lab/rules.json` (CLIP).
2. Embed current camera frame.
3. Rank & inject top-k rules before requesting action.
4. Compare vs baseline (accuracy, rationale content, JSON validity).

### Sequence Consistency
Compute over ordered frames: flip rate, longest streak, accuracy (if ground truth present). Clearly state which dataset (6 vs 10 frames) you evaluate.

### Prompt Ablation
Compare full structured schema vs concise vs minimal queries; measure JSON validity & action accuracy.

## Experiments Overview
| Notebook | Focus | Example Outputs |
|----------|-------|-----------------|
| experiment1_single_view_baseline | Single camera baseline | `outputs/experiment1_summary.json` |
| experiment2_multiview_fusion | Camera + map fusion | `outputs/experiment2_multiview_summary.json` |
| experiment3_rag_augmented_actions | Rule retrieval impact | `outputs/experiment3_rag_summary.json` |
| experiment4_comparative_metrics | Cross-experiment metrics | `metrics/experiment4_quick_aggregate.json` |
| experiment5_prompt_ablation | Prompt variants | `outputs/experiment5_prompt_validity.json` (optional) |
| experiment6_sequence_consistency | Temporal stability | `outputs/experiment6_sequence_consistency_summary.json` |

## Running the Lab (Quick Start)
```bash
cd vlm_decision_lab
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python evaluation.py --outputs outputs \
  --ground-truth example_edge_samples/ground_truth_actions.json \
  --save metrics/summary.json
```

## Evaluation Script
`evaluation.py` aggregates metrics from `.txt` raw outputs:
Metrics JSON fields: `frames_evaluated`, `json_validity`, `action_accuracy`, `flip_rate`, `longest_streak`, `actions`.

## Output Naming Conventions
| Pattern | Meaning |
|---------|---------|
| `frameXX_single_baseline.txt` | Raw model generation (single-view) |
| `experiment1_summary.json` | Baseline metrics |
| `experiment2_multiview_summary.json` | Multiview actions summary |
| `experiment3_rag_summary.json` | Baseline vs RAG comparison |
| `experiment6_sequence_consistency_summary.json` | Temporal stability metrics |

## RAG Retrieval & Caching
Embeddings cached in-memory. Force refresh with `refresh_cache=True`. Disk persistence (e.g., `rag_cache/rules_clip.pt`) is a future extension.

## Prompt Customization
```python
from utils import build_single_view_prompt
prompt = build_single_view_prompt("Emphasize collision avoidance.")
```
Maintain JSON contract for parser robustness.

## Single-Model Policy
Qwen2.5-VL Instruct is the sole model. Parameter `model_key` in `generate_action` is ignored (backward compatibility).


## Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| OOM / CUDA | GPU memory too small | Use CPU / smaller model / lower resolution |
| JSON parse failure | Model added prose / malformed JSON | Strengthen prompt, lower temperature, regex cleanup |
| No action extracted | Parser failed to find schema | Verify allowed actions unchanged |
| Slow RAG per frame | Re-embedding rules each time | Avoid repeated `refresh_cache=True` |

## Performance Tips
- Prefer GPU for speed (CPU works but slower).
- Lower `max_new_tokens` (e.g., 64) for shorter rationales.
- Optionally batch prompts by wrapping multiple calls (not built-in).

## Metrics Definitions
- Action Accuracy: proportion of frames with correct action.
- Flip Rate: changes between consecutive actions.
- Longest Streak: longest run of identical actions.
- JSON Validity: share of outputs parsed successfully.
- Rule Match Rate: fraction of injected rules referenced (substring) in rationale.
- Latency: average generation time per frame.
(*RAG Gain* not currently computed; implement separately if needed.)

