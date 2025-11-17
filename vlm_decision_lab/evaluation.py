"""Aggregate evaluation for VLM Decision Lab outputs.

Scans an outputs/ directory for text files containing model generations. Each file expected to contain
raw model output with an embedded JSON object specifying action + rationale.

Usage (from lab root):
    python evaluation.py --outputs outputs --ground-truth ground_truth_actions.json

Produces metrics JSON summary to metrics/summary.json.
"""
from __future__ import annotations
import os, json, argparse, time
from typing import Dict, List
from utils import parse_action_json, action_accuracy, flip_rate, longest_streak, ensure_dir


def load_ground_truth(path: str) -> Dict[str, str]:
    if not os.path.isfile(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def frame_key_from_filename(fname: str) -> str:
    # expects pattern frameXX_... or frameXX
    base = os.path.basename(fname)
    if 'frame' in base:
        for part in base.split('_'):
            if part.startswith('frame'):
                return part[:7]  # frame01
    return os.path.splitext(base)[0]


def collect_outputs(outputs_dir: str) -> List[Dict]:
    items = []
    for fname in sorted(os.listdir(outputs_dir)):
        if not fname.endswith('.txt'):
            continue
        path = os.path.join(outputs_dir, fname)
        text = open(path, 'r', encoding='utf-8').read()
        parsed = parse_action_json(text)
        items.append({
            'file': fname,
            'frame': frame_key_from_filename(fname),
            'raw': text,
            'parsed': parsed
        })
    return items


def compute_metrics(entries: List[Dict], gt_map: Dict[str, str]) -> Dict:
    parsed_objs = [e['parsed'] or {} for e in entries]
    actions = [p.get('action', '?') for p in parsed_objs]
    ground_truth_seq = [gt_map.get(e['frame'], None) for e in entries]
    gt_filtered = [g for g in ground_truth_seq if g is not None]
    # accuracy only where GT exists and lengths line up
    actionable_pairs = [p for p,g in zip(parsed_objs, ground_truth_seq) if g is not None]
    acc = action_accuracy(actionable_pairs, [g for g in ground_truth_seq if g is not None]) if gt_filtered else None
    validity = sum(1 for p in parsed_objs if 'action' in p)/len(parsed_objs) if parsed_objs else 0.0
    fr = flip_rate(actions)
    streak = longest_streak(actions)
    return {
        'frames_evaluated': len(entries),
        'json_validity': validity,
        'action_accuracy': acc,
        'flip_rate': fr,
        'longest_streak': streak,
        'actions': actions,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outputs', default='outputs', help='Directory of model output .txt files')
    ap.add_argument('--ground-truth', default='ground_truth_actions.json', help='Ground truth JSON file')
    ap.add_argument('--save', default='metrics/summary.json', help='Path to save metrics JSON')
    args = ap.parse_args()

    if not os.path.isdir(args.outputs):
        raise SystemExit(f"Outputs directory not found: {args.outputs}")

    gt = load_ground_truth(args.ground_truth)
    entries = collect_outputs(args.outputs)
    metrics = compute_metrics(entries, gt)
    ensure_dir(os.path.dirname(args.save))
    with open(args.save, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
