"""CLIP-based rule retrieval for autonomous driving simulation scenes.

Provides utilities to embed rule texts and scene images, rank rules by cosine similarity,
then compute comparison metrics against Qwen-based reasoning outputs.

Dependencies:
    transformers, torch, Pillow

Usage Example:
    from clip_rule_retrieval import CLIPRuleImageRetriever, compute_metrics
    retriever = CLIPRuleImageRetriever(device='cuda')
    rule_embs, rule_ids, rule_texts = retriever.embed_rules(rules)  # rules: list of {'id': int, 'text': str}
    img_emb = retriever.embed_image(Image.open(selected_image_path).convert('RGB'))
    ranking = retriever.rank_rules(img_emb, rule_embs, rule_ids, rule_texts, top_k=10)
    clip_rank = [str(r['rule_id']) for r in ranking]
    metrics = compute_metrics(clip_rank, batch_order, {str(r['rule_id']): r['score'] for r in parsed_iter})

"""
from __future__ import annotations
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import List, Dict, Tuple, Optional
import math

class CLIPRuleImageRetriever:
    def __init__(self, model_name: str = 'openai/clip-vit-base-patch32', device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self._rule_cache = None  # (embeddings, ids, texts)

    def embed_rules(self, rules: List[Dict]) -> Tuple[torch.Tensor, List[int], List[str]]:
        """Embed all rule texts.
        Args:
            rules: list of {'id': int|str, 'text': str}
        Returns:
            (embeddings tensor [N,D], ids list, texts list)
        """
        texts = [r['text'] for r in rules]
        ids = [r['id'] for r in rules]
        inputs = self.processor(text=texts, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            text_features = self.model.get_text_features(**inputs)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        self._rule_cache = (text_features, ids, texts)
        return text_features, ids, texts

    def embed_image(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            img_features = self.model.get_image_features(**inputs)
        img_features = torch.nn.functional.normalize(img_features, dim=-1)
        return img_features

    def rank_rules(
        self,
        image_embedding: torch.Tensor,
        rule_embeddings: torch.Tensor,
        rule_ids: List[int],
        rule_texts: List[str],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """Compute cosine similarity and return a ranked list of rule dicts.
        Args:
            image_embedding: [1,D] tensor
            rule_embeddings: [N,D] tensor (normalized)
            rule_ids: list of ids
            rule_texts: list of texts aligned with rule_ids
            top_k: if provided, truncate results
        Returns:
            List of {'rule_id': id, 'rule_text': str, 'similarity': float}
        """
        # image_embedding shape [1,D]; rule_embeddings [N,D]
        sims = (image_embedding @ rule_embeddings.T).squeeze(0)  # [N]
        ranked_indices = torch.argsort(sims, descending=True).tolist()
        results = [
            {
                'rule_id': rule_ids[i],
                'rule_text': rule_texts[i],
                'similarity': float(sims[i].item())
            }
            for i in ranked_indices
        ]
        if top_k is not None:
            results = results[:top_k]
        return results


def _spearman(a: List[int], b: List[int]) -> float:
    """Compute Spearman correlation given two rank lists represented by positions of the same items.
    a, b: positions aligned by the same item order.
    Returns Spearman rho or 0 if insufficient length.
    """
    n = len(a)
    if n < 2:
        return 0.0
    diff_sq = sum((a[i] - b[i])**2 for i in range(n))
    return 1 - (6 * diff_sq) / (n * (n*n - 1))


def compute_metrics(
    clip_rank: List[str],
    batch_order: List[str],
    iter_scores_map: Dict[str, float],
    k: int = 5
) -> Dict:
    """Produce metrics comparing CLIP ranking vs Qwen batch ordering and iterative scores.
    Args:
        clip_rank: list of rule_id strings ordered by CLIP similarity (descending)
        batch_order: list of rule_id strings ordered by Qwen batch reasoning (most -> least applicable)
        iter_scores_map: mapping rule_id string -> iterative Qwen applicability score
        k: top-k for overlap metrics (default 5)
    Returns:
        dict with metrics: spearman_clip_vs_batch, topk_overlap_count, topk_overlap_ids,
        mrr_clip_vs_batch_top, pearson_clip_similarity_vs_iter_score, ndcg_clip_vs_iter (basic),
        clip_top, batch_top, clip_rank_count
    """
    metrics = {}

    # Spearman correlation between ranking orders (only if same set)
    common = [rid for rid in batch_order if rid in clip_rank]
    if common:
        clip_positions = [clip_rank.index(rid) + 1 for rid in common]
        batch_positions = [batch_order.index(rid) + 1 for rid in common]
        metrics['spearman_clip_vs_batch'] = _spearman(clip_positions, batch_positions)
    else:
        metrics['spearman_clip_vs_batch'] = 0.0

    # Top-k overlap
    clip_top = clip_rank[:k]
    batch_top = batch_order[:k]
    overlap = sorted(set(clip_top) & set(batch_top))
    metrics['topk_overlap_count'] = len(overlap)
    metrics['topk_overlap_ids'] = overlap

    # MRR: ground truth = first in batch_order (most applicable per Qwen)
    if batch_order:
        gt = batch_order[0]
        if gt in clip_rank:
            rr = 1.0 / (clip_rank.index(gt) + 1)
        else:
            rr = 0.0
        metrics['mrr_clip_vs_batch_top'] = rr
    else:
        metrics['mrr_clip_vs_batch_top'] = 0.0

    # Pearson between CLIP similarity (inverse rank) and iterative scores
    # approximate similarity via inverse rank normalized
    import math
    iter_pairs = []
    for rid in clip_rank:
        if rid in iter_scores_map:
            # similarity proxy: higher rank => larger value
            sim_proxy = 1.0 / (clip_rank.index(rid) + 1)
            iter_pairs.append((sim_proxy, iter_scores_map[rid]))
    if len(iter_pairs) > 1:
        xs = [p[0] for p in iter_pairs]
        ys = [p[1] for p in iter_pairs]
        mx = sum(xs)/len(xs)
        my = sum(ys)/len(ys)
        num = sum((x-mx)*(y-my) for x,y in iter_pairs)
        den = math.sqrt(sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys))
        pearson = num/den if den else 0.0
    else:
        pearson = 0.0
    metrics['pearson_clip_rank_vs_iter_score'] = pearson

    # Simple nDCG@k with iterative score as relevance
    def _dcg(vals: List[float]) -> float:
        return sum(v / math.log2(i+2) for i, v in enumerate(vals))
    iter_scores_for_clip_top = [iter_scores_map.get(rid, 0.0) for rid in clip_top]
    ideal_scores = sorted(iter_scores_for_clip_top, reverse=True)
    dcg = _dcg(iter_scores_for_clip_top)
    idcg = _dcg(ideal_scores) if ideal_scores else 0.0
    metrics['ndcg_clip_top_vs_iter'] = (dcg / idcg) if idcg else 0.0

    metrics['clip_top'] = clip_top
    metrics['batch_top'] = batch_top
    metrics['clip_rank_count'] = len(clip_rank)
    return metrics

__all__ = [
    'CLIPRuleImageRetriever',
    'compute_metrics'
]
