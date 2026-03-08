"""
Anti-Copy Detector

Two-signal voting system for detecting model copies.

Signals:
  1. hidden_states cosine  – model internal representation (independent)
  2. logprob top-3 cosine  – output distribution direction (independent)

JS divergence and token agreement are computed for diagnostics but do NOT
vote, because they are derived from the same logprob/topk data as cosine
and carry redundant information.

Decision: is_copy when ALL available signals vote copy, with at least 1 signal.
"""

import numpy as np
from typing import Dict, List, Tuple

from affine.core.setup import logger
from affine.src.anticopy.loader import LogprobsLoader
from affine.src.anticopy.metrics import js_divergence_topk, token_agreement_rate
from affine.src.anticopy.models import CopyPair, MinerLogprobs


class AntiCopyDetector:
    """Detect model copies using two independent signals.

    Args:
        hs_threshold:      hidden_states cosine >= this → vote copy
        cosine_threshold:   logprob cosine >= this → vote copy
        min_tasks:          minimum shared tasks required for comparison
    """

    def __init__(
        self,
        hs_threshold: float = 0.99,
        cosine_threshold: float = 0.99,
        min_tasks: int = 10,
    ):
        self.hs_threshold = hs_threshold
        self.cosine_threshold = cosine_threshold
        self.min_tasks = min_tasks

    def _cosine_per_task(
        self, vecs_a: np.ndarray, vecs_b: np.ndarray
    ) -> np.ndarray:
        """Compute per-task cosine similarity. Shape (n_tasks,)."""
        norms_a = np.linalg.norm(vecs_a, axis=1, keepdims=True)
        norms_b = np.linalg.norm(vecs_b, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            na = np.where(norms_a > 0, vecs_a / norms_a, 0.0)
            nb = np.where(norms_b > 0, vecs_b / norms_b, 0.0)
        return (na * nb).sum(axis=1)

    def detect(self, miners: Dict[int, MinerLogprobs]) -> List[CopyPair]:
        """Run detection over all miner pairs.

        Returns:
            List of CopyPair, sorted by confidence descending
        """
        if len(miners) < 2:
            return []

        uid_list = sorted(miners.keys())
        n = len(uid_list)
        logger.info(f"anti_copy: running detection on {n} miners ({n*(n-1)//2} pairs)")

        results: List[CopyPair] = []

        for i, uid_a in enumerate(uid_list):
            for uid_b in uid_list[i + 1:]:
                ma, mb = miners[uid_a], miners[uid_b]

                # ── Logprob signals ──────────────────────────────────
                lp_common = sorted(
                    set(ma.task_logprobs) & set(mb.task_logprobs)
                )
                has_logprobs = len(lp_common) >= self.min_tasks

                med_cosine = float("nan")
                med_js = float("nan")
                med_agree = float("nan")
                task_cosine_map: Dict[int, float] = {}
                n_tasks = 0

                if has_logprobs:
                    vecs_a = np.stack([ma.task_logprobs[t] for t in lp_common])
                    vecs_b = np.stack([mb.task_logprobs[t] for t in lp_common])
                    cosines = self._cosine_per_task(vecs_a, vecs_b)
                    med_cosine = float(np.nanmedian(cosines))
                    task_cosine_map = {
                        t: float(c) for t, c in zip(lp_common, cosines)
                    }
                    n_tasks = len(lp_common)

                    # JS divergence
                    js_vals = []
                    for tid in lp_common:
                        topk_a = ma.task_topk.get(tid, [])
                        topk_b = mb.task_topk.get(tid, [])
                        if topk_a and topk_b:
                            js_vals.append(js_divergence_topk(topk_a, topk_b))
                    if js_vals:
                        med_js = float(np.nanmedian(js_vals))

                    # Token agreement
                    agree_vals = []
                    for tid in lp_common:
                        tok_a = ma.task_tokens.get(tid, [])
                        tok_b = mb.task_tokens.get(tid, [])
                        if tok_a and tok_b:
                            agree_vals.append(token_agreement_rate(tok_a, tok_b))
                    if agree_vals:
                        med_agree = float(np.nanmedian(agree_vals))

                # ── Hidden states signal ─────────────────────────────
                hs_common = sorted(
                    set(ma.task_hidden_states) & set(mb.task_hidden_states)
                )
                has_hs = len(hs_common) >= self.min_tasks
                med_hs = float("nan")

                if has_hs:
                    hs_a = np.stack([ma.task_hidden_states[t] for t in hs_common])
                    hs_b = np.stack([mb.task_hidden_states[t] for t in hs_common])
                    hs_cosines = self._cosine_per_task(hs_a, hs_b)
                    med_hs = float(np.nanmedian(hs_cosines))
                    # Use max task count
                    n_tasks = max(n_tasks, len(hs_common))

                # Need at least some data to compare
                if not has_logprobs and not has_hs:
                    continue

                # ── Voting (2 independent signals) ─────────────────
                votes = 0
                total_votes = 0

                if has_hs:
                    total_votes += 1
                    if med_hs >= self.hs_threshold:
                        votes += 1

                if has_logprobs and not np.isnan(med_cosine):
                    total_votes += 1
                    if med_cosine >= self.cosine_threshold:
                        votes += 1

                # All available signals must agree; need at least 1
                is_copy = total_votes >= 1 and votes == total_votes

                # Confidence: fraction of votes
                confidence = votes / max(total_votes, 1)

                results.append(
                    CopyPair(
                        uid_a=uid_a,
                        uid_b=uid_b,
                        hotkey_a=ma.hotkey,
                        hotkey_b=mb.hotkey,
                        cosine_similarity=med_cosine,
                        hs_cosine=med_hs,
                        js_divergence=med_js,
                        token_agreement=med_agree,
                        n_tasks=n_tasks,
                        is_copy=is_copy,
                        confidence=confidence,
                        votes=votes,
                        total_votes=total_votes,
                        task_cosines=task_cosine_map,
                    )
                )

        copies = [r for r in results if r.is_copy]
        results.sort(key=lambda r: r.confidence, reverse=True)

        logger.info(
            f"anti_copy: {len(copies)} copy pairs detected "
            f"out of {len(results)} pairs evaluated"
        )
        return results


async def detect_copies(
    miners: List[Dict],  # [{uid, hotkey, revision}]
    hs_threshold: float = 0.99,
    cosine_threshold: float = 0.93,
    min_tasks: int = 3,
) -> List[CopyPair]:
    """Convenience function: load logprobs from DB and run detection.

    Returns:
        All CopyPair results sorted by confidence (check .is_copy for flagged pairs)
    """
    loader = LogprobsLoader()
    miner_data = await loader.load_all_miners(miners)

    detector = AntiCopyDetector(
        hs_threshold=hs_threshold,
        cosine_threshold=cosine_threshold,
        min_tasks=min_tasks,
    )
    return detector.detect(miner_data)
