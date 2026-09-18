"""Belief update: Bayes-ish revision after each observation.

For each hypothesis we estimate P(observation | hypothesis) in [0.05, 0.95]:
LLM judges it when a real model exists, else deterministic word-overlap
between the observation and the hypothesis's predicted/supporting text.
Then: support *= likelihood, renormalize. Stale-proof and bounded.
"""
from __future__ import annotations

import re
from typing import List

from ..core.types import Hypothesis, Observation
from ..models import HeuristicModel, complete_json

_UPDATE_SYSTEM = (
    "You score evidence fit. For EACH hypothesis, given one new observation, "
    "return likelihood 0..1: how expected is this observation if that hypothesis "
    "were true? 0.9 = strongly predicted, 0.5 = unsurprising either way, "
    "0.1 = very surprising. Return JSON: {\"likelihoods\": {\"H1\": 0.8, ...}}"
)

_WORD = re.compile(r"[a-z]{4,}")
_STOP = {
    "that", "this", "with", "from", "have", "what", "would", "could",
    "also", "today", "shows", "show", "table", "data", "observations",
    "observation", "hours", "hour", "rows", "columns", "count", "many",
    "most", "such", "than", "then", "them", "they", "their", "there",
}


def _stems(text: str) -> set:
    out = set()
    for w in _WORD.findall(text.lower()):
        if w in _STOP:
            continue
        # crude stemming: drop plural/verb endings
        for end in ("ing", "ies", "es", "ed", "s"):
            if w.endswith(end) and len(w) - len(end) >= 4:
                w = w[:-len(end)] if end != "ies" else w[:-3] + "y"
                break
        if len(w) >= 4:
            out.add(w)
    return out


def likelihoods(hypotheses: List[Hypothesis], obs: Observation, model) -> dict:
    if isinstance(model, HeuristicModel):
        return _heuristic_likelihoods(hypotheses, obs)
    try:
        blob = "\n".join(f"{h.id}: {h.statement} | predicts: {h.predicted_observations}"
                         for h in hypotheses)
        obj = complete_json(
            model, _UPDATE_SYSTEM,
            f"Observation ({obs.source}, reliability {obs.reliability}): {obs.content}\n"
            f"Hypotheses:\n{blob}")
        out = {}
        for h in hypotheses:
            v = float(obj.get("likelihoods", {}).get(h.id, 0.5))
            out[h.id] = min(0.95, max(0.05, v))
        return out
    except Exception:
        return _heuristic_likelihoods(hypotheses, obs)


def _heuristic_likelihoods(hypotheses: List[Hypothesis], obs: Observation) -> dict:
    """Distinctive-stem matching: a hypothesis is supported when the observation
    contains words predicted by IT but not by its rivals. Sharpened so one
    observation can actually separate hypotheses."""
    obs_stems = _stems(obs.content)
    own = {h.id: _stems(" ".join(h.predicted_observations + h.supporting)) for h in hypotheses}
    contra = {h.id: _stems(" ".join(h.contradicting)) for h in hypotheses}
    # distinctive = stems this hyp uses that rivals don't
    scores = {}
    for h in hypotheses:
        rival = set()
        rival_contra = set()
        for h2 in hypotheses:
            if h2.id != h.id:
                rival |= own[h2.id]
                rival_contra |= contra[h2.id]
        distinctive = own[h.id] - rival
        hits = len(distinctive & obs_stems)
        hit_rate = hits / max(1, min(len(distinctive), 8))
        distinctive_contra = contra[h.id] - rival_contra
        chits = len(distinctive_contra & obs_stems)
        contra_rate = chits / max(1, min(len(distinctive_contra), 6))
        v = 0.15 + 0.75 * hit_rate - 0.60 * contra_rate
        v = 0.5 + (v - 0.5) * max(0.3, obs.reliability)
        scores[h.id] = min(0.95, max(0.05, v))
    # sharpen: softmax-ish spread so winners separate from losers
    ids = [h.id for h in hypotheses]
    exps = {k: pow(2.2, (scores[k] - 0.5) * 6) for k in ids}
    total = sum(exps.values()) or 1.0
    return {k: 0.08 + 0.84 * exps[k] / total for k in ids}


def update(hypotheses: List[Hypothesis], obs: Observation, model) -> List[Hypothesis]:
    lh = likelihoods(hypotheses, obs, model)
    for h in hypotheses:
        h.support = max(1e-6, h.support * lh.get(h.id, 0.5))
        if lh.get(h.id, 0.5) >= 0.65:
            h.supporting.append(obs.content[:200])
        elif lh.get(h.id, 0.5) <= 0.35:
            h.contradicting.append(obs.content[:200])
    total = sum(h.support for h in hypotheses) or 1.0
    for h in hypotheses:
        h.support /= total
    return hypotheses
