"""Experiment planning + ranking.

Planner: propose one candidate experiment per tool (LLM writes the args,
heuristic path matches tool keywords to hypotheses).
Ranker: score = information gain per unit cost, penalized by latency and risk.
Cheap, safe checks that split hypotheses go first.
"""
from __future__ import annotations

from typing import List

from ..core.types import Experiment, Hypothesis
from ..models import HeuristicModel, complete_json
from ..tools import Tool

_PLAN_SYSTEM = (
    "You design discriminating experiments. Given hypotheses and available tools "
    "(name + description), propose at most one experiment per tool: which tool, "
    "what args, which hypothesis ids it tests, and expected_gain 0..1 "
    "(1 = would settle it alone). Prefer cheap read-only checks first. "
    "Return JSON: {\"experiments\": [{\"tool\": ..., \"args\": {...}, "
    "\"targets\": [\"H1\"], \"why\": ..., \"expected_gain\": 0.8}]}"
)


def plan(hypotheses: List[Hypothesis], tools: List[Tool], model,
         question: str = "") -> List[Experiment]:
    if isinstance(model, HeuristicModel):
        return _heuristic_plan(hypotheses, tools)
    try:
        desc = "\n".join(f"- {t.name}: {t.description} (cost={t.cost})" for t in tools)
        hyps = "\n".join(f"- {h.id}: {h.statement} (missing: {h.missing})"
                         for h in hypotheses)
        obj = complete_json(
            model, _PLAN_SYSTEM,
            f"Observation: {question}\nHypotheses:\n{hyps}\nTools:\n{desc}")
        by_name = {t.name: t for t in tools}
        out: List[Experiment] = []
        for i, e in enumerate(obj.get("experiments", [])):
            tool = by_name.get(str(e.get("tool", "")))
            if tool is None:
                continue
            out.append(Experiment(
                id=f"E{i+1}", name=f"{tool.name} check",
                description=str(e.get("why", tool.description)),
                tool_name=tool.name,
                args=dict(e.get("args", {})),
                cost=tool.cost, latency_s=tool.latency_s, risk=tool.risk,
                expected_gain=float(e.get("expected_gain", 0.5)),
                targets=[str(t) for t in e.get("targets", [])] or [h.id for h in hypotheses],
            ))
        if out:
            return out
    except Exception:
        pass
    return _heuristic_plan(hypotheses, tools)


def _heuristic_plan(hypotheses: List[Hypothesis], tools: List[Tool]) -> List[Experiment]:
    """Match each tool to the hypotheses its keywords overlap with."""
    out: List[Experiment] = []
    for i, t in enumerate(tools):
        blob = (t.name + " " + t.description + " " + t.keywords).lower()
        targets = [h.id for h in hypotheses
                   if any(w in blob for w in h.statement.lower().split()[:40]
                           if len(w) > 4)
                   or any(w in blob for w in " ".join(h.predicted_observations).lower().split()
                           if len(w) > 5)]
        if not targets:  # unknown mapping: assume it could inform everything a little
            targets = [h.id for h in hypotheses]
        gain = 0.9 if len(targets) <= max(1, len(hypotheses) // 2) else 0.5
        out.append(Experiment(
            id=f"E{i+1}", name=f"{t.name} check", description=t.description,
            tool_name=t.name, args={}, cost=t.cost, latency_s=t.latency_s,
            risk=t.risk, expected_gain=gain, targets=targets))
    return out


def rank(experiments: List[Experiment], hypotheses: List[Hypothesis]) -> List[Experiment]:
    """Value-of-information per cost. Splits belief mass -> higher score."""
    n = max(1, len(hypotheses))
    scored = []
    for e in experiments:
        coverage = len(set(e.targets)) / n  # 1 = touches everything (less discriminating)
        split_bonus = 1.0 - abs(coverage - 0.5) * 2 * 0.4  # prefer ~half coverage
        denom = 1.0 + e.cost + 0.3 * e.latency_s + 5.0 * e.risk
        score = (e.expected_gain * split_bonus) / denom
        scored.append((score, e))
    scored.sort(key=lambda p: -p[0])
    return [e for _, e in scored]
