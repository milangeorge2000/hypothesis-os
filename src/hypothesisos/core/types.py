"""Core data types: what the agent knows, what it suspects, what it can try."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Observation:
    """Something currently known: a fact plus where it came from."""

    content: str
    source: str = "context"
    timestamp: str = field(default_factory=_now)
    provenance: Dict[str, Any] = field(default_factory=dict)
    reliability: float = 1.0  # 0..1, how much to trust this source


@dataclass
class Hypothesis:
    """One possible explanation for the initial observation."""

    id: str
    statement: str
    prior: float = 0.25
    support: float = 0.25  # current belief mass; sums to ~1 across hypotheses
    supporting: List[str] = field(default_factory=list)
    contradicting: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    predicted_observations: List[str] = field(default_factory=list)


@dataclass
class Experiment:
    """One action that could separate hypotheses (run SQL, check logs, call API...)."""

    id: str
    name: str
    description: str
    tool_name: str
    args: Dict[str, Any] = field(default_factory=dict)
    cost: float = 1.0  # abstract money units; higher = more expensive
    latency_s: float = 1.0
    risk: float = 0.0  # 0..1; 1 = irreversible / unsafe
    expected_gain: float = 0.5  # 0..1 estimate of how much uncertainty it removes
    targets: List[str] = field(default_factory=list)  # hypothesis ids it discriminates


@dataclass
class ExperimentResult:
    experiment: Experiment
    observation: Observation
    ok: bool = True
    error: str = ""


@dataclass
class EvidenceBundle:
    """What HypothesisOS hands back to the host agent."""

    question: str
    conclusion: str
    confidence: float
    abstain: bool
    hypotheses: List[Hypothesis] = field(default_factory=list)
    experiments_run: List[Experiment] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    remaining_uncertainty: float = 1.0
    trace: List[str] = field(default_factory=list)
    total_cost: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Question: {self.question}",
            f"Conclusion: {self.conclusion}",
            f"Confidence: {self.confidence:.2f}  Abstain: {self.abstain}",
            f"Experiments: {len(self.experiments_run)}  Cost: {self.total_cost:.2f}",
            "Hypotheses:",
        ]
        for h in sorted(self.hypotheses, key=lambda x: -x.support):
            lines.append(f"  [{h.support:.2f}] {h.statement}")
        lines.append("Observations:")
        for o in self.observations:
            lines.append(f"  ({o.source}) {o.content}")
        return "\n".join(lines)
