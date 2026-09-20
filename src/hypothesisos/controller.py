"""The epistemic control loop: Observe -> Hypothesize -> Experiment -> Update -> Act.

The host agent calls `investigate(question, context)` and gets back an
EvidenceBundle. HypothesisOS never answers the user directly; it returns
evidence so the HOST decides (answer / act / escalate / abstain).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .core.types import EvidenceBundle, Experiment, Hypothesis, Observation
from .engines import hypothesis as hyp_engine
from .engines import planner as planner_engine
from .engines import updater as updater_engine
from .models import HeuristicModel
from .tools import Tool


class HypothesisOS:
    """investigator = HypothesisOS(model=model, tools=tools)
    result = investigator.investigate(question, context=context)
    """

    def __init__(self, model=None, tools: Optional[List[Tool]] = None,
                 max_experiments: int = 4, confidence_threshold: float = 0.75,
                 min_gain: float = 0.15, n_hypotheses: int = 4):
        self.model = model or HeuristicModel()
        self.tools: List[Tool] = list(tools or [])
        self.max_experiments = max_experiments
        self.confidence_threshold = confidence_threshold
        self.min_gain = min_gain
        self.n_hypotheses = n_hypotheses

    # -- main entry -----------------------------------------------------
    def investigate(self, question: str, context: str = "",
                    tools: Optional[List[Tool]] = None) -> EvidenceBundle:
        tools = list(tools) if tools is not None else self.tools
        trace: List[str] = []
        observations = [Observation(content=f"{question} | {context}".strip(" |"),
                                    source="initial")]
        trace.append(f"OBSERVE: {question}")

        hypotheses = hyp_engine.generate(question, context, self.model, self.n_hypotheses)
        trace.append("HYPOTHESIZE: " + "; ".join(f"{h.id}={h.statement[:60]}" for h in hypotheses))

        by_tool: Dict[str, Tool] = {t.name: t for t in tools}
        candidates = planner_engine.plan(hypotheses, tools, self.model, question)
        ranked = planner_engine.rank(candidates, hypotheses)

        experiments_run: List[Experiment] = []
        total_cost = 0.0
        used = set()

        for _ in range(self.max_experiments):
            top = self._best(hypotheses)
            if top.support >= self.confidence_threshold:
                trace.append(f"STOP: {top.id} support {top.support:.2f} >= threshold")
                break
            nxt = next((e for e in ranked if e.id not in used and e.expected_gain >= self.min_gain), None)
            if nxt is None:
                trace.append("STOP: no useful experiment remains")
                break
            tool = by_tool.get(nxt.tool_name)
            if tool is None:
                used.add(nxt.id)
                continue
            if tool.risk >= 0.9:
                trace.append(f"SKIP {nxt.id}: risk {tool.risk} too high without human approval")
                used.add(nxt.id)
                continue
            used.add(nxt.id)
            trace.append(f"EXPERIMENT {nxt.id}: {nxt.tool_name} {nxt.args} (targets {nxt.targets})")
            obs = tool.run(nxt.args)
            observations.append(obs)
            total_cost += tool.cost
            experiments_run.append(nxt)
            trace.append(f"OBSERVED ({obs.source}): {obs.content[:160]}")
            hypotheses = updater_engine.update(hypotheses, obs, self.model)
            trace.append("BELIEFS: " + ", ".join(f"{h.id}={h.support:.2f}" for h in hypotheses))

        top = self._best(hypotheses)
        uncertainty = 1.0 - top.support
        abstain = top.support < self.confidence_threshold
        if abstain:
            conclusion = (f"Insufficient evidence: top explanation '{top.statement}' "
                          f"has support {top.support:.2f} below {self.confidence_threshold}. "
                          f"Do not act on the initial observation alone.")
        else:
            conclusion = f"{top.statement} (support {top.support:.2f})"
        trace.append(f"ACT: {'abstain' if abstain else 'conclude'} -> {conclusion[:120]}")
        return EvidenceBundle(
            question=question, conclusion=conclusion, confidence=top.support,
            abstain=abstain, hypotheses=hypotheses, experiments_run=experiments_run,
            observations=observations, remaining_uncertainty=uncertainty,
            trace=trace, total_cost=total_cost)

    @staticmethod
    def _best(hypotheses: List[Hypothesis]) -> Hypothesis:
        return max(hypotheses, key=lambda h: h.support)


Investigator = HypothesisOS  # alias: Investigator(model=..., tools=...)
