"""Framework adapters: use HypothesisOS without migrating your app."""
from __future__ import annotations

from typing as _t

from ..controller import HypothesisOS


def wrap_agent(agent_fn, investigator: HypothesisOS, trigger=None):
    """Wrap any agent callable: run it, and if `trigger(result)` says the
    answer is uncertain, investigate first and attach the evidence bundle.

    agent_fn(question, context) -> str
    trigger(answer) -> bool (default: always investigate)
    Returns fn(question, context) -> dict(answer, evidence).
    """
    check = trigger or (lambda ans: True)

    def wrapped(question: str, context: str = "") -> dict:
        answer = agent_fn(question, context)
        evidence = None
        if check(answer):
            evidence = investigator.investigate(question, context=context)
        return {"answer": answer, "evidence": evidence}

    return wrapped


def langgraph_node(investigator: HypothesisOS, question_key="question",
                   context_key="context", out_key="evidence"):
    """Build a LangGraph node: state in -> {out_key: EvidenceBundle} out.

    graph.add_node("epistemic_check", langgraph_node(investigator))
    """
    def node(state: dict) -> dict:
        bundle = investigator.investigate(
            state.get(question_key, ""), context=str(state.get(context_key, "")))
        return {out_key: bundle}

    node.__name__ = "epistemic_check"
    return node
