"""LangChain adapter (optional dependency).

Both directions, no migration needed:

    from hypothesisos.adapters.langchain import (
        langchain_tools_to_hypothesisos, investigator_as_langchain_tool)

    # your existing LangChain tools become HypothesisOS tools
    investigator = HypothesisOS(model=model,
                                tools=langchain_tools_to_hypothesisos(lc_tools))

    # ...or the investigator becomes one tool in your LangChain agent
    agent = create_react_agent(model, [investigator_as_langchain_tool(investigator)])

Requires `langchain-core` only when these helpers are called.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..controller import HypothesisOS
from ..tools import FunctionTool


def langchain_tools_to_hypothesisos(lc_tools, costs: Dict[str, float] | None = None,
                                    default_cost: float = 1.0) -> List[FunctionTool]:
    """Wrap LangChain `Tool` / `StructuredTool` objects as HypothesisOS tools.

    costs: optional {tool_name: cost} so cheap checks (watermarks, status
    pages) are scheduled before expensive ones. Unknown tools get default_cost.
    """
    costs = costs or {}
    wrapped = []
    for t in lc_tools:
        wrapped.append(FunctionTool(
            name=t.name,
            description=getattr(t, "description", "") or t.name,
            cost=costs.get(t.name, default_cost),
            keywords=getattr(t, "description", "") or "",
            fn=_make_fn(t),
        ))
    return wrapped


def _make_fn(lc_tool):
    def fn(args: Dict[str, Any]) -> str:
        payload: Any = args.get("input", args) if isinstance(args, dict) else args
        invoke = getattr(lc_tool, "invoke", None)
        if callable(invoke):
            try:
                return str(invoke(payload))
            except Exception:
                pass
        run = getattr(lc_tool, "run", None)
        if callable(run):
            return str(run(payload if isinstance(payload, str) else str(payload)))
        raise AttributeError(f"LangChain tool {lc_tool!r} has neither invoke() nor run()")

    return fn


def investigator_as_langchain_tool(investigator: HypothesisOS,
                                   name: str = "investigate",
                                   max_experiments: int | None = None):
    """Expose an investigator as a LangChain StructuredTool.

    Returns the tool's text summary (conclusion + confidence + key
    observations) so the LangChain agent can reason over the evidence.
    """
    try:
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError(
            "investigator_as_langchain_tool needs langchain-core: "
            "pip install langchain-core") from exc

    class _Input(BaseModel):
        question: str = Field(description="The claim or symptom to investigate.")
        context: str = Field(default="",
                             description="Background facts the agent already knows.")

    def _run(question: str, context: str = "") -> str:
        if max_experiments is not None:
            investigator.max_experiments = max_experiments
        bundle = investigator.investigate(question, context=context)
        top = [f"[{h.support:.2f}] {h.statement}" for h in
               sorted(bundle.hypotheses, key=lambda h: -h.support)[:3]]
        obs = [f"({o.source}) {o.content[:200]}" for o in bundle.observations[1:4]]
        return ("\n".join([f"Conclusion: {bundle.conclusion}",
                           f"Confidence: {bundle.confidence:.2f}",
                           f"Abstain: {bundle.abstain}",
                           "Top hypotheses:"] + top + ["Key observations:"] + obs))

    return StructuredTool.from_function(
        func=_run,
        name=name,
        description=("Investigate a claim before acting on it. Use when evidence "
                     "is incomplete, ambiguous, stale, or contradictory. Input is "
                     "a question plus known context; output is an evidence-backed "
                     "conclusion with confidence, or an honest abstention."),
        args_schema=_Input,
    )
