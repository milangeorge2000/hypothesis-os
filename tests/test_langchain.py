"""LangChain adapter tests. Skipped when langchain-core isn't installed."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

lc = pytest.importorskip("langchain_core")

from hypothesisos import HeuristicModel, HypothesisOS  # noqa: E402
from hypothesisos.adapters.langchain import (  # noqa: E402
    investigator_as_langchain_tool,
    langchain_tools_to_hypothesisos,
)
import sim_analytics  # noqa: E402


def _lc_tools(sim):
    from langchain_core.tools import Tool
    return [Tool(name=t.name, description=t.description,
                 func=lambda x, t=t: t.run({}).content)
            for t in sim["tools"]]


def test_langchain_tools_become_hypothesisos_tools():
    sim = sim_analytics.build("etl_failure")
    wrapped = langchain_tools_to_hypothesisos(
        _lc_tools(sim), costs={"check_watermark": 0.2})
    assert [t.name for t in wrapped] == [t.name for t in sim["tools"]]
    assert wrapped[1].cost == 0.2
    obs = wrapped[1].run({})
    assert "watermark" in obs.content.lower()


def test_investigate_through_wrapped_langchain_tools():
    sim = sim_analytics.build("payment_delay")
    inv = HypothesisOS(model=HeuristicModel(),
                       tools=langchain_tools_to_hypothesisos(_lc_tools(sim)),
                       max_experiments=4, confidence_threshold=0.55)
    r = inv.investigate(sim["question"], context=sim["context"])
    top = max(r.hypotheses, key=lambda h: h.support)
    assert "payment" in top.statement.lower()


def test_investigator_as_langchain_tool():
    sim = sim_analytics.build("etl_failure")
    inv = HypothesisOS(model=HeuristicModel(),
                       tools=langchain_tools_to_hypothesisos(_lc_tools(sim)),
                       max_experiments=4, confidence_threshold=0.55)
    tool = investigator_as_langchain_tool(inv)
    assert tool.name == "investigate"
    out = tool.invoke({"question": sim["question"], "context": sim["context"]})
    assert "Conclusion:" in out and "Confidence:" in out
    assert "etl" in out.lower() or "pipeline" in out.lower()


def test_helpful_error_without_langchain(monkeypatch):
    import hypothesisos.adapters.langchain as lc_mod
    monkeypatch.setitem(sys.modules, "langchain_core", None)
    monkeypatch.setitem(sys.modules, "langchain_core.tools", None)
    inv = HypothesisOS(model=HeuristicModel(), tools=[])
    with pytest.raises(RuntimeError, match="pip install langchain-core"):
        investigator_as_langchain_tool(inv)
    # module object itself still usable for the other direction
    assert lc_mod.langchain_tools_to_hypothesisos([]) == []
