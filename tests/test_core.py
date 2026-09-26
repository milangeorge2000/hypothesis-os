"""Offline deterministic tests: no network, HeuristicModel only."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from hypothesisos import (  # noqa: E402
    EvidenceBundle,
    FunctionTool,
    HypothesisOS,
    HeuristicModel,
    SQLTool,
)
from hypothesisos.engines import planner as planner_engine  # noqa: E402
import sim_analytics  # noqa: E402
import sim_ops  # noqa: E402


def _investigate(world, builder, threshold=0.55, max_exp=4, seed=7):
    sim = builder(world) if builder is sim_ops.build else builder(world, seed=seed)
    inv = HypothesisOS(model=HeuristicModel(), tools=sim["tools"],
                       max_experiments=max_exp, confidence_threshold=threshold)
    return sim, inv.investigate(sim["question"], context=sim["context"])


def test_analytics_all_worlds():
    keys = {"etl_failure": "etl", "demand_drop": "demand",
            "tz_error": "timezone", "payment_delay": "payment"}
    for world, key in keys.items():
        for seed in (7, 11, 42):
            sim, r = _investigate(world, sim_analytics.build, seed=seed)
            top = max(r.hypotheses, key=lambda h: h.support)
            assert key in top.statement.lower(), \
                f"{world} seed={seed}: top={top.statement} (truth={sim['truth']})"
            assert isinstance(r, EvidenceBundle)
            assert len(r.experiments_run) >= 1


def test_ops_all_worlds():
    keys = {"deploy_regression": "deploy", "db_exhaustion": "database",
            "traffic_surge": "traffic", "downstream_fail": "downstream"}
    for world, key in keys.items():
        sim, r = _investigate(world, sim_ops.build)
        top = max(r.hypotheses, key=lambda h: h.support)
        assert key in top.statement.lower(), \
            f"{world}: top={top.statement} (truth={sim['truth']})"


def test_abstains_when_nothing_distinguishes():
    sim = sim_analytics.build("etl_failure")
    inv = HypothesisOS(model=HeuristicModel(), tools=sim["tools"],
                       max_experiments=0, confidence_threshold=0.99)
    r = inv.investigate(sim["question"], context=sim["context"])
    assert r.abstain is True
    assert "Insufficient evidence" in r.conclusion


def test_risky_tools_skipped():
    sim = sim_analytics.build("etl_failure")
    dangerous = FunctionTool(
        name="drop_table", description="Drops a table.", risk=0.95,
        keywords="drop delete everything", fn=lambda a: "dropped")
    inv = HypothesisOS(model=HeuristicModel(), tools=sim["tools"] + [dangerous],
                       max_experiments=10, confidence_threshold=0.99)
    r = inv.investigate(sim["question"], context=sim["context"])
    assert all(e.tool_name != "drop_table" for e in r.experiments_run)
    assert any("risk" in t for t in r.trace)


def test_cheap_informative_tool_goes_first():
    sim = sim_analytics.build("etl_failure")
    inv = HypothesisOS(model=HeuristicModel(), tools=sim["tools"],
                       max_experiments=1, confidence_threshold=0.99)
    r = inv.investigate(sim["question"], context=sim["context"])
    assert r.experiments_run[0].tool_name == "check_watermark"


def test_sql_tool_is_read_only():
    import sqlite3
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE t (a INTEGER)")
    tool = SQLTool(name="sql", description="sql", connection=con)
    obs = tool.run({"query": "DROP TABLE t"})
    assert "TOOL_ERROR" in obs.content


def test_ranker_prefers_cheap_splits():
    from hypothesisos.core.types import Experiment, Hypothesis
    hyps = [Hypothesis(id=f"H{i}", statement=f"s{i}") for i in range(4)]
    exps = [
        Experiment(id="E1", name="pricey", description="d", tool_name="t",
                   cost=10.0, expected_gain=0.9, targets=["H1"]),
        Experiment(id="E2", name="cheap", description="d", tool_name="t",
                   cost=0.1, expected_gain=0.7, targets=["H1", "H2"]),
    ]
    assert planner_engine.rank(exps, hyps)[0].id == "E2"
