"""Opt-in live test. Burns real API budget; only runs with HYPOTHESISOS_LIVE=1.

    HYPOTHESISOS_LIVE=1 python3 -m pytest tests/test_live.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from live_eval import WORLDS, baseline_answer, load_key  # noqa: E402
from hypothesisos import HypothesisOS, OpenRouterModel  # noqa: E402
import sim_analytics  # noqa: E402

LIVE = os.getenv("HYPOTHESISOS_LIVE") == "1"
pytestmark = pytest.mark.skipif(not LIVE, reason="set HYPOTHESISOS_LIVE=1 to run")


def test_live_demand_drop_investigation_beats_guess():
    """The money case: revenue down could be real or a data problem.
    Investigator must find the truth with evidence; baseline often guesses."""
    sim = sim_analytics.build("demand_drop")
    model = OpenRouterModel(api_key=load_key(),
                            model=os.getenv("HYPOTHESISOS_MODEL",
                                            "nvidia/nemotron-3-super-120b-a12b:free"))
    base = baseline_answer(model, sim["question"], sim["context"])
    inv = HypothesisOS(model=model, tools=sim["tools"], max_experiments=4,
                       confidence_threshold=0.7)
    r = inv.investigate(sim["question"], context=sim["context"])
    top = max(r.hypotheses, key=lambda h: h.support)
    assert "demand" in top.statement.lower(), f"got: {top.statement}\n{r.summary()}"
    assert not r.abstain
    assert 1 <= len(r.experiments_run) <= 4
    print(f"\nbaseline was: {base[:200]}")
    print(f"\ninvestigator: {r.summary()}")
