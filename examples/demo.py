"""5-minute demo: 'Revenue is down 27%.' Guess vs investigate.

Runs fully offline (no API key). For the live LLM version, set
OPENROUTER_API_KEY and pass --live.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from hypothesisos import HypothesisOS, HeuristicModel
from sim_analytics import build


def main():
    live = "--live" in sys.argv
    world = sys.argv[sys.argv.index("--world") + 1] if "--world" in sys.argv else "etl_failure"
    sim = build(world)
    print(f"Hidden truth (agent CANNOT see this): {sim['truth']}\n")

    if live:
        from hypothesisos import OpenRouterModel
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("set OPENROUTER_API_KEY")
        model = OpenRouterModel(api_key=key,
                                model=os.getenv("HYPOTHESISOS_MODEL",
                                                "nvidia/nemotron-3-super-120b-a12b:free"))
    else:
        model = HeuristicModel()
        print("(offline mode: deterministic heuristic reasoning, no API calls)\n")

    print("--- Ordinary agent (no investigation) ---")
    print("Revenue fell 27%, so customer demand must have declined. Reporting sales drop.\n")

    print("--- HypothesisOS ---")
    inv = HypothesisOS(model=model, tools=sim["tools"],
                       max_experiments=4, confidence_threshold=0.6)
    result = inv.investigate(sim["question"], context=sim["context"])
    for step in result.trace:
        print(" ", step)
    print()
    print(result.summary())


if __name__ == "__main__":
    main()
