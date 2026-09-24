"""Live eval: baseline (LLM, no tools) vs HypothesisOS (same LLM + tools).

Key loading (never printed): $OPENROUTER_API_KEY, else
../../drift-prover/config.yaml -> llm.openrouter_api_key (sibling checkout).
Run:  python3 examples/live_eval.py [--worlds etl_failure,demand_drop] [--model x-ai/grok-4.3]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hypothesisos import HypothesisOS, OpenRouterModel  # noqa: E402
import sim_analytics  # noqa: E402
import sim_ops  # noqa: E402


def load_key() -> str:
    if os.getenv("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    for cand in (
        os.path.join(os.path.dirname(__file__), "..", "..", "drift-prover", "config.yaml"),
        os.path.expanduser("~/.config/hypothesisos/key"),
    ):
        if os.path.exists(cand):
            import re
            text = open(cand).read()
            m = re.search(r"openrouter_api_key:\s*(\S+)", text)
            if m and "replace" not in m.group(1):
                return m.group(1).strip().strip("'\"")
    raise SystemExit("No OpenRouter key: set OPENROUTER_API_KEY")


WORLDS = [
    ("analytics", "etl_failure", ["etl", "pipeline", "ingestion", "incomplete", "partial"]),
    ("analytics", "demand_drop", ["demand", "genuine", "customers bought less"]),
    ("analytics", "tz_error", ["timezone", "time zone", "boundary", "misattributed", "shift"]),
    ("analytics", "payment_delay", ["payment", "missing payment", "join"]),
    ("ops", "deploy_regression", ["deploy", "regression", "release", "rollback"]),
    ("ops", "db_exhaustion", ["database", "connection", "pool", "exhaustion", "slow quer"]),
    ("ops", "traffic_surge", ["traffic", "surge", "volume", "scale"]),
    ("ops", "downstream_fail", ["downstream", "dependency", "payments-svc", "failover"]),
]


def baseline_answer(model, question, context) -> str:
    return model.complete(
        "You are a senior on-call engineer. Answer in two sentences max.",
        f"{question}\nContext: {context}\nWhat is the most likely cause? Be decisive.")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", default="",
                    help="comma list like etl_failure,demand_drop (default: all)")
    ap.add_argument("--model", default="nvidia/nemotron-3-super-120b-a12b:free")
    ap.add_argument("--max-exp", type=int, default=4)
    args = ap.parse_args()

    only = set(w.strip() for w in args.worlds.split(",") if w.strip())
    model = OpenRouterModel(api_key=load_key(), model=args.model)
    rows = []
    for domain, world, keys in WORLDS:
        if only and world not in only:
            continue
        sim = (sim_analytics.build(world) if domain == "analytics"
               else sim_ops.build(world))
        base = baseline_answer(model, sim["question"], sim["context"])
        inv = HypothesisOS(model=model, tools=sim["tools"],
                           max_experiments=args.max_exp,
                           confidence_threshold=0.7)
        try:
            r = inv.investigate(sim["question"], context=sim["context"])
            top = max(r.hypotheses, key=lambda h: h.support)
            conc = r.conclusion
            n_exp = len(r.experiments_run)
            abst = r.abstain
        except Exception as exc:  # API hiccup: record, don't crash the table
            conc, top, n_exp, abst = f"ERROR: {type(exc).__name__}", None, 0, True
        low = (conc + " " + base).lower()
        hit_h = any(k in conc.lower() for k in keys)
        hit_b = any(k in base.lower() for k in keys)
        rows.append((world, sim["truth"], hit_b, hit_h, n_exp, abst, base, conc))
        print(f"[{world}] truth={sim['truth']}")
        print(f"  baseline : {'HIT' if hit_b else 'MISS'} {base[:160]}")
        print(f"  hypoOS   : {'HIT' if hit_h else 'MISS'} (exp={n_exp} abst={abst}) {conc[:200]}")
        print()

    nb = sum(1 for r in rows if r[2])
    nh = sum(1 for r in rows if r[3])
    print(f"BASELINE {nb}/{len(rows)}  |  HYPOTHESISOS {nh}/{len(rows)}")
    print(f"avg experiments: {sum(r[4] for r in rows)/max(1,len(rows)):.1f}")


if __name__ == "__main__":
    main()
