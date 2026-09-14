"""Hypothesis generation: competing explanations for one observation.

LLM path when a real model is supplied; deterministic templates otherwise
(and as a safety net when the LLM returns garbage). Goal: 3-5 DISTINCT,
testable explanations, never just rephrasings of the question.
"""
from __future__ import annotations

from typing import List

from ..core.types import Hypothesis
from ..models import HeuristicModel, complete_json

_SYSTEM = (
    "You generate competing hypotheses like a careful investigator. "
    "Given one observation, list 3-5 DISTINCT possible explanations, including "
    "at least one boring/measurement explanation (pipeline delay, stale data, "
    "timezone or logging error) and at most one genuine-change explanation. "
    "For each: statement, what evidence would support it, what would refute it, "
    "what is missing, and one predicted observation. "
    "Return JSON: {\"hypotheses\": [{\"statement\": ..., \"supporting\": [...], "
    "\"contradicting\": [...], \"missing\": [...], \"predicted\": [...]}]}"
)

# Deterministic fallback templates keyed by domain keywords.
_TEMPLATES = [
    (("revenue", "sales", "orders", "etl", "pipeline", "ingest"), [
        ("Genuine demand decline: customers bought less in the measured period.",
         ["orders table shows fewer distinct customers", "payments match orders"],
         ["revenue recomputation matches dashboard", "customer counts flat while revenue fell"],
         ["customer-level breakdown for today"],
         ["distinct customer count is also down sharply"]),
        ("ETL/pipeline delay: ingestion stopped early so today's data is partial.",
         ["ingestion watermark lags scheduled time", "row counts drop after a cutoff"],
         ["watermark is current and row counts are normal"],
         ["pipeline watermark timestamp"],
         ["ingestion watermark lags scheduled time by hours"]),
        ("Timezone-boundary error: the day window shifted so hours are misattributed.",
         ["boundary rows cluster at midnight", "hourly distribution shifted"],
         ["hourly distribution matches prior days"],
         ["hourly order distribution"],
         ["hourly distribution shifted vs yesterday"]),
        ("Missing payment records: orders exist but payment join drops rows.",
         ["orders count normal but paid count low", "unmatched payment ids"],
         ["every order has a matching payment"],
         ["order-to-payment join coverage"],
         ["many of today's orders lack matching payments"]),
    ]),
    (("latency", "slow", "p99", "api", "timeout", "error rate"), [
        ("Recent deployment introduced a regression.",
         ["deploy timestamp precedes latency rise", "rollback restores latency"],
         ["no deploy near the onset"],
         ["deploy log around onset time"],
         ["a deploy finished minutes before latency rose"]),
        ("Database connection exhaustion / slow queries.",
         ["pool saturation with wait queues", "slow query log entries"],
         ["pool healthy and query times flat"],
         ["db pool stats and slow queries"],
         ["pool saturation with wait queues and slow queries"]),
        ("Traffic surge: more requests, same capacity.",
         ["request rate up proportionally", "autoscaling lagging"],
         ["request rate flat at baseline"],
         ["requests-per-minute vs yesterday"],
         ["request volume is sharply higher than baseline"]),
        ("Downstream dependency failure causing queueing/timeouts.",
         ["downstream error rate up", "timeout share up"],
         ["all downstreams healthy"],
         ["downstream health + timeout breakdown"],
         ["one downstream shows errors matching the onset"]),
    ]),
]


def _fallback(question: str, context: str) -> List[Hypothesis]:
    text = (question + " " + context).lower()
    for keywords, hyps in _TEMPLATES:
        if any(k in text for k in keywords):
            out = []
            for i, (stmt, sup, con, miss, pred) in enumerate(hyps):
                out.append(Hypothesis(
                    id=f"H{i+1}", statement=stmt, prior=1 / len(hyps),
                    support=1 / len(hyps), supporting=list(sup),
                    contradicting=list(con), missing=list(miss),
                    predicted_observations=list(pred)))
            return out
    # domain-free generic set
    generic = [
        ("The observed change is real and caused by the most obvious driver.",
         ["direct measurement confirms the change"], ["independent source disagrees"],
         ["independent confirmation"], ["a second source shows the same change"]),
        ("The data is incomplete: collection stopped early or is delayed.",
         ["freshness check shows staleness"], ["freshness check is current"],
         ["freshness/source timestamp"], ["the source timestamp lags expectations"]),
        ("The measurement is wrong: window, timezone, join, or filter error.",
         ["recomputed metric differs"], ["recomputation matches"],
         ["independent recomputation"], ["recomputing with fixed window changes the number"]),
        ("A coupled upstream system changed and propagated here.",
         ["upstream change precedes this symptom"], ["upstreams all healthy"],
         ["upstream change log"], ["an upstream event matches the onset time"]),
    ]
    return [Hypothesis(id=f"H{i+1}", statement=s, prior=0.25, support=0.25,
                       supporting=list(a), contradicting=list(b),
                       missing=list(c), predicted_observations=list(d))
            for i, (s, a, b, c, d) in enumerate(generic)]


def generate(question: str, context: str, model, n: int = 4) -> List[Hypothesis]:
    if isinstance(model, HeuristicModel):
        return _fallback(question, context)[:n]
    try:
        obj = complete_json(
            model, _SYSTEM,
            f"Observation: {question}\nKnown context: {context}\nList {n} hypotheses.")
        items = obj.get("hypotheses", [])[:n]
        if len(items) < 2:
            raise ValueError("too few hypotheses")
        total = len(items)
        return [Hypothesis(
            id=f"H{i+1}", statement=str(h.get("statement", "unknown")),
            prior=1 / total, support=1 / total,
            supporting=[str(x) for x in h.get("supporting", [])],
            contradicting=[str(x) for x in h.get("contradicting", [])],
            missing=[str(x) for x in h.get("missing", [])],
            predicted_observations=[str(x) for x in h.get("predicted", [])],
        ) for i, h in enumerate(items)]
    except Exception:
        return _fallback(question, context)[:n]
