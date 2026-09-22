"""Simulated ops environment: API p99 latency spiked 4x.

Hidden worlds: deploy_regression | db_exhaustion | traffic_surge | downstream_fail
Tools: check_deploys, check_db, check_traffic, check_downstream.
"""
from __future__ import annotations

from hypothesisos import FunctionTool


def build(world: str):
    data = {
        "deploy_regression": {
            "deploys": "a deploy finished minutes before latency rose: "
                       "14:02 deploy api v2.14.1 finished; 14:09 p99 180ms->720ms; deploy timestamp precedes latency rise",
            "db": "pool 40/100 used, waits 0ms, slow queries none: pool healthy and query times flat",
            "traffic": "rpm 1.2k, request rate flat at baseline 1.2k; autoscaling fine, CPU 40%",
            "downstream": "all downstreams healthy, error rate 0.01%"},
        "db_exhaustion": {
            "deploys": "last deploy 6 days ago; no deploy near the onset",
            "db": "pool saturation with wait queues and slow queries: pool 99/100 used, "
                  "waits 450ms, 12 slow queries over 2s on orders table",
            "traffic": "rpm 1.25k, request rate flat at baseline 1.2k",
            "downstream": "all downstreams healthy"},
        "traffic_surge": {
            "deploys": "last deploy 6 days ago; no deploy near the onset",
            "db": "pool 60/100, waits 5ms: pool healthy and query times flat",
            "traffic": "request volume is sharply higher than baseline: rpm 4.8k vs 1.2k; "
                       "autoscaling lagging, CPU 95%",
            "downstream": "all downstreams healthy, retries up slightly"},
        "downstream_fail": {
            "deploys": "last deploy 6 days ago; no deploy near the onset",
            "db": "pool 45/100, waits 2ms: pool healthy and query times flat",
            "traffic": "rpm 1.2k, request rate flat at baseline",
            "downstream": "one downstream shows errors matching the onset: payments-svc "
                          "error rate 18% since 14:05, timeouts 22% of p99"},
    }[world]
    truth = {
        "deploy_regression": "recent deployment regression",
        "db_exhaustion": "database connection exhaustion",
        "traffic_surge": "traffic surge",
        "downstream_fail": "downstream service failure",
    }[world]

    def mk(name, text, kw):
        return FunctionTool(name=name, cost=0.5 if name != "check_deploys" else 0.2,
                            latency_s=0.4, description=f"Ops check: {name}. {text[:60]}",
                            keywords=kw, fn=lambda a, t=text: t)

    tools = [
        mk("check_deploys", data["deploys"], "deploy release regression version shipped onset"),
        mk("check_db", data["db"], "database pool connections waits slow queries exhaustion"),
        mk("check_traffic", data["traffic"], "traffic requests rpm surge volume autoscaling cpu"),
        mk("check_downstream", data["downstream"], "downstream dependency timeout errors failure queueing"),
    ]
    return {"tools": tools,
            "question": "API p99 latency spiked 4x (180ms -> 720ms) at ~14:09. Why?",
            "context": "Alert fired 14:12. Last known good 14:00. Decide: rollback, scale, failover, or wait?",
            "hidden_world": world, "truth": truth}
