"""Simulated production analytics environment.

Hidden worlds (same symptom, different truth):
  demand_drop   - customers really bought less
  etl_failure   - ingestion stopped 6h early; data is partial
  tz_error      - day window shifted; hours misattributed
  payment_delay - orders exist, payment records missing -> join drops revenue

Agent tools: query_revenue, check_watermark, hourly_breakdown, payment_coverage.
The agent sees only tool outputs, never `hidden_world`.
"""
from __future__ import annotations

import random
import sqlite3
from typing import Dict, List

from hypothesisos import FunctionTool


def build(world: str, seed: int = 7, day: str = "2026-09-26"):
    rnd = random.Random(seed)
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE orders (id INTEGER, customer TEXT, amount REAL, hour INTEGER, day TEXT)")
    con.execute("CREATE TABLE pipeline (job TEXT, watermark_day TEXT, watermark_hour INTEGER, status TEXT)")
    con.execute("CREATE TABLE payments (order_id INTEGER, status TEXT)")

    base_customers, base_n = 200, 400
    drop = world == "demand_drop"
    n_today = 290 if drop else base_n
    customers_today = 150 if drop else base_customers

    # realistic daily pattern: lunch + dinner peaks (so a timezone shift is visible)
    import math
    hour_w = [1 + 2 * math.exp(-((h - 12) / 3) ** 2) + 2.5 * math.exp(-((h - 19) / 2.5) ** 2)
              for h in range(24)]

    def pick_hour():
        return rnd.choices(range(24), weights=hour_w)[0]

    # yesterday: full normal day
    for i in range(base_n):
        con.execute("INSERT INTO orders VALUES (?,?,?,?,?)",
                    (10000 + i, f"cust_{rnd.randrange(base_customers)}",
                     round(rnd.uniform(20, 200), 2), pick_hour(), "2026-09-25"))
    # today: depends on world
    cutoff = 18 if world == "etl_failure" else 24  # ingestion stopped at 18:00
    for i in range(n_today):
        hr = pick_hour()
        if hr >= cutoff:
            continue  # rows after cutoff never ingested
        stored_hr = (hr + 5) % 24 if world == "tz_error" else hr
        con.execute("INSERT INTO orders VALUES (?,?,?,?,?)",
                    (20000 + i, f"cust_{rnd.randrange(customers_today)}",
                     round(rnd.uniform(20, 200), 2), stored_hr, day))
    con.execute("INSERT INTO pipeline VALUES (?,?,?,?)",
                ("etl_orders", day, cutoff if world == "etl_failure" else 24, "ok"))
    for i in range(base_n):  # yesterday's payments all arrived and matched
        con.execute("INSERT INTO payments VALUES (?,?)", (10000 + i, "paid"))
    for i in range(n_today):
        if world == "payment_delay" and i % 3 == 0:
            continue  # payment record never arrived (independent of orders)
        status = "missing" if (world == "payment_delay") else "paid"
        con.execute("INSERT INTO payments VALUES (?,?)", (20000 + i, status))
    con.commit()

    truth = {
        "demand_drop": "genuine demand decline",
        "etl_failure": "ETL pipeline delay (data incomplete)",
        "tz_error": "timezone-boundary error",
        "payment_delay": "missing payment records",
    }[world]

    def q(label: str, sql: str, summary: str = "") -> str:
        cur = con.execute(sql)
        rows = cur.fetchmany(20)
        base = (f"{label} columns={[d[0] for d in (cur.description or [])]} "
                f"rows={[list(r) for r in rows]}")
        return base + (f" {summary}" if summary else "")

    def hourly_summary() -> str:
        t = dict(con.execute("SELECT hour, COUNT(*) FROM orders WHERE day='2026-09-26' "
                             "GROUP BY hour").fetchall())
        y = dict(con.execute("SELECT hour, COUNT(*) FROM orders WHERE day='2026-09-25' "
                             "GROUP BY hour").fetchall())
        if not t:
            return "SUMMARY: no orders ingested at all today"
        last = max(t)
        if last < 22 and len(y) == 24:
            return (f"SUMMARY: row counts drop after a cutoff: hourly distribution ends abruptly "
                    f"after hour {last}; hours {last+1}-23 have zero rows vs full day yesterday")
        if len(t) == 24 and y:
            # cross-correlation over lags: systematic shift vs same shape?
            def corr(lag):
                return sum(t.get((h + lag) % 24, 0) * y.get(h, 0) for h in range(24))
            best_lag = max(range(-8, 9), key=lambda l: corr(l))
            if abs(best_lag) >= 2:
                return (f"SUMMARY: hourly distribution shifted vs yesterday "
                        f"(best alignment at lag {best_lag}h; boundary rows cluster at midnight)")
            return "SUMMARY: hourly distribution matches prior days; complete 24 hours, same shape"
        return "SUMMARY: hourly distribution looks normal"

    def watermark_summary() -> str:
        wh = con.execute("SELECT watermark_hour FROM pipeline").fetchone()[0]
        if wh < 24:
            return (f"SUMMARY: ingestion watermark lags scheduled time: stuck at hour {wh}, "
                    f"{24-wh} hours behind schedule, ingestion stopped early")
        return "SUMMARY: watermark is current and row counts are normal; ingestion complete"

    def payment_summary() -> str:
        n, paid = con.execute(
            "SELECT COUNT(*), SUM(CASE WHEN payments.status='paid' THEN 1 ELSE 0 END) "
            "FROM orders LEFT JOIN payments ON payments.order_id=orders.id "
            "WHERE day='2026-09-26'").fetchone()
        pct = (paid or 0) / max(1, n)
        if pct > 0.98:
            return f"SUMMARY: every order has a matching payment; full coverage {paid}/{n}"
        return (f"SUMMARY: many of today's orders lack matching payments: "
                f"only {100*pct:.0f}% coverage ({paid}/{n} paid); unmatched payment ids present")

    def revenue_summary() -> str:
        rows = dict(con.execute(
            "SELECT day, ROUND(SUM(amount),2) FROM orders JOIN payments "
            "ON payments.order_id=orders.id WHERE payments.status='paid' "
            "GROUP BY day").fetchall())
        cust = dict(con.execute(
            "SELECT day, COUNT(DISTINCT customer) FROM orders GROUP BY day").fetchall())
        rev_y, rev_t = rows.get("2026-09-25", 0), rows.get("2026-09-26", 0)
        drop = (rev_y - rev_t) / max(1, rev_y)
        cy, ct = cust.get("2026-09-25", 0), cust.get("2026-09-26", 0)
        cust_drop = (cy - ct) / max(1, cy)
        cust_line = ("distinct customer count is also down sharply"
                     if cust_drop >= 0.20 else
                     "customer counts flat while revenue fell (within noise)")
        return (f"SUMMARY: revenue {rev_t} vs {rev_y} yesterday ({100*drop:.0f}% drop); "
                f"{cust_line}: {ct} vs {cy} yesterday")

    tools = [
        FunctionTool(name="query_revenue", cost=1.0, latency_s=1.0,
                     description="Total revenue per day from orders joined to paid payments.",
                     keywords="revenue sales totals demand orders payments join",
                     fn=lambda a: q("REVENUE daily totals",
                                    "SELECT day, ROUND(SUM(amount),2) AS revenue, COUNT(*) AS n FROM orders "
                                    "JOIN payments ON payments.order_id = orders.id "
                                    "WHERE payments.status='paid' GROUP BY day ORDER BY day",
                                    revenue_summary())),
        FunctionTool(name="check_watermark", cost=0.2, latency_s=0.3,
                     description="Pipeline ingestion watermark: how far ingestion got today.",
                     keywords="watermark pipeline ingestion freshness etl stopped delayed hours",
                     fn=lambda a: q("PIPELINE watermark status",
                                    "SELECT * FROM pipeline", watermark_summary())),
        FunctionTool(name="hourly_breakdown", cost=0.5, latency_s=0.5,
                     description="Orders per hour today vs distribution shape.",
                     keywords="hourly distribution midnight timezone shifted pattern hours",
                     fn=lambda a: q("HOURLY per-hour order counts",
                                    "SELECT hour, COUNT(*) c FROM orders WHERE day='2026-09-26' "
                                    "GROUP BY hour ORDER BY hour", hourly_summary())),
        FunctionTool(name="payment_coverage", cost=0.5, latency_s=0.5,
                     description="Share of today orders with matching paid payment.",
                     keywords="payment coverage join missing unmatched paid orders",
                     fn=lambda a: q("PAYMENT join coverage",
                                    "SELECT COUNT(*) AS orders, "
                                    "SUM(CASE WHEN payments.status='paid' THEN 1 ELSE 0 END) AS paid "
                                    "FROM orders LEFT JOIN payments ON payments.order_id=orders.id "
                                    "WHERE day='2026-09-26'", payment_summary())),
    ]
    question = "Revenue is down ~27% today. Why?"
    context = ("Dashboard: revenue today vs yesterday -27%. "
               "On-call suspects a real sales drop. Verify before reporting.")
    return {"tools": tools, "question": question, "context": context,
            "hidden_world": world, "truth": truth, "con": con}
