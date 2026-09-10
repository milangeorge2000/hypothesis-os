"""Tool interface plus stdlib-only implementations.

A Tool is anything the host agent already has: a function, SQL connection,
HTTP endpoint. HypothesisOS never invents capabilities; it only picks
which of YOUR tools to run next.
"""
from __future__ import annotations

import sqlite3
import urllib.request
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from .core.types import Observation


@dataclass
class Tool:
    name: str
    description: str
    cost: float = 1.0
    latency_s: float = 1.0
    risk: float = 0.0
    # space-separated keywords used by the offline planner/ranker
    keywords: str = ""

    def run(self, args: Dict[str, Any]) -> Observation:
        raise NotImplementedError


@dataclass
class FunctionTool(Tool):
    """Wrap any existing Python callable: fn(dict) -> str."""

    fn: Callable[[Dict[str, Any]], str] = field(default=lambda a: "")

    def run(self, args: Dict[str, Any]) -> Observation:
        try:
            out = self.fn(args)
            return Observation(
                content=str(out), source=self.name,
                provenance={"tool": self.name, "args": args},
            )
        except Exception as exc:  # tool failure is itself an observation
            return Observation(
                content=f"TOOL_ERROR {self.name}: {exc}",
                source=self.name, reliability=0.5,
                provenance={"tool": self.name, "args": args, "error": str(exc)},
            )


@dataclass
class SQLTool(Tool):
    """Read-only SQL over sqlite. Blocks writes to keep experiments safe."""

    connection: Any = None  # sqlite3.Connection

    _FORBIDDEN = ("insert", "update", "delete", "drop", "alter", "create", "replace")

    def run(self, args: Dict[str, Any]) -> Observation:
        query = str(args.get("query", ""))
        lowered = query.strip().lower()
        if not lowered.startswith("select") or any(w in lowered for w in self._FORBIDDEN):
            return Observation(
                content="TOOL_ERROR: only read-only SELECT queries allowed",
                source=self.name, reliability=0.5,
            )
        try:
            cur = self.connection.execute(query)
            rows = cur.fetchmany(50)
            cols = [d[0] for d in (cur.description or [])]
            payload = {"columns": cols, "rows": [list(r) for r in rows]}
            return Observation(
                content=f"SQL result: {json.dumps(payload)[:2000]}",
                source=self.name,
                provenance={"tool": self.name, "query": query},
            )
        except Exception as exc:
            return Observation(
                content=f"TOOL_ERROR {self.name}: {exc}", source=self.name,
                reliability=0.5,
            )


@dataclass
class HTTPTool(Tool):
    """Simple GET tool for status pages / APIs. No auth headers from args."""

    base_url: str = ""

    def run(self, args: Dict[str, Any]) -> Observation:
        path = str(args.get("path", "/"))
        url = self.base_url.rstrip("/") + path
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                body = r.read(4000).decode("utf-8", errors="replace")
            return Observation(
                content=f"GET {path} -> {r.status}: {body[:1500]}",
                source=self.name, provenance={"tool": self.name, "url": url},
            )
        except Exception as exc:
            return Observation(
                content=f"TOOL_ERROR {self.name}: {exc}", source=self.name,
                reliability=0.5,
            )
