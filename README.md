# HypothesisOS

**Give AI agents the ability to investigate what they don't know instead of guessing.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE)
[![No heavy deps](https://img.shields.io/badge/deps-stdlib%20only-lightgrey)]()

LLMs are good at reasoning over information they already have. Reasoning cannot recover information that is absent from the context. When evidence is incomplete, agents still answer — confidently. HypothesisOS adds the missing loop:

```
Observe → Hypothesize → Experiment → Update → Act
```

It is **not an agent framework**. You keep your agent, model, prompts, memory, and tools. When a conclusion needs evidence you don't have yet, you call one function — HypothesisOS investigates with *your* tools and returns a structured evidence bundle. Your agent decides what to do with it.

```python
from hypothesisos import HypothesisOS, create_model

model = create_model()  # auto-detects OPENAI/ANTHROPIC/GEMINI/GROQ/OPENROUTER key
investigator = HypothesisOS(model=model, tools=tools)  # YOUR tools
result = investigator.investigate("Revenue is down 27%. Why?", context=context)

print(result.conclusion)   # what the evidence supports — or an honest abstention
print(result.confidence)   # 0..1 top-hypothesis support
print(result.abstain)      # True when evidence isn't enough: don't act
```

## The difference

```
Ordinary agent:  evidence → reason → answer            (even when evidence is thin)

HypothesisOS:    evidence → competing hypotheses → cheapest discriminating
                 check → run YOUR tool → update beliefs → repeat until
                 confident or budget spent → evidence bundle back to YOUR agent
```

## Install

```bash
pip install hypothesisos            # once published
# or from source:
git clone git@github.com:milangeorge2000/hypothesis-os.git
cd hypothesis-os && pip install -e .
```

Requirements: Python 3.9+, nothing else at runtime (`certifi` for TLS). No Postgres, Redis, or servers.

## 5-minute demo (no API key needed)

```bash
python3 examples/demo.py                      # ETL-failure world, fully offline
python3 examples/demo.py --world demand_drop  # genuine-drop world
python3 examples/demo.py --world tz_error
python3 examples/demo.py --world payment_delay
```

Expected every time: the ordinary agent blames sales; HypothesisOS finds the real cause. Sample trace:

```
EXPERIMENT E2: check_watermark {} (targets ['H2', 'H3'])
OBSERVED (check_watermark): ... ingestion watermark lags scheduled time ...
BELIEFS: H1=0.19, H2=0.49, H3=0.19, H4=0.14
EXPERIMENT E3: hourly_breakdown {} ...
BELIEFS: H1=0.08, H2=0.77, H3=0.09, H4=0.06
ACT: conclude -> ETL/pipeline delay: ingestion stopped early ...
```

## Model providers

One-line setup for every major API. Keys come from the environment and are never logged. All clients are stdlib-only with retry on 429/transient 5xx.

| Provider | `create_model(...)` | Env vars |
|---|---|---|
| OpenAI | `create_model("openai")` | `OPENAI_API_KEY`, `OPENAI_MODEL` (default `gpt-4o-mini`), `OPENAI_BASE_URL` |
| Anthropic | `create_model("anthropic")` | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` |
| Gemini | `create_model("gemini")` | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`), `GEMINI_MODEL` (default `gemini-2.5-flash`) |
| Groq | `create_model("groq")` | `GROQ_API_KEY`, `GROQ_MODEL` (default `llama-3.3-70b-versatile`) |
| OpenRouter | `create_model("openrouter")` | `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` |
| Ollama (local) | `create_model("ollama")` | `OLLAMA_HOST` (default `http://localhost:11434`), `OLLAMA_MODEL` |
| Any OpenAI-compatible proxy | `create_model("openai", base_url="http://...")` | — |
| Offline (no network) | `create_model("heuristic")` | — |

```python
from hypothesisos import HypothesisOS, create_model

# zero-config: picks up whatever key is set, offline mode if none
investigator = HypothesisOS.from_env(tools=tools)
result = investigator.investigate("API p99 spiked 4x. Why?")

# explicit
model = create_model("anthropic", model="claude-sonnet-4-20250514")
model = create_model("openai", base_url="http://localhost:8000/v1", api_key="sk-...")
```

**Bring your own model:** anything with `.complete(system, user, max_tokens=800) -> str` works — LangChain, LiteLLM, a local server. No adapter code needed.

## Tools (yours, wrapped)

HypothesisOS never invents capabilities — it only chooses which of *your* tools runs next.

```python
from hypothesisos import FunctionTool, SQLTool, HTTPTool

tools = [
    FunctionTool(name="check_watermark", cost=0.2,   # cheap → tried first
                 description="Pipeline ingestion watermark: how far ingestion got.",
                 keywords="watermark ingestion etl freshness",
                 fn=lambda args: db.query("SELECT * FROM pipeline")),
    SQLTool(name="warehouse", description="Read-only analytics DB.",
            cost=1.0, connection=sqlite_conn),        # writes rejected
    HTTPTool(name="status", description="Status page.",
             base_url="https://status.example.com"),  # simple GETs
]
```

`cost` / `latency_s` / `risk` drive scheduling: cheap, safe checks that split hypotheses run first. Tools with `risk >= 0.9` are skipped without human approval.

## Use with your framework

```python
# LangGraph: an epistemic node before consequential decisions
from hypothesisos.adapters import langgraph_node
graph.add_node("epistemic_check", langgraph_node(investigator))

# Any custom agent: investigate only when the answer looks uncertain
from hypothesisos.adapters import wrap_agent
safe_agent = wrap_agent(my_agent, investigator,
                        trigger=lambda ans: "probably" in ans.lower())
out = safe_agent("Revenue is down 27%. Why?")
# {"answer": ..., "evidence": EvidenceBundle(...)}

# Direct call (works with MCP tools, SQL agents, coding agents, ...)
bundle = investigator.investigate(question, context=context, tools=extra_tools)
```

## API reference

**`HypothesisOS(model, tools, max_experiments=4, confidence_threshold=0.75, min_gain=0.15, n_hypotheses=4)`**
**`HypothesisOS.from_env(tools, provider=None)`** — auto-detect model from env.

**`investigate(question, context="") → EvidenceBundle`:**

| Field | Meaning |
|---|---|
| `conclusion` | Top explanation, or "Insufficient evidence…" when abstaining |
| `confidence` | Support of the top hypothesis (0..1) |
| `abstain` | `True` when nothing clears the threshold — don't act |
| `hypotheses` | Each with `statement`, `support`, `supporting`, `contradicting`, `missing`, `predicted_observations` |
| `experiments_run` / `observations` | What was tried, what came back, with provenance |
| `remaining_uncertainty` | `1 − top support` |
| `trace` | Human-readable Observe→…→Act log |
| `total_cost` | Sum of tool costs |

The loop stops when a hypothesis clears `confidence_threshold`, the experiment budget is spent, no experiment clears `min_gain`, or only risky tools remain.

## Measured results

Live model via OpenRouter, 8 hidden worlds (4 analytics + 4 ops), same model for both columns:

| hidden truth | baseline (LLM, no tools) | HypothesisOS |
|---|---|---|
| ETL stopped 6h early | hedged guess | ✅ 2 experiments, 0.85 |
| genuine demand fall | ❌ blamed checkout bug | ✅ 3 experiments, 0.80 |
| timezone shift | ❌ wrong | ✅ 3 experiments, 0.71 |
| missing payments | ❌ blamed pricing change | ✅ 3 experiments, 0.84 |
| bad deploy | correct (obvious) | ✅ 2 experiments, 0.74 |
| pool exhausted | partially right | ✅ 3 experiments, 0.74 |
| 4x traffic | ❌ blamed deploy | ✅ 3 experiments, 0.73 |
| downstream down | ❌ blamed deploy | ✅ 4 experiments, 0.78 |

Baseline **3/8** → HypothesisOS **8/8**. The baseline fails the same way every time: a confident, plausible, wrong answer. Offline heuristic mode scores 8/8 across seeds with zero API calls.

Reproduce: `python3 examples/live_eval.py` (needs `OPENROUTER_API_KEY`).

## Tests

```bash
pytest tests/test_core.py tests/test_providers.py   # offline, deterministic
HYPOTHESISOS_LIVE=1 pytest tests/test_live.py        # opt-in, spends API budget
```

## Project layout

```
src/hypothesisos/
  controller.py      the Observe→…→Act loop (investigate, from_env)
  core/types.py      Observation, Hypothesis, Experiment, EvidenceBundle
  models.py          Model protocol + offline HeuristicModel
  providers.py       OpenAI, Anthropic, Gemini, Groq, OpenRouter, Ollama, create_model
  tools.py           Tool, FunctionTool, SQLTool, HTTPTool
  engines/           hypothesis generation, experiment planning + ranking, belief update
  adapters/          wrap_agent(), langgraph_node()
examples/            demo.py, sim_analytics.py, sim_ops.py, live_eval.py
tests/               test_core.py, test_providers.py, test_live.py (opt-in)
```

## Design rules

1. **Never replaces your agent.** Returns evidence; your agent acts.
2. **Read-only by default.** `SQLTool` rejects writes; risky tools need a human.
3. **Cheap checks first.** Ranked by information gain per cost, penalized by latency and risk.
4. **Abstains honestly.** Below threshold → says so instead of guessing.
5. **Works with no LLM.** Deterministic offline path keeps everything testable.

## Contributing

PRs welcome: new providers, ranking strategies, belief updaters, framework adapters. Keep the core stdlib-only, add tests, use conventional commits (`feat:`, `fix:`, `test:`, `docs:`).

## License

Apache-2.0 — see [LICENSE](LICENSE).
