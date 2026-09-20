"""HypothesisOS: give AI agents the ability to investigate what they don't know."""

from .controller import HypothesisOS, Investigator
from .core.types import EvidenceBundle, Experiment, Hypothesis, Observation
from .models import HeuristicModel
from .tools import FunctionTool, HTTPTool, SQLTool, Tool

__all__ = [
    "HypothesisOS", "Investigator", "EvidenceBundle", "Experiment",
    "Hypothesis", "Observation", "HeuristicModel", "FunctionTool",
    "HTTPTool", "SQLTool", "Tool",
]

__version__ = "0.1.0"
