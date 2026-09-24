"""HypothesisOS: give AI agents the ability to investigate what they don't know."""

from .controller import HypothesisOS, Investigator
from .core.types import EvidenceBundle, Experiment, Hypothesis, Observation
from .models import HeuristicModel
from .providers import (
    AnthropicModel,
    GeminiModel,
    GroqModel,
    OllamaModel,
    OpenAICompatibleModel,
    OpenAIModel,
    OpenRouterModel,
    create_model,
)
from .tools import FunctionTool, HTTPTool, SQLTool, Tool

__all__ = [
    "HypothesisOS", "Investigator", "EvidenceBundle", "Experiment",
    "Hypothesis", "Observation", "HeuristicModel", "FunctionTool",
    "HTTPTool", "SQLTool", "Tool",
    # providers
    "create_model", "OpenAIModel", "AnthropicModel", "GeminiModel",
    "GroqModel", "OpenRouterModel", "OllamaModel", "OpenAICompatibleModel",
]

__version__ = "0.1.0"
