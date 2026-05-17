"""Load and manage strategy YAML definitions."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from src.strategies.models import StrategyDefinition

logger = logging.getLogger(__name__)

_DEFAULTS_DIR = Path(__file__).parent / "defaults"


def load_strategy(name: str, custom_dir: Path | None = None) -> StrategyDefinition:
    """Load a strategy YAML by name.

    Searches custom_dir first, then the built-in defaults directory.
    Raises FileNotFoundError if the strategy does not exist.
    """
    filename = f"{name}.yaml"

    # Check custom dir first
    if custom_dir is not None:
        custom_path = custom_dir / filename
        if custom_path.exists():
            return _load_yaml(custom_path)

    # Check defaults
    default_path = _DEFAULTS_DIR / filename
    if default_path.exists():
        return _load_yaml(default_path)

    raise FileNotFoundError(
        f"Strategy '{name}' not found. Searched: "
        f"{custom_dir}, {_DEFAULTS_DIR}"
    )


def list_strategies(custom_dir: Path | None = None) -> list[StrategyDefinition]:
    """Return all available strategies from defaults and custom directory."""
    strategies: list[StrategyDefinition] = []
    seen_names: set[str] = set()

    # Load from custom dir (higher priority)
    if custom_dir is not None and custom_dir.exists():
        for path in sorted(custom_dir.glob("*.yaml")):
            try:
                s = _load_yaml(path)
                if s.name not in seen_names:
                    seen_names.add(s.name)
                    strategies.append(s)
            except Exception as exc:
                logger.warning("Failed to load strategy %s: %s", path, exc)

    # Load from defaults
    if _DEFAULTS_DIR.exists():
        for path in sorted(_DEFAULTS_DIR.glob("*.yaml")):
            try:
                s = _load_yaml(path)
                if s.name not in seen_names:
                    seen_names.add(s.name)
                    strategies.append(s)
            except Exception as exc:
                logger.warning("Failed to load strategy %s: %s", path, exc)

    return strategies


def apply_strategy_to_workflow(
    strategy: StrategyDefinition,
    selected_analysts: list[str] | None,
) -> list[str]:
    """Filter and order analysts based on strategy config.

    Returns the list of analyst keys to enable.
    If the strategy specifies agents, only those are used.
    Otherwise falls back to the provided selected_analysts or all analysts.
    """
    if not strategy.agents:
        return selected_analysts or []

    strategy_agents = [a.agent_key for a in strategy.agents if a.enabled]

    if selected_analysts is None:
        return strategy_agents

    # Intersect: keep only analysts that are in both lists
    selected_set = set(selected_analysts)
    return [a for a in strategy_agents if a in selected_set]


def _load_yaml(path: Path) -> StrategyDefinition:
    """Parse a YAML file into a StrategyDefinition."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return StrategyDefinition(**raw)
