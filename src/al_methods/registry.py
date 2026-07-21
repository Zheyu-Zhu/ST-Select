"""Registry for all AL strategies — discover and instantiate by name."""

import inspect
from typing import Dict, List, Type

from .base import ActiveLearningStrategy

AL_REGISTRY: Dict[str, Type[ActiveLearningStrategy]] = {}


def register(cls: Type[ActiveLearningStrategy]) -> Type[ActiveLearningStrategy]:
    AL_REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str, **kwargs) -> ActiveLearningStrategy:
    if name not in AL_REGISTRY:
        raise KeyError(
            f"Unknown AL strategy '{name}'. Available: {list(AL_REGISTRY.keys())}"
        )
    cls = AL_REGISTRY[name]
    # Forward only the kwargs this strategy's constructor actually accepts, so
    # callers can pass e.g. seed=... uniformly without crashing strategies whose
    # __init__ does not take it. This is what makes multi-seed runs actually vary
    # the acquisition RNG (see run_experiment._run_one_run).
    params = inspect.signature(cls.__init__).parameters
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if accepts_var_kwargs:
        accepted = kwargs
    else:
        accepted = {key: val for key, val in kwargs.items() if key in params}
    return cls(**accepted)


def list_strategies() -> List[str]:
    return list(AL_REGISTRY.keys())