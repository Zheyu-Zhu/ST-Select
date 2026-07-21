from .base import ActiveLearningStrategy, select_next_batch
from .registry import AL_REGISTRY, get_strategy, list_strategies

# Import every strategy subpackage so that all @register decorators run and the
# registry is fully populated on `import al_methods` (no manual per-family imports).
from . import uncertainty  # noqa: E402,F401
from . import diversity  # noqa: E402,F401
from . import hybrid  # noqa: E402,F401
from . import medical  # noqa: E402,F401
from . import spatial  # noqa: E402,F401
from . import rl  # noqa: E402,F401

from .families import report_family, family_of_methods  # noqa: E402

__all__ = [
    "ActiveLearningStrategy",
    "select_next_batch",
    "AL_REGISTRY",
    "get_strategy",
    "list_strategies",
    "report_family",
    "family_of_methods",
]