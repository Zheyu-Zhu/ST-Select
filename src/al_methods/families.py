"""Map registered strategies to the paper's reporting families.

The code carries 8 internal `family` labels (baseline, uncertainty, diversity,
hybrid, adversarial, medical, spatial, rl). The paper reports **5 families + RL**:
Random, Uncertainty, Coverage, Hybrid, Spatial, RL. This module is the single
source of truth for that rollup (used by run_experiment's by_family block and
the family plots).

    baseline     -> Random
    uncertainty  -> Uncertainty
    diversity    -> Coverage
    hybrid       -> Hybrid
    adversarial  -> Hybrid          (merged)
    spatial      -> Spatial
    rl           -> RL
    medical      -> per-method:
        cald, ceal, confidnet, suggestive_annotation -> Uncertainty
        coregcn                                        -> Coverage
"""

from .registry import AL_REGISTRY

# internal family string -> reporting family
_FAMILY_TO_REPORT = {
    "baseline": "Random",
    "uncertainty": "Uncertainty",
    "diversity": "Coverage",
    "hybrid": "Hybrid",
    "adversarial": "Hybrid",
    "spatial": "Spatial",
    "rl": "RL",
}

# per-method overrides (the medical family splits by algorithm)
_METHOD_OVERRIDE = {
    "cald": "Uncertainty",
    "ceal": "Uncertainty",
    "confidnet": "Uncertainty",
    "suggestive_annotation": "Uncertainty",
    "coregcn": "Coverage",
}


def report_family(method_name: str) -> str:
    """Reporting family for a registered strategy name."""
    if method_name in _METHOD_OVERRIDE:
        return _METHOD_OVERRIDE[method_name]
    cls = AL_REGISTRY.get(method_name)
    internal = getattr(cls, "family", "base") if cls is not None else "base"
    return _FAMILY_TO_REPORT.get(internal, internal.capitalize())


def family_of_methods(method_names):
    """{reporting_family: [method_name, ...]} for the given methods."""
    out = {}
    for m in method_names:
        out.setdefault(report_family(m), []).append(m)
    return out
