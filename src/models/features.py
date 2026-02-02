from __future__ import annotations

import math
from typing import Any, Dict, List

from ..types import ContextSignature, NCCLConfig, ParameterSpace


class ConfigFeaturizer:
    def __init__(self, parameter_space: ParameterSpace) -> None:
        self.parameter_space = parameter_space

    def encode(self, config: NCCLConfig, context: ContextSignature | None = None) -> List[float]:
        vector: List[float] = []
        for name, spec in sorted(self.parameter_space.specs.items()):
            value = config.params.get(name, spec.default)
            if spec.kind == "bool":
                vector.append(1.0 if bool(value) else 0.0)
            elif spec.kind == "enum":
                choices = spec.choices or []
                for choice in choices:
                    vector.append(1.0 if value == choice else 0.0)
            elif spec.kind in ("int", "float"):
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    numeric = 0.0
                vector.append(math.log(max(1.0, numeric)))
            else:
                vector.append(0.0)

        if context is not None:
            vector.append(float(context.nodes))
            vector.append(float(context.gpus_per_node or 0))
        return vector
