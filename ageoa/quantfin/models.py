from __future__ import annotations

from typing import Any, Callable, Dict, List, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny

class YieldCurve(BaseModel):
    """Abstract base class for yield curves."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def disc(self, t: float) -> float:
        raise NotImplementedError("Subclasses must implement disc")

    def forward(self, t1: float, t2: float) -> float:
        return np.log(self.disc(t1) / self.disc(t2)) / (t2 - t1)

    def spot(self, t: float) -> float:
        return self.forward(0.0, t)

class FlatCurve(YieldCurve):
    """A flat yield curve with one continuously compounded rate."""
    rate: float = Field(..., ge=0.0)

    def disc(self, t: float) -> float:
        return float(np.exp(-self.rate * t))

class NetYC(YieldCurve):
    """YieldCurve representing the difference between two YieldCurves."""
    yc1: SerializeAsAny[YieldCurve]
    yc2: SerializeAsAny[YieldCurve]

    def disc(self, t: float) -> float:
        return self.yc1.disc(t) / self.yc2.disc(t)

class CashFlow(BaseModel):
    time: float = Field(..., ge=0.0)
    amount: float

class CCProcessor(BaseModel):
    monitor_time: float = Field(..., ge=0.0)
    payout_func_names: List[str] = Field(
        default_factory=list,
        description="Deterministic identifiers for payout functions",
    )
    payout_funcs: List[Callable[[Dict[float, Any]], CashFlow]] = Field(
        default_factory=list,
        exclude=True,
        repr=False,
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

class ContingentClaim(BaseModel):
    processors: List[CCProcessor] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

class DiscretizeModel(BaseModel):
    """Interface for models that can be discretized for MC simulation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)


@runtime_checkable
class SeededMonteCarloSimulator(Protocol):
    """Deterministic simulator boundary using an explicit seeded RNG."""

    def __call__(
        self,
        model: DiscretizeModel,
        claim: ContingentClaim,
        rng: np.random.Generator,
        trials: int,
        anti: bool,
    ) -> float:
        ...
