import numpy as np
from pydantic import BaseModel, Field
from typing import Callable, List, Dict, Any, Union

class YieldCurve(BaseModel):
    """Abstract base class for yield curves."""
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
    yc1: YieldCurve
    yc2: YieldCurve

    def disc(self, t: float) -> float:
        return self.yc1.disc(t) / self.yc2.disc(t)

class CashFlow(BaseModel):
    time: float = Field(..., ge=0.0)
    amount: float

class CCProcessor(BaseModel):
    monitor_time: float = Field(..., ge=0.0)
    payout_funcs: List[Callable[[Dict[float, Any]], CashFlow]]

    class Config:
        arbitrary_types_allowed = True

class ContingentClaim(BaseModel):
    processors: List[CCProcessor] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

class DiscretizeModel(BaseModel):
    """Interface for models that can be discretized for MC simulation."""
    pass
