"""
Suitability policy, portfolio construction, backtesting, and experiment running.
data.py is market data/features, model.py is PyTorch, this file owns decisions/evaluation,
and app.py is presentation. Forecasts may only create small relative tilts inside hard
client contraints.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from math import isfinite, sqrt
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

# Keep Matplotlib's font/config cache in the project's already-ignored cache.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / "data" / "cache" / "matplotlib")
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import (
    ALL_ASSETS,
    CASH_ASSET,
    DEFENSIVE_ASSETS,
    GROWTH_ASSETS,
    RISKY_ASSETS,
    TARGET_COLUMN,
    build_feature_panel,
    download_prices,
)
from model import ModelBundle, walk_forward_predictions

PERIODS_PER_YEAR = 12
VALID_STABILITY = frozenset({"low", "medium", "high"})
VALID_OBJECTIVES = frozenset(
    {"preservation", "income", "growth", "aggressive_growth"}
)

PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "balanced": {
        "name": "Balanced accumulation scenario",
        "age": 42,
        "investable_assets": 450_000,
        "annual_income": 210_000,
        "monthly_contribution": 5_000,
        "goal_amount": 1_500_000,
        "horizon_years": 15,
        "risk_tolerance": 3,
        "max_drawdown_tolerance": 0.22,
        "monthly_expenses": 8_500,
        "liquidity_needed_12m": 30_000,
        "emergency_cash_held": 60_000,
        "income_stability": "high",
        "objective": "growth",
        "excluded_assets": (),
        "max_single_asset_weight": 0.35,
    },
    "conservative": {
        "name": "Conservative income scenario",
        "age": 61,
        "investable_assets": 850_000,
        "annual_income": 180_000,
        "monthly_contribution": 2_500,
        "goal_amount": 1_000_000,
        "horizon_years": 7,
        "risk_tolerance": 2,
        "max_drawdown_tolerance": 0.12,
        "monthly_expenses": 9_000,
        "liquidity_needed_12m": 100_000,
        "emergency_cash_held": 108_000,
        "income_stability": "high",
        "objective": "income",
        "excluded_assets": ("EEM",),
        "max_single_asset_weight": 0.35,
    },
    "growth": {
        "name": "Long-horizon growth scenario",
        "age": 29,
        "investable_assets": 125_000,
        "annual_income": 150_000,
        "monthly_contribution": 4_000,
        "goal_amount": 2_000_000,
        "horizon_years": 30,
        "risk_tolerance": 5,
        "max_drawdown_tolerance": 0.35,
        "monthly_expenses": 5_500,
        "liquidity_needed_12m": 10_000,
        "emergency_cash_held": 33_000,
        "income_stability": "medium",
        "objective": "aggressive_growth",
        "excluded_assets": (),
        "max_single_asset_weight": 0.45,
    },
}

class InfeasibleProfileError(ValueError):
    """Raised when client constraints cannot form a long-only portfolio."""

@dataclass(frozen=True)
class ClientProfile:
    # Validated client facts used by the demonstration suitability policy.

    name: str
    age: int
    investable_assets: float
    annual_income: float
    monthly_contribution: float
    goal_amount: float
    horizon_years: float
    risk_tolerance: int
    max_drawdown_tolerance: float
    monthly_expenses: float
    liquidity_needed_12m: float
    emergency_cash_held: float
    income_stability: str
    objective: str
    excluded_assets: tuple[str, ...] = field(default_factory=tuple)
    max_single_asset_weight: float = 0.45

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("name must not be empty")
        object.__setattr__(self, "name", name)
        if isinstance(self.age, bool) or not isinstance(self.age, int):
            raise TypeError("age must be an integer")
        if not 18 <= self.age <= 120:
            raise ValueError("age must be between 18 and 120")

        nonnegative = (
            "annual_income",
            "monthly_contribution",
            "monthly_expenses",
            "liquidity_needed_12m",
            "emergency_cash_held",
        )
        positive = ("investable_assets", "goal_amount", "horizon_years")
        for field_name in nonnegative + positive:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be numeric")
            if not isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
            if field_name in nonnegative and float(value) < 0:
                raise ValueError(f"{field_name} must be non-negative")
            if field_name in positive and float(value) <= 0:
                raise ValueError(f"{field_name} must be greater than zero")

        if (
            isinstance(self.risk_tolerance, bool)
            or not isinstance(self.risk_tolerance, int)
            or self.risk_tolerance not in range(1, 6)
        ):
            raise ValueError("risk_tolerance must be an integer from 1 to 5")
        if not 0 < float(self.max_drawdown_tolerance) <= 1:
            raise ValueError("max_drawdown_tolerance must be in (0, 1]")
        if not 0 < float(self.max_single_asset_weight) <= 1:
            raise ValueError("max_single_asset_weight must be in (0, 1]")

        stability = str(self.income_stability).strip().lower()
        objective = str(self.objective).strip().lower().replace(" ", "_")
        if stability not in VALID_STABILITY:
            raise ValueError(f"income_stability must be one of {sorted(VALID_STABILITY)}")
        if objective not in VALID_OBJECTIVES:
            raise ValueError(f"objective must be one of {sorted(VALID_OBJECTIVES)}")
        object.__setattr__(self, "income_stability", stability)
        object.__setattr__(self, "objective", objective)

        raw = (self.excluded_assets,) if isinstance(self.excluded_assets, str) else tuple(self.excluded_assets)
        exclusions = tuple(sorted({str(asset).strip().upper() for asset in raw}))
        unknown = set(exclusions).difference(ALL_ASSETS)
        if unknown:
            raise ValueError(f"unknown excluded assets: {sorted(unknown)}")
        object.__setattr__(self, "excluded_assets", exclusions)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["excluded_assets"] = list(self.excluded_assets)
        return result

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ClientProfile":
        return cls(**dict(values))

@dataclass(frozen=True)
class SuitabilityPolicy:
    # Auditable constraints and strategic allocation for one client.

    cash_floor: float
    growth_cap: float
    growth_target: float
    target_volatility: float
    strategic_weights: dict[str, float]
    risk_label: str
    goal_required_return: float | None
    warnings: tuple[str, ...]
    rationales: tuple[str, ...]
    excluded_assets: tuple[str, ...]
    max_single_asset_weight: float
    actionable: bool = True
    abstention_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

_RISK_LABELS = {
    1: "conservative",
    2: "moderately_conservative",
    3: "balanced",
    4: "growth",
    5: "aggressive_growth",
}
_BASE_GROWTH_TARGET = {1: 0.25, 2: 0.40, 3: 0.60, 4: 0.75, 5: 0.88}
_RISK_GROWTH_CAP = {1: 0.35, 2: 0.50, 3: 0.70, 4: 0.85, 5: 0.95}
_TARGET_VOLATILITY = {1: 0.06, 2: 0.08, 3: 0.10, 4: 0.12, 5: 0.15}
_OBJECTIVE_ADJUSTMENT = {
    "preservation": -0.15,
    "income": -0.05,
    "growth": 0.0,
    "aggressive_growth": 0.08,
}
_OBJECTIVE_CAP = {
    "preservation": 0.35,
    "income": 0.55,
    "growth": 0.85,
    "aggressive_growth": 0.95,
}
_GROWTH_PREFERENCES = {"SPY": 0.55, "EFA": 0.20, "EEM": 0.10, "VNQ": 0.15}
_DEFENSIVE_PREFERENCES = {"AGG": 0.55, "TIP": 0.30, "GLD": 0.15}

def _required_annual_return(profile: ClientProfile) -> float | None:
    months = max(1, int(round(profile.horizon_years * 12)))
    principal = float(profile.investable_assets)
    contribution = float(profile.monthly_contribution)
    target = float(profile.goal_amount)

    def future_value(annual_rate: float) -> float:
        monthly_rate = (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0
        principal_value = principal * (1.0 + monthly_rate) ** months
        if abs(monthly_rate) < 1e-12:
            contribution_value = contribution * months
        else:
            contribution_value = contribution * (
                ((1.0 + monthly_rate) ** months - 1.0) / monthly_rate
            )
        return principal_value + contribution_value

    low, high = -0.99, 1.0
    if target <= future_value(low):
        return low
    while future_value(high) < target and high < 1024.0:
        high *= 2.0
    if future_value(high) < target:
        return None
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if future_value(midpoint) < target:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0

def _allocate_policy_bucket(
    total: float, preferences: Mapping[str, float], cap: float
) -> dict[str, float]:
    assets = tuple(preferences)
    if total < -1e-10 or total > len(assets) * cap + 1e-10:
        raise InfeasibleProfileError("asset exclusions and the single-asset cap conflict")
    if not assets:
        if total > 1e-10:
            raise InfeasibleProfileError("no eligible assets remain for a required bucket")
        return {}
    remaining = max(0.0, total)
    available = list(assets)
    result = {asset: 0.0 for asset in assets}
    while remaining > 1e-12 and available:
        preference_sum = sum(max(0.0, preferences[a]) for a in available)
        proposed = {
            a: (
                remaining / len(available)
                if preference_sum <= 0
                else remaining * max(0.0, preferences[a]) / preference_sum
            )
            for a in available
        }
        capped = [a for a, value in proposed.items() if value > cap + 1e-12]
        if not capped:
            for asset, value in proposed.items():
                result[asset] += value
            remaining = 0.0
            break
        for asset in capped:
            room = cap - result[asset]
            result[asset] += room
            remaining -= room
            available.remove(asset)
    if remaining > 1e-8:
        raise InfeasibleProfileError("single-asset cap leaves no feasible allocation")
    return result

def build_policy(
    profile: ClientProfile, *, raise_on_abstention: bool = False
) -> SuitabilityPolicy:
    # Turn a client profile into binding, inspectable portfolio guardrails.

    if not isinstance(profile, ClientProfile):
        raise TypeError("profile must be a ClientProfile")
    included = tuple(asset for asset in ALL_ASSETS if asset not in profile.excluded_assets)
    if not included:
        raise InfeasibleProfileError("all assets have been excluded")
    cap = float(profile.max_single_asset_weight)
    included_non_cash = tuple(asset for asset in included if asset != CASH_ASSET)
    total_capacity = len(included_non_cash) * cap + (1.0 if CASH_ASSET in included else 0.0)
    if total_capacity < 1.0 - 1e-10:
        raise InfeasibleProfileError("too few eligible assets remain for max_single_asset_weight")

    reserve_months = {"low": 12, "medium": 9, "high": 6}[profile.income_stability]
    reserve_target = reserve_months * float(profile.monthly_expenses)
    reserve_shortfall = max(0.0, reserve_target - float(profile.emergency_cash_held))
    near_term_cash = reserve_shortfall + float(profile.liquidity_needed_12m)
    cash_floor = min(1.0, near_term_cash / float(profile.investable_assets))
    warnings: list[str] = []
    rationales: list[str] = [
        f"Cash floor covers ${near_term_cash:,.0f} of stated 12-month needs and the "
        f"shortfall to a {reserve_months}-month emergency reserve.",
        "The model may only tilt within this policy; it cannot override liquidity, "
        "concentration, growth, or long-only constraints.",
    ]
    if cash_floor > 0 and CASH_ASSET not in included:
        raise InfeasibleProfileError("BIL is excluded but the profile requires a cash floor")

    if profile.horizon_years < 3:
        horizon_cap = 0.25
    elif profile.horizon_years < 5:
        horizon_cap = 0.45
    elif profile.horizon_years < 10:
        horizon_cap = 0.70
    else:
        horizon_cap = 0.95
    drawdown_cap = min(0.95, float(profile.max_drawdown_tolerance) / 0.45)
    growth_cap = max(
        0.0,
        min(
            _RISK_GROWTH_CAP[profile.risk_tolerance],
            _OBJECTIVE_CAP[profile.objective],
            horizon_cap,
            drawdown_cap,
            1.0 - cash_floor,
        ),
    )
    requested_growth = max(
        0.0,
        _BASE_GROWTH_TARGET[profile.risk_tolerance]
        + _OBJECTIVE_ADJUSTMENT[profile.objective],
    )
    eligible_growth = tuple(asset for asset in GROWTH_ASSETS if asset in included)
    eligible_defensive = tuple(asset for asset in DEFENSIVE_ASSETS if asset in included)
    cash_capacity = 1.0 if CASH_ASSET in included else 0.0
    growth_capacity = min(growth_cap, len(eligible_growth) * cap)
    defensive_capacity = len(eligible_defensive) * cap
    minimum_growth = max(0.0, 1.0 - defensive_capacity - cash_capacity)
    if minimum_growth > growth_capacity + 1e-10:
        raise InfeasibleProfileError("growth cap, exclusions, and concentration cap cannot all be satisfied")
    growth_target = max(min(requested_growth, growth_capacity), minimum_growth)
    cash_target = min(max(cash_floor, 1.0 - growth_target - defensive_capacity), cash_capacity)
    defensive_target = 1.0 - growth_target - cash_target
    if defensive_target > defensive_capacity + 1e-10 or defensive_target < -1e-10:
        raise InfeasibleProfileError("no feasible defensive allocation remains")
    if growth_target + 1e-10 < requested_growth:
        warnings.append(
            "Growth exposure was reduced by horizon, drawdown, objective, liquidity, "
            "exclusion, or concentration guardrails."
        )
    if growth_target > requested_growth + 1e-10:
        warnings.append(
            "Growth exposure increased because exclusions and the single-asset cap "
            "left insufficient defensive capacity."
        )

    growth_preferences = {a: _GROWTH_PREFERENCES.get(a, 1.0) for a in eligible_growth}
    defensive_preferences = {a: _DEFENSIVE_PREFERENCES.get(a, 1.0) for a in eligible_defensive}
    weights = {asset: 0.0 for asset in ALL_ASSETS}
    weights.update(_allocate_policy_bucket(growth_target, growth_preferences, cap))
    weights.update(_allocate_policy_bucket(defensive_target, defensive_preferences, cap))
    if CASH_ASSET in included:
        weights[CASH_ASSET] = cash_target
    weights = {asset: float(weights.get(asset, 0.0)) for asset in ALL_ASSETS}
    if abs(sum(weights.values()) - 1.0) > 1e-8:
        raise RuntimeError("internal policy allocation did not sum to one")

    required_return = _required_annual_return(profile)
    actionable = True
    abstention_reason: str | None = None
    if required_return is None:
        warnings.append("The goal return could not be solved within the numeric range.")
        actionable = False
        abstention_reason = "The stated goal is outside the model's numeric planning range."
    elif required_return > 0.15:
        warnings.append(
            f"The goal requires approximately {required_return:.1%} annualized return; "
            "consider changing the goal, horizon, or contribution rather than relying "
            "on an aggressive forecast."
        )
    if required_return is not None and required_return > 0.30:
        actionable = False
        abstention_reason = (
            "The required return exceeds the policy's 30% planning guardrail, so the "
            "engine abstains from forecast-driven tilts."
        )
    if cash_floor >= 0.95:
        actionable = False
        abstention_reason = (
            "Near-term cash needs consume at least 95% of investable assets; the engine "
            "abstains from forecast-driven tilts."
        )
    if abstention_reason:
        warnings.append(abstention_reason)
    if raise_on_abstention and not actionable:
        raise InfeasibleProfileError(abstention_reason or "policy is not actionable")

    rationales.extend(
        [
            f"Growth is capped at {growth_cap:.0%}; the strategic target is {growth_target:.0%}.",
            "Drawdown tolerance becomes a conservative growth cap using a 45% growth-asset stress loss.",
            f"The annualized volatility budget is {_TARGET_VOLATILITY[profile.risk_tolerance]:.0%}.",
            "The concentration cap applies to non-cash assets; BIL may exceed it for liquidity.",
        ]
    )
    return SuitabilityPolicy(
        cash_floor=float(cash_floor),
        growth_cap=float(growth_cap),
        growth_target=float(growth_target),
        target_volatility=float(_TARGET_VOLATILITY[profile.risk_tolerance]),
        strategic_weights=weights,
        risk_label=_RISK_LABELS[profile.risk_tolerance],
        goal_required_return=required_return,
        warnings=tuple(warnings),
        rationales=tuple(rationales),
        excluded_assets=profile.excluded_assets,
        max_single_asset_weight=cap,
        actionable=actionable,
        abstention_reason=abstention_reason,
    )

assess_suitability = build_policy
derive_policy = build_policy

# Constrained portfolio construction

def _asset_series(values: Mapping[str, float] | pd.Series | None) -> pd.Series:
    if values is None:
        return pd.Series(np.nan, index=ALL_ASSETS, dtype=float)
    result = pd.to_numeric(pd.Series(values), errors="coerce")
    result.index = result.index.map(lambda value: str(value).upper())
    return result.groupby(level=0).last().reindex(ALL_ASSETS).astype(float)

def _allocate_capped_series(total: float, desired: pd.Series, cap: float) -> pd.Series:
    assets = list(desired.index)
    if total < -1e-10 or total > len(assets) * cap + 1e-10:
        raise InfeasibleProfileError("bucket total conflicts with concentration cap")
    result = pd.Series(0.0, index=assets, dtype=float)
    remaining = max(0.0, float(total))
    available = list(assets)
    preferences = desired.clip(lower=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    while remaining > 1e-12 and available:
        preference_sum = float(preferences.loc[available].sum())
        if preference_sum <= 0:
            proposed = pd.Series(remaining / len(available), index=available)
        else:
            proposed = remaining * preferences.loc[available] / preference_sum
        capped = [asset for asset in available if proposed[asset] > cap + 1e-12]
        if not capped:
            result.loc[available] += proposed
            remaining = 0.0
            break
        for asset in capped:
            room = max(0.0, cap - result[asset])
            result[asset] += room
            remaining -= room
            available.remove(asset)
    if remaining > 1e-8:
        raise InfeasibleProfileError("no room remains under the concentration cap")
    return result

def _project_constraints(desired: pd.Series, policy: SuitabilityPolicy) -> pd.Series:
    # Project desired weights onto all client policy constraints.

    desired = desired.reindex(ALL_ASSETS).fillna(0.0).clip(lower=0.0)
    desired.loc[list(policy.excluded_assets)] = 0.0
    included = [asset for asset in ALL_ASSETS if asset not in policy.excluded_assets]
    cap = float(policy.max_single_asset_weight)
    included_non_cash = [asset for asset in included if asset != CASH_ASSET]
    total_capacity = len(included_non_cash) * cap + (1.0 if CASH_ASSET in included else 0.0)
    if total_capacity < 1.0 - 1e-10:
        raise InfeasibleProfileError("single-asset cap and exclusions are infeasible")

    growth = [asset for asset in GROWTH_ASSETS if asset in included]
    defensive = [asset for asset in DEFENSIVE_ASSETS if asset in included]
    has_cash = CASH_ASSET in included
    cash_capacity = 1.0 if has_cash else 0.0  # BIL is exempt from the non-cash cap.
    growth_capacity = min(float(policy.growth_cap), len(growth) * cap)
    defensive_capacity = len(defensive) * cap
    cash_floor = float(policy.cash_floor)
    if cash_floor > cash_capacity + 1e-10:
        raise InfeasibleProfileError("cash floor exceeds available BIL capacity")

    desired_total = float(desired.sum())
    if desired_total <= 0:
        desired = pd.Series(policy.strategic_weights, dtype=float).reindex(
            ALL_ASSETS, fill_value=0.0
        )
        desired_total = float(desired.sum())
    desired = desired / desired_total
    desired_cash = float(desired.get(CASH_ASSET, 0.0)) if has_cash else 0.0
    desired_growth = float(desired.reindex(growth).sum())
    cash_weight = min(cash_capacity, max(cash_floor, desired_cash))
    growth_weight = min(growth_capacity, max(0.0, desired_growth))
    defensive_weight = 1.0 - cash_weight - growth_weight

    if defensive_weight < 0:
        reduction = min(growth_weight, -defensive_weight)
        growth_weight -= reduction
        defensive_weight += reduction
        if defensive_weight < -1e-10:
            cash_weight += defensive_weight
            defensive_weight = 0.0
    if defensive_weight > defensive_capacity:
        deficit = defensive_weight - defensive_capacity
        add_growth = min(deficit, growth_capacity - growth_weight)
        growth_weight += add_growth
        deficit -= add_growth
        add_cash = min(deficit, cash_capacity - cash_weight)
        cash_weight += add_cash
        deficit -= add_cash
        defensive_weight = defensive_capacity
        if deficit > 1e-8:
            raise InfeasibleProfileError("policy constraints leave the portfolio underfunded")

    defensive_weight = 1.0 - cash_weight - growth_weight
    if defensive_weight < -1e-8 or defensive_weight > defensive_capacity + 1e-8:
        raise InfeasibleProfileError("unable to satisfy policy bucket totals")
    result = pd.Series(0.0, index=ALL_ASSETS, dtype=float)
    if growth:
        result.loc[growth] = _allocate_capped_series(
            growth_weight, desired.reindex(growth).fillna(0.0), cap
        )
    if defensive:
        result.loc[defensive] = _allocate_capped_series(
            defensive_weight, desired.reindex(defensive).fillna(0.0), cap
        )
    if has_cash:
        result.loc[CASH_ASSET] = cash_weight
    result.loc[list(policy.excluded_assets)] = 0.0
    result[result.abs() < 1e-14] = 0.0
    if abs(float(result.sum()) - 1.0) > 1e-8:
        raise RuntimeError("constraint projection did not sum to one")
    return result

def _bucket_zscores(forecasts: pd.Series, assets: list[str]) -> pd.Series:
    values = forecasts.reindex(assets)
    valid = values.dropna()
    scores = pd.Series(0.0, index=assets, dtype=float)
    if len(valid) < 2 or float(valid.std(ddof=0)) < 1e-12:
        return scores
    scores.loc[valid.index] = ((valid - float(valid.mean())) / float(valid.std(ddof=0))).clip(-2.0, 2.0)
    return scores

def _trailing_momentum(
    trailing_returns: pd.DataFrame | None, *, months: int
) -> pd.Series:
    if trailing_returns is None or len(trailing_returns) < months:
        return pd.Series(np.nan, index=ALL_ASSETS, dtype=float)
    frame = trailing_returns.copy()
    frame.columns = [str(column).upper() for column in frame.columns]
    frame = frame.reindex(columns=ALL_ASSETS).apply(pd.to_numeric, errors="coerce")
    window = frame.tail(months)
    if len(window) < months:
        return pd.Series(np.nan, index=ALL_ASSETS, dtype=float)
    return (1.0 + window).prod(min_count=months) - 1.0

def _estimated_volatility(
    weights: pd.Series, trailing_returns: pd.DataFrame | None
) -> float | None:
    if trailing_returns is None or len(trailing_returns) < 3:
        return None
    frame = trailing_returns.copy()
    frame.columns = [str(column).upper() for column in frame.columns]
    frame = frame.reindex(columns=ALL_ASSETS).apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if len(frame) < 3:
        return None
    covariance = frame.cov(min_periods=3).reindex(index=ALL_ASSETS, columns=ALL_ASSETS)
    covariance = covariance.fillna(0.0).to_numpy(dtype=float) * 12.0
    shrunk = 0.5 * covariance + 0.5 * np.diag(np.diag(covariance))
    variance = float(weights.to_numpy(dtype=float) @ shrunk @ weights.to_numpy(dtype=float))
    if not np.isfinite(variance) or variance < 0:
        return None
    return sqrt(max(0.0, variance))

def construct_portfolio(
    predicted_excess_returns: Mapping[str, float] | pd.Series | None,
    trailing_returns: pd.DataFrame | None,
    policy: SuitabilityPolicy,
    current_weights: Mapping[str, float] | pd.Series | None = None,
    *,
    tilt_strength: float = 0.25,
    turnover_blend: float = 0.50,
    model_signal_weight: float = 0.25,
    momentum_months: int = 9,
) -> tuple[pd.Series, dict[str, Any]]:
    # Build a long-only recommendation; client rules always outrank signals.

    if not isinstance(policy, SuitabilityPolicy):
        raise TypeError("policy must be a SuitabilityPolicy")
    if not 0 <= tilt_strength <= 0.25:
        raise ValueError("tilt_strength must be between 0 and 0.25")
    if not 0 <= turnover_blend <= 1:
        raise ValueError("turnover_blend must be between 0 and 1")
    if not 0 <= model_signal_weight <= 1:
        raise ValueError("model_signal_weight must be between 0 and 1")
    if not isinstance(momentum_months, int) or momentum_months < 2:
        raise ValueError("momentum_months must be an integer of at least 2")

    strategic = _project_constraints(
        pd.Series(policy.strategic_weights, dtype=float).reindex(ALL_ASSETS, fill_value=0.0),
        policy,
    )
    forecasts = _asset_series(predicted_excess_returns)
    eligible = forecasts.drop(index=list(policy.excluded_assets), errors="ignore")
    finite_count = int(np.isfinite(eligible).sum())
    forecast_used = policy.actionable and finite_count >= 2
    fallback_reason: str | None = None
    if not policy.actionable:
        fallback_reason = policy.abstention_reason or "suitability policy abstained"
    elif finite_count < 2:
        fallback_reason = "fewer than two finite asset forecasts were available"

    desired = strategic.copy()
    momentum = _trailing_momentum(trailing_returns, months=momentum_months)
    momentum_available = int(
        np.isfinite(momentum.drop(index=list(policy.excluded_assets), errors="ignore")).sum()
    ) >= 2
    if forecast_used:
        growth = [a for a in GROWTH_ASSETS if a not in policy.excluded_assets and strategic[a] > 0]
        defensive = [a for a in DEFENSIVE_ASSETS if a not in policy.excluded_assets and strategic[a] > 0]
        for bucket in (growth, defensive):
            if not bucket:
                continue
            model_scores = _bucket_zscores(forecasts, bucket)
            scores = model_scores
            if momentum_available:
                scores = (
                    model_signal_weight * model_scores
                    + (1.0 - model_signal_weight) * _bucket_zscores(momentum, bucket)
                )
            bucket_total = float(strategic.loc[bucket].sum())
            tilted = strategic.loc[bucket] * (1.0 + tilt_strength * scores)
            if float(tilted.sum()) > 0:
                desired.loc[bucket] = bucket_total * tilted / float(tilted.sum())
        desired = _project_constraints(desired, policy)

    pre_risk = desired.copy()
    estimated_before = _estimated_volatility(pre_risk, trailing_returns)
    volatility_scaled = False
    if (
        estimated_before is not None
        and estimated_before > policy.target_volatility + 1e-12
        and CASH_ASSET not in policy.excluded_assets
    ):
        scale = max(0.0, min(1.0, policy.target_volatility / estimated_before))
        scaled = pre_risk.copy()
        noncash = [asset for asset in ALL_ASSETS if asset != CASH_ASSET]
        scaled.loc[noncash] *= scale
        scaled.loc[CASH_ASSET] = 1.0 - float(scaled.loc[noncash].sum())
        desired = _project_constraints(scaled, policy)
        volatility_scaled = True

    current_valid = False
    if current_weights is not None:
        current = _asset_series(current_weights).fillna(0.0).clip(lower=0.0)
        current.loc[list(policy.excluded_assets)] = 0.0
        current_valid = float(current.sum()) > 0
        if current_valid:
            current /= float(current.sum())
            desired = _project_constraints(current + turnover_blend * (desired - current), policy)
    final_weights = _project_constraints(desired, policy)
    estimated_after = _estimated_volatility(final_weights, trailing_returns)

    active = final_weights - strategic
    explanation: dict[str, Any] = {
        "forecast_used": forecast_used,
        "fallback_reason": fallback_reason,
        "policy_actionable": policy.actionable,
        "risk_label": policy.risk_label,
        "strategic_weights": strategic.round(8).to_dict(),
        "pre_risk_weights": pre_risk.round(8).to_dict(),
        "target_weights": final_weights.round(8).to_dict(),
        "top_overweights": active[active > 1e-6].sort_values(ascending=False).head(3).round(6).to_dict(),
        "top_underweights": active[active < -1e-6].sort_values().head(3).round(6).to_dict(),
        "forecast_ranking": forecasts.dropna().sort_values(ascending=False).index.tolist(),
        "momentum_ranking": momentum.dropna().sort_values(ascending=False).index.tolist(),
        "signal_blend": {
            "model_weight": model_signal_weight if momentum_available else 1.0,
            "momentum_weight": (1.0 - model_signal_weight) if momentum_available else 0.0,
            "momentum_months": momentum_months,
            "momentum_available": momentum_available,
        },
        "volatility_scaled": volatility_scaled,
        "estimated_volatility_before": estimated_before,
        "estimated_volatility_after": estimated_after,
        "target_volatility": policy.target_volatility,
        "turnover_blend": turnover_blend if current_valid else None,
        "constraints": {
            "cash_floor": policy.cash_floor,
            "growth_cap": policy.growth_cap,
            "max_single_asset_weight": policy.max_single_asset_weight,
            "cash_exempt_from_single_asset_cap": True,
            "excluded_assets": list(policy.excluded_assets),
            "long_only": True,
            "leverage": False,
        },
    }
    return final_weights.rename("weight"), explanation

recommend_portfolio = construct_portfolio
build_portfolio = construct_portfolio

# Performance metrics

def _clean_returns(returns: pd.Series | list[float] | np.ndarray) -> pd.Series:
    result = pd.to_numeric(pd.Series(returns), errors="coerce").dropna().astype(float)
    if (result <= -1.0).any():
        raise ValueError("returns must be greater than -100%")
    return result

def cagr(returns: pd.Series | list[float] | np.ndarray) -> float:
    values = _clean_returns(returns)
    if values.empty:
        return float("nan")
    terminal = float((1.0 + values).prod())
    return terminal ** (PERIODS_PER_YEAR / len(values)) - 1.0 if terminal > 0 else float("nan")

def annualized_volatility(returns: pd.Series | list[float] | np.ndarray) -> float:
    values = _clean_returns(returns)
    return float(values.std(ddof=1) * sqrt(PERIODS_PER_YEAR)) if len(values) >= 2 else float("nan")

def _active_returns(
    returns: pd.Series | list[float] | np.ndarray,
    bil_returns: pd.Series | list[float] | np.ndarray | None,
) -> pd.Series:
    strategy = _clean_returns(returns)
    if bil_returns is None:
        return strategy
    benchmark = pd.to_numeric(pd.Series(bil_returns), errors="coerce")
    if isinstance(returns, pd.Series) and isinstance(bil_returns, pd.Series):
        joined = pd.concat([pd.to_numeric(returns, errors="coerce"), benchmark], axis=1, join="inner").dropna()
        return (joined.iloc[:, 0] - joined.iloc[:, 1]).astype(float)
    count = min(len(strategy), int(benchmark.notna().sum()))
    if count == 0:
        return pd.Series(dtype=float)
    return strategy.iloc[:count].reset_index(drop=True) - benchmark.dropna().iloc[:count].reset_index(drop=True)

def sharpe_ratio(
    returns: pd.Series | list[float] | np.ndarray,
    bil_returns: pd.Series | list[float] | np.ndarray | None = None,
) -> float:
    active = _active_returns(returns, bil_returns).dropna()
    if len(active) < 2 or float(active.std(ddof=1)) < 1e-15:
        return float("nan")
    return float(active.mean() / active.std(ddof=1) * sqrt(PERIODS_PER_YEAR))

def sortino_ratio(
    returns: pd.Series | list[float] | np.ndarray,
    bil_returns: pd.Series | list[float] | np.ndarray | None = None,
) -> float:
    active = _active_returns(returns, bil_returns).dropna()
    if active.empty:
        return float("nan")
    downside = float(np.sqrt(np.mean(np.minimum(active, 0.0) ** 2)))
    return (
        float(active.mean() * PERIODS_PER_YEAR / (downside * sqrt(PERIODS_PER_YEAR)))
        if downside >= 1e-15
        else float("nan")
    )

def max_drawdown(returns: pd.Series | list[float] | np.ndarray) -> float:
    values = _clean_returns(returns)
    if values.empty:
        return float("nan")
    wealth = (1.0 + values).cumprod()
    wealth = pd.concat([pd.Series([1.0]), wealth.reset_index(drop=True)])
    return float((wealth / wealth.cummax() - 1.0).min())

def calmar_ratio(returns: pd.Series | list[float] | np.ndarray) -> float:
    growth, drawdown = cagr(returns), max_drawdown(returns)
    if not np.isfinite(growth) or not np.isfinite(drawdown) or abs(drawdown) < 1e-15:
        return float("nan")
    return float(growth / abs(drawdown))

def portfolio_turnover(
    weights: pd.DataFrame,
    *,
    one_way: bool = False,
    initial_weights: pd.Series | Mapping[str, float] | None = None,
) -> pd.Series:
    if not isinstance(weights, pd.DataFrame):
        raise TypeError("weights must be a pandas DataFrame")
    numeric = weights.apply(pd.to_numeric, errors="coerce")
    changes = numeric.diff()
    if initial_weights is not None and not numeric.empty:
        initial = pd.Series(initial_weights, dtype=float).reindex(numeric.columns).fillna(0.0)
        changes.iloc[0] = numeric.iloc[0] - initial
    result = changes.abs().sum(axis=1, min_count=1).fillna(0.0)
    return (0.5 * result if one_way else result).rename("turnover")

def performance_summary(
    returns: pd.Series | list[float] | np.ndarray,
    bil_returns: pd.Series | list[float] | np.ndarray | None = None,
    turnover: pd.Series | list[float] | np.ndarray | None = None,
) -> dict[str, float]:
    values = _clean_returns(returns)
    turnover_values = (
        pd.Series(dtype=float)
        if turnover is None
        else pd.to_numeric(pd.Series(turnover), errors="coerce").dropna()
    )
    average_turnover = float(turnover_values.mean()) if not turnover_values.empty else float("nan")
    return {
        "cagr": cagr(values),
        "annualized_volatility": annualized_volatility(values),
        "bil_relative_sharpe": sharpe_ratio(values, bil_returns),
        "bil_relative_sortino": sortino_ratio(values, bil_returns),
        "max_drawdown": max_drawdown(values),
        "calmar": calmar_ratio(values),
        "average_monthly_turnover": average_turnover,
        "annualized_turnover": average_turnover * PERIODS_PER_YEAR,
        "months": float(len(values)),
    }

def metrics_table(
    returns: pd.DataFrame,
    *,
    bil_column: str = "BIL",
    turnover: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    bil = returns[bil_column] if bil_column in returns.columns else None
    rows = {
        str(column): performance_summary(
            returns[column], bil, turnover[column] if turnover is not None and column in turnover else None
        )
        for column in returns.columns
    }
    return pd.DataFrame.from_dict(rows, orient="index")

annualized_return = cagr
volatility = annualized_volatility
sharpe = sharpe_ratio
sortino = sortino_ratio
turnover = portfolio_turnover

# No-look-ahead portfolio simulation

@dataclass(frozen=True)
class BacktestResult:
    returns: pd.DataFrame
    weights: pd.DataFrame
    turnover: pd.DataFrame
    metrics: pd.DataFrame
    transaction_costs: pd.DataFrame
    policy: SuitabilityPolicy
    explanations: dict[Any, dict[str, Any]]

    def __getitem__(self, key: str) -> Any:
        if key not in self.keys():
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def keys(self) -> tuple[str, ...]:
        return ("returns", "weights", "turnover", "metrics", "transaction_costs", "policy", "explanations")

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.keys()}

def _validate_returns(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(monthly_returns, pd.DataFrame):
        raise TypeError("monthly_returns must be a pandas DataFrame")
    if monthly_returns.empty:
        raise ValueError("monthly_returns must not be empty")
    frame = monthly_returns.copy()
    frame.columns = [str(column).upper() for column in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("monthly_returns contains duplicate asset columns")
    missing = set(ALL_ASSETS).difference(frame.columns)
    if missing:
        raise ValueError(f"monthly_returns is missing required assets: {sorted(missing)}")
    if frame.index.has_duplicates:
        raise ValueError("monthly_returns index must contain unique decision periods")
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index()
    frame = frame.reindex(columns=ALL_ASSETS).apply(pd.to_numeric, errors="coerce")
    if not (np.isfinite(frame.to_numpy(dtype=float)) | frame.isna().to_numpy()).all():
        raise ValueError("monthly_returns contains infinite values")
    if (frame <= -1.0).any(axis=None):
        raise ValueError("monthly asset returns must be greater than -100%")
    return frame.astype(float)

def _validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame")
    if predictions.empty:
        raise ValueError("predictions must not be empty")
    frame = predictions.copy()
    frame.columns = [str(column).upper() for column in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("predictions contains duplicate asset columns")
    if frame.index.has_duplicates:
        frame = frame.groupby(level=0).last()
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index()
    return frame.reindex(columns=ALL_ASSETS).apply(pd.to_numeric, errors="coerce")

def _prediction_application_map(
    return_index: pd.Index, predictions: pd.DataFrame
) -> dict[int, pd.Series]:
    # Map every decision to the first strictly later monthly return.

    application: dict[int, pd.Series] = {}
    is_datetime = isinstance(return_index, pd.DatetimeIndex)
    for decision_date, row in predictions.iterrows():
        if is_datetime:
            try:
                decision_value = pd.Timestamp(decision_date)
                if return_index.tz is not None and decision_value.tzinfo is None:
                    decision_value = decision_value.tz_localize(return_index.tz)
                if return_index.tz is None and decision_value.tzinfo is not None:
                    decision_value = decision_value.tz_localize(None)
                position = int(return_index.searchsorted(decision_value, side="right"))
            except (TypeError, ValueError):
                continue
        else:
            try:
                location = return_index.get_loc(decision_date)
            except KeyError:
                continue
            if isinstance(location, (slice, np.ndarray)):
                continue
            position = int(location) + 1
        if 0 < position < len(return_index):
            application[position] = row
    return application

def _drift_weights(target: pd.Series, asset_returns: pd.Series) -> pd.Series:
    ending = target * (1.0 + asset_returns)
    total = float(ending.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("portfolio wealth became non-positive")
    return ending / total

def _fixed_weight_backtest(
    asset_returns: pd.DataFrame,
    target: pd.Series,
    transaction_cost_rate: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    target = target.reindex(ALL_ASSETS).fillna(0.0).astype(float)
    target /= float(target.sum())
    pretrade = target.copy()
    net_returns: list[float] = []
    turnovers: list[float] = []
    costs: list[float] = []
    for _, row in asset_returns.iterrows():
        dollar_turnover = float((target - pretrade).abs().sum())
        cost = transaction_cost_rate * dollar_turnover
        net_returns.append(float(target @ row) - cost)
        turnovers.append(dollar_turnover)
        costs.append(cost)
        pretrade = _drift_weights(target, row)
    index = asset_returns.index
    return pd.Series(net_returns, index=index), pd.Series(turnovers, index=index), pd.Series(costs, index=index)

def run_backtest(
    monthly_returns: pd.DataFrame,
    predictions: pd.DataFrame,
    profile: ClientProfile,
    *,
    transaction_cost_bps: float = 10.0,
    covariance_lookback_months: int = 36,
    tilt_strength: float = 0.25,
    turnover_blend: float = 0.50,
    model_signal_weight: float = 0.25,
    momentum_months: int = 9,
) -> BacktestResult:
    # Apply every decision strictly next month with drift and full-L1 costs.

    if not isinstance(profile, ClientProfile):
        raise TypeError("profile must be a ClientProfile")
    if not np.isfinite(transaction_cost_bps) or transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be finite and non-negative")
    if not isinstance(covariance_lookback_months, int) or covariance_lookback_months < 3:
        raise ValueError("covariance_lookback_months must be an integer of at least 3")
    returns = _validate_returns(monthly_returns)
    forecasts = _validate_predictions(predictions)
    application_map = _prediction_application_map(returns.index, forecasts)
    if not application_map:
        raise ValueError("no prediction has a later monthly return to evaluate")

    first_position = min(application_map)
    evaluation_returns = returns.iloc[first_position:].copy()
    if evaluation_returns.isna().any(axis=None):
        missing_dates = evaluation_returns.index[evaluation_returns.isna().any(axis=1)]
        raise ValueError(f"evaluation-period asset returns contain missing values at {list(missing_dates[:3])}")

    policy = build_policy(profile)
    strategic = pd.Series(policy.strategic_weights, dtype=float).reindex(ALL_ASSETS, fill_value=0.0)
    pretrade = strategic.copy()
    strategy_returns: list[float] = []
    strategy_turnover: list[float] = []
    strategy_costs: list[float] = []
    strategy_weights: list[pd.Series] = []
    explanations: dict[Any, dict[str, Any]] = {}
    transaction_cost_rate = float(transaction_cost_bps) / 10_000.0

    for position in range(first_position, len(returns)):
        return_date = returns.index[position]
        if position in application_map:
            trailing = returns.iloc[max(0, position - covariance_lookback_months):position]
            target, explanation = construct_portfolio(
                application_map[position],
                trailing,
                policy,
                current_weights=pretrade,
                tilt_strength=tilt_strength,
                turnover_blend=turnover_blend,
                model_signal_weight=model_signal_weight,
                momentum_months=momentum_months,
            )
            explanations[return_date] = explanation
        else:
            target = pretrade.copy()
        row = returns.iloc[position]
        dollar_turnover = float((target - pretrade).abs().sum())
        transaction_cost = transaction_cost_rate * dollar_turnover
        strategy_returns.append(float(target @ row) - transaction_cost)
        strategy_turnover.append(dollar_turnover)
        strategy_costs.append(transaction_cost)
        strategy_weights.append(target.rename(return_date))
        pretrade = _drift_weights(target, row)

    strategy_index = evaluation_returns.index
    weight_frame = pd.DataFrame(strategy_weights, index=strategy_index).reindex(columns=ALL_ASSETS)
    benchmark_targets = {
        "static_profile": strategic,
        "60_40": pd.Series({"SPY": 0.60, "AGG": 0.40}),
        "SPY": pd.Series({"SPY": 1.0}),
        "equal_risky": pd.Series(1.0 / len(RISKY_ASSETS), index=RISKY_ASSETS),
        "BIL": pd.Series({CASH_ASSET: 1.0}),
    }
    result_returns = pd.DataFrame({"strategy": strategy_returns}, index=strategy_index)
    result_turnover = pd.DataFrame({"strategy": strategy_turnover}, index=strategy_index)
    result_costs = pd.DataFrame({"strategy": strategy_costs}, index=strategy_index)
    for name, target in benchmark_targets.items():
        benchmark_return, benchmark_turnover, benchmark_cost = _fixed_weight_backtest(
            evaluation_returns, target, transaction_cost_rate
        )
        result_returns[name] = benchmark_return
        result_turnover[name] = benchmark_turnover
        result_costs[name] = benchmark_cost
    return BacktestResult(
        returns=result_returns,
        weights=weight_frame,
        turnover=result_turnover,
        metrics=metrics_table(result_returns, bil_column="BIL", turnover=result_turnover),
        transaction_costs=result_costs,
        policy=policy,
        explanations=explanations,
    )

backtest = run_backtest
run_walk_forward_backtest = run_backtest

# Complete experiment and minimal reporting artifacts

@dataclass(frozen=True)
class ExperimentResult:
    # In-memory outputs from one complete, reproducible experiment run.

    prices: pd.DataFrame
    predictions: pd.DataFrame
    retraining_log: pd.DataFrame
    backtests: dict[str, BacktestResult]
    reported_evaluation_metrics: dict[str, pd.DataFrame]
    ablations: pd.DataFrame
    forecast_diagnostics: dict[str, Any]
    latest_recommendation: dict[str, Any]
    results_dir: Path

    @property
    def holdout_metrics(self) -> dict[str, pd.DataFrame]:
        # Compatibility alias; these are reported evaluations, not pristine holdouts.

        return self.reported_evaluation_metrics

def _month_end(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).to_period("M").to_timestamp("M")

def _reported_metrics(
    result: BacktestResult, *, reported_start: str | pd.Timestamp
) -> pd.DataFrame:
    start = _month_end(reported_start)
    returns = result.returns.loc[result.returns.index >= start]
    if len(returns) < 24:
        raise ValueError("The reported evaluation must contain at least 24 months")
    return metrics_table(
        returns,
        bil_column="BIL",
        turnover=result.turnover.reindex(returns.index),
    )

def _forecast_diagnostics(
    prices: pd.DataFrame, predictions: pd.DataFrame
) -> dict[str, Any]:
    # Evaluate raw neural scores before the momentum anchor and policy rules.

    returns = prices.pct_change(fill_method=None)
    predicted_values: list[float] = []
    realized_values: list[float] = []
    monthly_rank_ic: list[float] = []
    evaluated_months = 0
    for decision_date, forecast_row in predictions.iterrows():
        next_position = int(returns.index.searchsorted(decision_date, side="right"))
        if next_position >= len(returns):
            continue
        realized = (
            returns.iloc[next_position].reindex(RISKY_ASSETS)
            - float(returns.iloc[next_position][CASH_ASSET])
        )
        valid = pd.concat([forecast_row.reindex(RISKY_ASSETS), realized], axis=1).dropna()
        if len(valid) < 2:
            continue
        predicted_values.extend(valid.iloc[:, 0].astype(float).tolist())
        realized_values.extend(valid.iloc[:, 1].astype(float).tolist())
        rank_ic = valid.iloc[:, 0].rank().corr(valid.iloc[:, 1].rank())
        if pd.notna(rank_ic):
            monthly_rank_ic.append(float(rank_ic))
        evaluated_months += 1
    predicted = pd.Series(predicted_values, dtype=float)
    realized = pd.Series(realized_values, dtype=float)
    error = predicted - realized
    zero_error = -realized
    rank_ic_series = pd.Series(monthly_rank_ic, dtype=float)
    return {
        "evaluated_months": evaluated_months,
        "asset_month_observations": int(len(realized)),
        "mean_absolute_error": float(error.abs().mean()),
        "zero_forecast_mean_absolute_error": float(zero_error.abs().mean()),
        "root_mean_squared_error": float((error.pow(2).mean()) ** 0.5),
        "directional_accuracy": float(((predicted > 0) == (realized > 0)).mean()),
        "mean_monthly_rank_information_coefficient": float(rank_ic_series.mean()),
        "positive_rank_ic_month_fraction": float((rank_ic_series > 0).mean()),
        "model_beats_zero_forecast_mae": bool(error.abs().mean() < zero_error.abs().mean()),
        "scope": "raw neural predictions before momentum anchor, constraints, and costs",
    }

def latest_recommendation(
    profile: ClientProfile,
    prices: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    # Construct the latest recommendation for any feasible profile.

    decision_date = pd.Timestamp(predictions.index[-1])
    monthly_returns = prices.pct_change(fill_method=None)
    policy = build_policy(profile)
    weights, explanation = construct_portfolio(
        predictions.loc[decision_date], monthly_returns.loc[:decision_date].tail(36), policy
    )
    return {
        "as_of": decision_date.strftime("%Y-%m-%d"),
        "profile": profile.to_dict(),
        "policy": policy.to_dict(),
        "predicted_excess_returns": predictions.loc[decision_date].round(8).to_dict(),
        "recommended_weights": weights.round(8).to_dict(),
        "explanation": explanation,
        "disclaimer": "Educational research prototype using a synthetic example profile; not investment, tax, or legal advice.",
    }

def _ablation_metrics(
    monthly_returns: pd.DataFrame,
    predictions: pd.DataFrame,
    profile: ClientProfile,
    ensemble_result: BacktestResult,
    *,
    reported_start: str | pd.Timestamp,
) -> pd.DataFrame:
    variants = {
        "ensemble_25_model_75_momentum": ensemble_result,
        "momentum_only": run_backtest(
            monthly_returns, predictions, profile, model_signal_weight=0.0
        ),
        "neural_only": run_backtest(
            monthly_returns, predictions, profile, model_signal_weight=1.0
        ),
    }
    start = _month_end(reported_start)
    rows: dict[str, dict[str, float]] = {}
    for name, result in variants.items():
        returns = result.returns.loc[result.returns.index >= start]
        rows[name] = performance_summary(
            returns["strategy"],
            returns["BIL"],
            result.turnover.loc[returns.index, "strategy"],
        )
    static_returns = ensemble_result.returns.loc[ensemble_result.returns.index >= start]
    rows["static_profile_no_signal"] = performance_summary(
        static_returns["static_profile"],
        static_returns["BIL"],
        ensemble_result.turnover.loc[static_returns.index, "static_profile"],
    )
    return pd.DataFrame.from_dict(rows, orient="index")

def _save_equity_curve(monthly_returns: pd.DataFrame, path: Path) -> None:
    curves = (1.0 + monthly_returns.fillna(0.0)).cumprod()
    labels = {
        "strategy": "Wealth Management ML Project",
        "static_profile": "Static profile",
        "60_40": "60/40",
        "SPY": "SPY",
    }
    columns = [column for column in labels if column in curves]
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        for column in columns:
            ax.plot(
                curves.index,
                curves[column],
                label=labels[column],
                linewidth=2.8 if column == "strategy" else 1.7,
                alpha=1.0 if column == "strategy" else 0.82,
            )
        ax.set_title("Balanced profile — reported evaluation growth of $1", loc="left", fontsize=15, fontweight="bold")
        ax.set_ylabel("Growth of $1")
        ax.set_xlabel("")
        ax.legend(frameon=False, ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

def _save_allocation_chart(weights: pd.Series, path: Path, as_of: str) -> None:
    ordered = weights[weights > 0.0005].sort_values()
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        bars = ax.barh(ordered.index, ordered.values * 100.0, color="#3B82F6")
        ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
        ax.set_xlim(0, max(10.0, float(ordered.max() * 115.0)))
        ax.set_xlabel("Portfolio weight")
        ax.set_title(f"Balanced profile allocation as of {as_of}", loc="left", fontsize=14, fontweight="bold")
        ax.spines[["top", "right", "left"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

def run_experiment(
    project_root: str | Path | None = None,
    *,
    force_download: bool = False,
    backtest_start: str = "2015-01-01",
    holdout_start: str = "2021-01-01",
    max_epochs: int = 220,
    patience: int = 25,
    quick: bool = False,
) -> ExperimentResult:
    # Download, train, backtest, and write the project's results.

    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parent
    cache_path = root / "data" / "cache" / "etf_monthly_prices.csv"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if quick:
        max_epochs = min(max_epochs, 60)
        patience = min(patience, 10)

    prices = download_prices(cache_path, force=force_download)
    panel = build_feature_panel(prices)
    predictions, retraining_log, latest_bundle = walk_forward_predictions(
        panel,
        backtest_start=backtest_start,
        max_epochs=max_epochs,
        patience=patience,
    )
    latest_bundle.save(results_dir / "model.pt")
    profiles = {
        name: ClientProfile.from_mapping(values) for name, values in PROFILE_PRESETS.items()
    }
    monthly_returns = prices.pct_change(fill_method=None)
    backtests = {
        name: run_backtest(monthly_returns, predictions, profile)
        for name, profile in profiles.items()
    }
    reported_metrics = {
        name: _reported_metrics(result, reported_start=holdout_start)
        for name, result in backtests.items()
    }
    balanced = backtests["balanced"]
    reported_start = _month_end(holdout_start)
    balanced_evaluation_returns = balanced.returns.loc[balanced.returns.index >= reported_start]
    ablations = _ablation_metrics(
        monthly_returns,
        predictions,
        profiles["balanced"],
        balanced,
        reported_start=holdout_start,
    )
    diagnostics = _forecast_diagnostics(prices, predictions)
    recommendation = latest_recommendation(profiles["balanced"], prices, predictions)

    predictions.to_csv(results_dir / "walk_forward_predictions.csv", index_label="decision_date")
    retraining_log.to_csv(results_dir / "retraining_log.csv", index=False)
    balanced_evaluation_returns.to_csv(results_dir / "reported_evaluation_returns.csv", index_label="date")
    reported_metrics["balanced"].to_csv(results_dir / "reported_evaluation_metrics.csv", index_label="portfolio")
    pd.concat(reported_metrics, names=["profile", "portfolio"]).to_csv(
        results_dir / "all_profile_reported_evaluation_metrics.csv"
    )
    ablations.to_csv(results_dir / "ablations.csv", index_label="variant")
    pd.DataFrame([diagnostics]).to_csv(results_dir / "forecast_diagnostics.csv", index=False)
    latest_weights = pd.Series(recommendation["recommended_weights"], dtype=float)
    latest_table = pd.DataFrame(
        {
            "recommended_weight": latest_weights,
            "strategic_weight": pd.Series(recommendation["policy"]["strategic_weights"], dtype=float),
            "predicted_excess_return": pd.Series(recommendation["predicted_excess_returns"], dtype=float),
        }
    ).reindex(ALL_ASSETS)
    latest_table.index.name = "asset"
    latest_table.to_csv(results_dir / "latest_allocation.csv")
    _save_equity_curve(balanced_evaluation_returns, results_dir / "equity_curve.png")
    _save_allocation_chart(latest_weights, results_dir / "latest_allocation.png", recommendation["as_of"])

    return ExperimentResult(
        prices=prices,
        predictions=predictions,
        retraining_log=retraining_log,
        backtests=backtests,
        reported_evaluation_metrics=reported_metrics,
        ablations=ablations,
        forecast_diagnostics=diagnostics,
        latest_recommendation=recommendation,
        results_dir=results_dir,
    )

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download data, train PyTorch models, and run the project."
    )
    parser.add_argument("--force-download", action="store_true", help="Refresh Yahoo data instead of using the cache.")
    parser.add_argument("--backtest-start", default="2015-01-01", help="First model decision month.")
    parser.add_argument("--holdout-start", default="2021-01-01", help="Start of the reported evaluation period.")
    parser.add_argument("--quick", action="store_true", help="Use fewer epochs for a fast smoke run.")
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> ExperimentResult:
    args = parse_args(argv)
    result = run_experiment(
        force_download=args.force_download,
        backtest_start=args.backtest_start,
        holdout_start=args.holdout_start,
        quick=args.quick,
    )
    metrics = result.reported_evaluation_metrics["balanced"]
    strategy = metrics.loc["strategy"]
    static = metrics.loc["static_profile"]
    ablations = result.ablations
    neural_gate = (
        ablations.loc["ensemble_25_model_75_momentum", "bil_relative_sharpe"]
        > ablations.loc["momentum_only", "bil_relative_sharpe"]
    )
    print("\nProject experiment complete")
    print(f"Data: {result.prices.index[0]:%b %Y} to {result.prices.index[-1]:%b %Y} ({len(result.prices)} complete months)")
    print(f"Balanced reported-evaluation CAGR: {strategy['cagr']:.2%} vs {static['cagr']:.2%} static profile")
    print(f"Balanced BIL-relative Sharpe: {strategy['bil_relative_sharpe']:.2f} vs {static['bil_relative_sharpe']:.2f} static profile")
    print(f"Neural incremental-value gate: {'passed' if neural_gate else 'failed — challenger is not deployment-ready'}")
    print(f"Results: {result.results_dir}")
    return result

__all__ = [
    "BacktestResult",
    "ClientProfile",
    "ExperimentResult",
    "InfeasibleProfileError",
    "PROFILE_PRESETS",
    "SuitabilityPolicy",
    "annualized_return",
    "annualized_volatility",
    "assess_suitability",
    "backtest",
    "build_policy",
    "build_portfolio",
    "cagr",
    "calmar_ratio",
    "construct_portfolio",
    "latest_recommendation",
    "max_drawdown",
    "metrics_table",
    "performance_summary",
    "portfolio_turnover",
    "recommend_portfolio",
    "run_backtest",
    "run_experiment",
    "run_walk_forward_backtest",
    "sharpe_ratio",
    "sortino_ratio",
]

if __name__ == "__main__":
    main()
