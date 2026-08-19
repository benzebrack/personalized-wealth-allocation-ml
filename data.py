"""
This file keeps the entire data layer in one place - the small ETF
universe, Yahoo Finance download/cache logic, monthly validation, and the 26
features used by the PyTorch model. Every feature on a row dated t uses
only prices observed on or before t.  The label is the asset's return from
t to t+1 minus the return on BIL over the same month.
"""

from __future__ import annotations

from datetime import date, datetime
import os
from pathlib import Path
from typing import Final, TypeAlias

import numpy as np
import pandas as pd

RISKY_ASSETS: Final[tuple[str, ...]] = (
    "SPY",  # US large-cap equities
    "EFA",  # developed-market equities outside the US and Canada
    "EEM",  # emerging-market equities
    "AGG",  # US investment-grade bonds
    "TIP",  # US inflation-protected Treasuries
    "VNQ",  # US real estate investment trusts
    "GLD",  # gold
)
CASH_ASSET: Final[str] = "BIL"
ALL_ASSETS: Final[tuple[str, ...]] = (*RISKY_ASSETS, CASH_ASSET)
GROWTH_ASSETS: Final[tuple[str, ...]] = ("SPY", "EFA", "EEM", "VNQ")
DEFENSIVE_ASSETS: Final[tuple[str, ...]] = ("AGG", "TIP", "GLD")

DEFAULT_START_DATE: Final[str] = "2007-01-01"
RETURN_WINDOWS: Final[tuple[int, ...]] = (1, 3, 6, 12)
VOLATILITY_WINDOWS: Final[tuple[int, ...]] = (3, 6, 12)

ASSET_RETURN_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"asset_return_{window}m" for window in RETURN_WINDOWS
)
ASSET_VOLATILITY_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"asset_volatility_{window}m" for window in VOLATILITY_WINDOWS
)
ASSET_STATE_COLUMNS: Final[tuple[str, ...]] = (
    "asset_trend_10m",
    "asset_drawdown_12m",
    "asset_relative_return_1m",
)
SPY_CONTEXT_COLUMNS: Final[tuple[str, ...]] = (
    *(f"spy_return_{window}m" for window in RETURN_WINDOWS),
    *(f"spy_volatility_{window}m" for window in VOLATILITY_WINDOWS),
    "spy_trend_10m",
    "spy_drawdown_12m",
)
ASSET_INDICATOR_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"asset_is_{asset.lower()}" for asset in RISKY_ASSETS
)

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    *ASSET_RETURN_COLUMNS,
    *ASSET_VOLATILITY_COLUMNS,
    *ASSET_STATE_COLUMNS,
    *SPY_CONTEXT_COLUMNS,
    *ASSET_INDICATOR_COLUMNS,
)
TARGET_COLUMN: Final[str] = "target_excess_return"

DateLike: TypeAlias = str | date | datetime | pd.Timestamp
PathLike: TypeAlias = str | os.PathLike[str]

def _normalise_close_frame(raw: pd.DataFrame) -> pd.DataFrame:
    # Extract one adjusted-close column per ticker across yfinance versions.
    if raw.empty:
        raise ValueError("Yahoo Finance returned no price observations.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw.xs("Close", axis=1, level=0)
        elif "Close" in raw.columns.get_level_values(1):
            close = raw.xs("Close", axis=1, level=1)
        else:
            raise ValueError("The download does not contain adjusted close prices.")
    elif "Close" in raw.columns and len(ALL_ASSETS) == 1:
        close = raw[["Close"]].rename(columns={"Close": ALL_ASSETS[0]})
    else:
        # This also makes simple mocked downloads convenient in unit tests.
        close = raw.copy()

    close.columns = pd.Index([str(column).upper() for column in close.columns])
    missing = [asset for asset in ALL_ASSETS if asset not in close.columns]
    if missing:
        raise ValueError("Missing close prices for required assets: " + ", ".join(missing))

    close = close.loc[:, list(ALL_ASSETS)].apply(pd.to_numeric, errors="coerce")
    index = pd.DatetimeIndex(pd.to_datetime(close.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    close.index = index.normalize()
    return close.sort_index()

def _to_complete_month_ends(
    daily_close: pd.DataFrame, *, end: DateLike | None
) -> pd.DataFrame:
    # Keep the last trading close in each fully completed calendar month.

    months = daily_close.index.to_period("M")
    monthly = daily_close.groupby(months, sort=True).last()
    monthly.index = pd.PeriodIndex(monthly.index, freq="M").to_timestamp("M")

    # With no explicit end, the current partial month cannot enter the cache.
    exclusive_end = (
        pd.Timestamp(end).tz_localize(None).normalize()
        if end is not None
        else pd.Timestamp.now().tz_localize(None).normalize()
    )
    return monthly.loc[monthly.index < exclusive_end].dropna(axis=0, how="any")

def validate_prices(prices: pd.DataFrame, *, source: str = "prices") -> pd.DataFrame:
    # Validate a monthly price frame and return canonical month-end values.

    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError(f"{source} contains no complete monthly observations.")

    frame = prices.copy()
    frame.columns = pd.Index([str(column).upper() for column in frame.columns])
    missing = [asset for asset in ALL_ASSETS if asset not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required assets: {', '.join(missing)}")

    frame = frame.loc[:, list(ALL_ASSETS)].apply(pd.to_numeric, errors="coerce")
    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    frame.index = index.to_period("M").to_timestamp("M")
    frame.index.name = "date"
    frame = frame.sort_index()

    if frame.index.has_duplicates:
        raise ValueError(f"{source} contains duplicate monthly observations.")
    if frame.isna().any(axis=None):
        bad = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"{source} contains missing values for: {', '.join(bad)}")

    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{source} contains non-finite prices.")
    if (values <= 0.0).any():
        raise ValueError(f"{source} contains zero or negative prices.")
    return frame.astype(float)

def _slice_requested_range(
    prices: pd.DataFrame, *, start: DateLike, end: DateLike | None
) -> pd.DataFrame:
    start_timestamp = pd.Timestamp(start).tz_localize(None).normalize()
    selected = prices.loc[prices.index >= start_timestamp]
    if end is not None:
        end_timestamp = pd.Timestamp(end).tz_localize(None).normalize()
        selected = selected.loc[selected.index < end_timestamp]
    if selected.empty:
        raise ValueError("No complete monthly prices fall inside the requested range.")
    return selected

def download_prices(
    cache_path: PathLike,
    start: DateLike = DEFAULT_START_DATE,
    end: DateLike | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Yahoo Finance is convenient for a research demonstration so is used.
    auto_adjust=True incorporates splits and cash distributions into the return
    history. The current incomplete month is excluded, and validated results are
    written to a local CSV cache.
    """

    cache = Path(cache_path).expanduser()
    if cache.exists() and not force:
        cached = pd.read_csv(cache, index_col=0, parse_dates=True)
        validated = validate_prices(cached, source=f"Price cache {cache}")
        return _slice_requested_range(validated, start=start, end=end)

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "Downloading prices requires yfinance. Install the project requirements."
        ) from exc

    raw = yf.download(
        tickers=list(ALL_ASSETS),
        start=pd.Timestamp(start).strftime("%Y-%m-%d"),
        end=pd.Timestamp(end).strftime("%Y-%m-%d") if end is not None else None,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    daily_close = _normalise_close_frame(raw)
    monthly = _to_complete_month_ends(daily_close, end=end)
    validated = validate_prices(monthly, source="Downloaded prices")

    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_name(f".{cache.name}.tmp")
    validated.to_csv(temporary, index_label="date")
    temporary.replace(cache)
    return _slice_requested_range(validated, start=start, end=end)

def _validated_feature_prices(prices: pd.DataFrame) -> pd.DataFrame:
    # Add the consecutive-month requirement needed for safe return windows.

    frame = validate_prices(prices, source="prices")
    periods = frame.index.to_period("M")
    expected = pd.period_range(periods[0], periods[-1], freq="M")
    if not periods.equals(expected):
        raise ValueError("prices must contain consecutive monthly observations.")
    return frame

def build_feature_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Creating the 26-feature asset-month panel without look-ahead leakage.
    Rows are keyed by feature_date and risky asset. Returns, volatility,
    trend, drawdown, relative strength, broad SPY context, and asset identity are
    all calculated from information available by that month-end. The latest
    seven rows intentionally retain a missing target so they can be scored for
    a current recommendation.
    """

    frame = _validated_feature_prices(prices)
    risky_prices = frame.loc[:, list(RISKY_ASSETS)]

    returns = {
        window: risky_prices.pct_change(periods=window, fill_method=None)
        for window in RETURN_WINDOWS
    }
    monthly_returns = returns[1]
    volatilities = {
        window: monthly_returns.rolling(window, min_periods=window).std(ddof=1)
        * np.sqrt(12.0)
        for window in VOLATILITY_WINDOWS
    }
    trend = risky_prices.divide(
        risky_prices.rolling(10, min_periods=10).mean()
    ).subtract(1.0)
    drawdown = risky_prices.divide(
        risky_prices.rolling(12, min_periods=12).max()
    ).subtract(1.0)
    relative_return = monthly_returns.subtract(monthly_returns.mean(axis=1), axis=0)

    next_risky_return = risky_prices.shift(-1).divide(risky_prices).subtract(1.0)
    next_cash_return = (
        frame[CASH_ASSET].shift(-1).divide(frame[CASH_ASSET]).subtract(1.0)
    )
    excess_target = next_risky_return.subtract(next_cash_return, axis=0)

    context: dict[str, pd.Series] = {
        **{
            f"spy_return_{window}m": returns[window]["SPY"]
            for window in RETURN_WINDOWS
        },
        **{
            f"spy_volatility_{window}m": volatilities[window]["SPY"]
            for window in VOLATILITY_WINDOWS
        },
        "spy_trend_10m": trend["SPY"],
        "spy_drawdown_12m": drawdown["SPY"],
    }

    panels: list[pd.DataFrame] = []
    for asset in RISKY_ASSETS:
        asset_panel = pd.DataFrame(index=frame.index)
        asset_panel["feature_date"] = frame.index
        asset_panel["target_date"] = frame.index + pd.offsets.MonthEnd(1)
        asset_panel["asset"] = asset

        for window, column in zip(RETURN_WINDOWS, ASSET_RETURN_COLUMNS, strict=True):
            asset_panel[column] = returns[window][asset]
        for window, column in zip(
            VOLATILITY_WINDOWS, ASSET_VOLATILITY_COLUMNS, strict=True
        ):
            asset_panel[column] = volatilities[window][asset]
        asset_panel["asset_trend_10m"] = trend[asset]
        asset_panel["asset_drawdown_12m"] = drawdown[asset]
        asset_panel["asset_relative_return_1m"] = relative_return[asset]

        for column in SPY_CONTEXT_COLUMNS:
            asset_panel[column] = context[column]
        for candidate, column in zip(
            RISKY_ASSETS, ASSET_INDICATOR_COLUMNS, strict=True
        ):
            asset_panel[column] = float(asset == candidate)

        asset_panel[TARGET_COLUMN] = excess_target[asset]
        panels.append(asset_panel)

    panel = pd.concat(panels, axis=0, ignore_index=True)
    finite_features = np.isfinite(
        panel.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    ).all(axis=1)
    panel = panel.loc[finite_features].copy()
    panel = panel.sort_values(["feature_date", "asset"], kind="stable")
    panel = panel.reset_index(drop=True)
    ordered = [
        "feature_date",
        "target_date",
        "asset",
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]
    return panel.loc[:, ordered]

__all__ = [
    "ALL_ASSETS",
    "CASH_ASSET",
    "DEFENSIVE_ASSETS",
    "FEATURE_COLUMNS",
    "GROWTH_ASSETS",
    "RISKY_ASSETS",
    "TARGET_COLUMN",
    "build_feature_panel",
    "download_prices",
    "validate_prices",
]
