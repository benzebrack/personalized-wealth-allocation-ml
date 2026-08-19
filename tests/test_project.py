# Offline checks for the Wealth Management ML project.

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import backtest as backtest_module  # noqa: E402
import model as model_module  # noqa: E402
from backtest import (  # noqa: E402
    PROFILE_PRESETS,
    ClientProfile,
    InfeasibleProfileError,
    build_policy,
    construct_portfolio,
    run_backtest,
)
from data import (  # noqa: E402
    ALL_ASSETS,
    CASH_ASSET,
    FEATURE_COLUMNS,
    GROWTH_ASSETS,
    RISKY_ASSETS,
    TARGET_COLUMN,
    build_feature_panel,
)
from model import ModelBundle, train_model, walk_forward_predictions  # noqa: E402

def synthetic_prices(months: int = 52) -> pd.DataFrame:
    # Create deterministic, complete month-end prices without network access.

    dates = pd.period_range("2012-01", periods=months, freq="M").to_timestamp("M")
    phase = np.arange(months, dtype=float)
    monthly = pd.DataFrame(index=dates, columns=ALL_ASSETS, dtype=float)
    for position, asset in enumerate(ALL_ASSETS):
        if asset == CASH_ASSET:
            monthly[asset] = 0.001 + 0.00005 * np.sin(phase / 7.0)
        else:
            monthly[asset] = (
                0.003
                + 0.00035 * position
                + 0.008 * np.sin(phase / (2.7 + position * 0.15) + position)
            )
    prices = 100.0 * (1.0 + monthly).cumprod()
    prices.index.name = "date"
    return prices


def client_profile(**overrides: object) -> ClientProfile:
    values: dict[str, object] = {
        "name": "Balanced test household",
        "age": 32,
        "investable_assets": 200_000.0,
        "annual_income": 125_000.0,
        "monthly_contribution": 1_500.0,
        "goal_amount": 650_000.0,
        "horizon_years": 15.0,
        "risk_tolerance": 3,
        "max_drawdown_tolerance": 0.25,
        "monthly_expenses": 3_000.0,
        "liquidity_needed_12m": 10_000.0,
        "emergency_cash_held": 7_000.0,
        "income_stability": "medium",
        "objective": "growth",
        "excluded_assets": (),
        "max_single_asset_weight": 0.45,
    }
    values.update(overrides)
    return ClientProfile(**values)


def test_project_has_exactly_four_root_python_files() -> None:
    assert {path.name for path in PROJECT_ROOT.glob("*.py")} == {
        "app.py",
        "backtest.py",
        "data.py",
        "model.py",
    }


def test_targets_are_next_month_and_future_prices_do_not_change_features() -> None:
    prices = synthetic_prices()
    panel = build_feature_panel(prices)
    expected_dates = panel["feature_date"] + pd.offsets.MonthEnd(1)
    pd.testing.assert_series_equal(panel["target_date"], expected_dates, check_names=False)

    feature_date = pd.DatetimeIndex(panel["feature_date"].unique())[-6]
    target_date = feature_date + pd.offsets.MonthEnd(1)
    rows = panel.loc[panel["feature_date"] == feature_date].set_index("asset")
    cash_return = prices.loc[target_date, CASH_ASSET] / prices.loc[feature_date, CASH_ASSET] - 1.0
    for asset in RISKY_ASSETS:
        risky_return = prices.loc[target_date, asset] / prices.loc[feature_date, asset] - 1.0
        assert rows.loc[asset, TARGET_COLUMN] == pytest.approx(risky_return - cash_return)

    shocked = prices.copy()
    future = shocked.index > feature_date
    shocked.loc[future] = shocked.loc[future].mul(
        np.linspace(1.25, 2.0, len(ALL_ASSETS)), axis="columns"
    )
    rebuilt = build_feature_panel(shocked)
    before = panel.loc[panel["feature_date"] == feature_date].sort_values("asset")
    after = rebuilt.loc[rebuilt["feature_date"] == feature_date].sort_values("asset")
    np.testing.assert_allclose(before.loc[:, FEATURE_COLUMNS], after.loc[:, FEATURE_COLUMNS], atol=0.0, rtol=0.0)
    assert not np.allclose(before[TARGET_COLUMN], after[TARGET_COLUMN])

    latest = panel.loc[panel["feature_date"] == panel["feature_date"].max()]
    assert set(latest["asset"]) == set(RISKY_ASSETS)
    assert latest[TARGET_COLUMN].isna().all()

@pytest.mark.parametrize(
    "override",
    [
        {"name": "  "},
        {"age": 17},
        {"investable_assets": 0.0},
        {"risk_tolerance": True},
        {"max_drawdown_tolerance": 0.0},
        {"income_stability": "uncertain"},
        {"objective": "speculation"},
        {"excluded_assets": ("QQQ",)},
    ],
)
def test_profile_rejects_unsupported_inputs(override: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        client_profile(**override)


def test_policy_enforces_liquidity_exclusions_and_abstention() -> None:
    policy = build_policy(
        client_profile(excluded_assets=("eem", "GLD"), max_single_asset_weight=0.30)
    )
    strategic = pd.Series(policy.strategic_weights).reindex(ALL_ASSETS)

    assert policy.cash_floor == pytest.approx(0.15)
    assert strategic.sum() == pytest.approx(1.0)
    assert (strategic >= 0.0).all()
    assert strategic["EEM"] == 0.0
    assert strategic["GLD"] == 0.0
    assert strategic[CASH_ASSET] >= policy.cash_floor
    assert (strategic.drop(CASH_ASSET) <= 0.30 + 1e-12).all()

    high_cash_profile = client_profile(
        monthly_expenses=0.0,
        emergency_cash_held=0.0,
        liquidity_needed_12m=190_000.0,
    )
    high_cash_policy = build_policy(high_cash_profile)
    assert high_cash_policy.cash_floor == pytest.approx(0.95)
    assert not high_cash_policy.actionable
    with pytest.raises(InfeasibleProfileError, match="95%"):
        build_policy(high_cash_profile, raise_on_abstention=True)

@pytest.mark.parametrize("preset_name", PROFILE_PRESETS)
def test_every_embedded_profile_preset_is_feasible(preset_name: str) -> None:
    profile = ClientProfile.from_mapping(PROFILE_PRESETS[preset_name])
    policy = build_policy(profile)
    weights = pd.Series(policy.strategic_weights).reindex(ALL_ASSETS)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0.0).all()
    assert weights[CASH_ASSET] >= policy.cash_floor - 1e-12


def test_portfolio_honors_every_hard_constraint_and_safe_fallback() -> None:
    policy = build_policy(
        client_profile(excluded_assets=("EEM", "GLD"), max_single_asset_weight=0.30)
    )
    forecasts = pd.Series(
        {"SPY": 5.0, "EFA": -4.0, "EEM": 100.0, "AGG": -3.0,
         "TIP": 4.0, "VNQ": 2.0, "GLD": 100.0, "BIL": 0.0}
    )
    trailing = synthetic_prices(40).pct_change(fill_method=None).dropna()
    weights, explanation = construct_portfolio(
        forecasts, trailing, policy, tilt_strength=0.25, turnover_blend=1.0
    )

    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= -1e-12).all()
    assert weights[CASH_ASSET] >= policy.cash_floor - 1e-12
    assert weights.reindex(GROWTH_ASSETS).sum() <= policy.growth_cap + 1e-12
    assert (weights.drop(CASH_ASSET) <= policy.max_single_asset_weight + 1e-12).all()
    assert weights["EEM"] == 0.0 and weights["GLD"] == 0.0
    assert explanation["forecast_used"]

    fallback_policy = build_policy(client_profile())
    fallback, fallback_explanation = construct_portfolio({"SPY": 0.02}, None, fallback_policy)
    expected = pd.Series(fallback_policy.strategic_weights).reindex(ALL_ASSETS)
    np.testing.assert_allclose(fallback, expected, atol=1e-12, rtol=0.0)
    assert not fallback_explanation["forecast_used"]

def test_backtest_applies_forecast_next_month_and_deducts_stated_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.period_range("2020-10", periods=7, freq="M").to_timestamp("M")
    phase = np.arange(len(dates), dtype=float)
    monthly_returns = pd.DataFrame(
        {
            asset: (
                np.full(len(dates), 0.001)
                if asset == CASH_ASSET
                else 0.001 + 0.0001 * position + 0.004 * np.sin(phase + position)
            )
            for position, asset in enumerate(ALL_ASSETS)
        },
        index=dates,
    )
    decision_dates = dates[3:6]
    predictions = pd.DataFrame(
        [
            [0.08, -0.05, 0.01, -0.04, 0.06, 0.03, -0.02, 0.0],
            [-0.07, 0.09, 0.02, 0.05, -0.03, -0.01, 0.07, 0.0],
            [0.02, 0.01, -0.08, 0.08, 0.04, 0.07, -0.06, 0.0],
        ],
        index=decision_dates,
        columns=ALL_ASSETS,
    )

    real_construct = backtest_module.construct_portfolio
    seen: list[pd.Series] = []

    def recording_construct(forecasts: pd.Series, trailing: pd.DataFrame, policy: object,
                            current_weights: pd.Series | None = None, **kwargs: object) -> tuple[pd.Series, dict]:
        seen.append(forecasts.copy())
        return real_construct(forecasts, trailing, policy, current_weights=current_weights, **kwargs)

    monkeypatch.setattr(backtest_module, "construct_portfolio", recording_construct)
    result = run_backtest(monthly_returns, predictions, client_profile(), transaction_cost_bps=25.0, turnover_blend=1.0)

    assert list(result.returns.index) == list(decision_dates + pd.offsets.MonthEnd(1))
    assert len(seen) == len(predictions)
    np.testing.assert_allclose(result.weights.sum(axis=1), 1.0, atol=1e-10)
    assert (result.weights >= -1e-12).all(axis=None)
    np.testing.assert_allclose(
        result.transaction_costs["strategy"], result.turnover["strategy"] * 0.0025, atol=1e-15, rtol=0.0
    )
    assert {"strategy", "static_profile", "60_40", "SPY", "equal_risky", "BIL"}.issubset(result.metrics.index)

def test_pytorch_training_and_checkpoint_round_trip(tmp_path: Path) -> None:
    panel = build_feature_panel(synthetic_prices(48))
    bundle = train_model(panel, validation_months=4, max_epochs=2, patience=2, batch_size=128, seed=11)
    latest = panel.loc[panel["feature_date"] == panel["feature_date"].max()]
    predictions = bundle.predict(latest)

    assert isinstance(bundle, ModelBundle)
    assert len(predictions) == len(RISKY_ASSETS)
    assert np.isfinite(predictions).all()
    assert bundle.metadata["dropped_unlabelled_observations"] == len(RISKY_ASSETS)

    restored = ModelBundle.load(bundle.save(tmp_path / "tiny_model.pt"))
    np.testing.assert_allclose(restored.predict(latest), predictions, atol=0.0, rtol=0.0)
    assert restored.metadata == bundle.metadata

def test_walk_forward_never_trains_on_an_unrealized_target(monkeypatch: pytest.MonkeyPatch) -> None:
    panel = build_feature_panel(synthetic_prices(56))
    feature_dates = pd.DatetimeIndex(sorted(panel["feature_date"].unique()))
    backtest_start = feature_dates[14]
    training_calls: list[pd.DataFrame] = []

    class RecordingBundle:
        def __init__(self, training_frame: pd.DataFrame, seed: int) -> None:
            dates = pd.DatetimeIndex(training_frame["feature_date"].unique()).sort_values()
            self.metadata = {
                "trained_through": dates[-1].strftime("%Y-%m-%d"),
                "refit_observations": len(training_frame),
                "selected_epoch": 1,
                "best_validation_loss": 0.0,
                "validation_start": dates[-12].strftime("%Y-%m-%d"),
                "validation_end": dates[-1].strftime("%Y-%m-%d"),
                "seed": seed,
            }

        def predict(self, scoring: pd.DataFrame) -> pd.Series:
            return pd.Series(np.linspace(-0.01, 0.01, len(scoring)), index=scoring.index)

    def recording_train(training_frame: pd.DataFrame, *, seed: int, **_: object) -> RecordingBundle:
        training_calls.append(training_frame.copy())
        return RecordingBundle(training_frame, seed)

    monkeypatch.setattr(model_module, "train_model", recording_train)
    predictions, log, _ = walk_forward_predictions(
        panel,
        backtest_start=backtest_start,
        retrain_every_months=4,
        minimum_training_months=13,
        max_epochs=1,
        patience=1,
        seed=19,
    )

    assert len(training_calls) == len(log) >= 2
    for training_frame, event in zip(training_calls, log.to_dict("records"), strict=True):
        decision_date = pd.Timestamp(event["decision_date"])
        assert training_frame[TARGET_COLUMN].notna().all()
        assert (training_frame["target_date"] <= decision_date).all()
        assert training_frame["feature_date"].max() < decision_date
    assert predictions.index[0] == pd.Timestamp(backtest_start)
    assert tuple(predictions.columns) == ALL_ASSETS
    assert predictions[CASH_ASSET].eq(0.0).all()
