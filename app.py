# Profile -> Constrained allocation

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from backtest import (
    PROFILE_PRESETS,
    ClientProfile,
    InfeasibleProfileError,
    build_policy,
    construct_portfolio,
    metrics_table,
    run_backtest,
)
from data import ALL_ASSETS, CASH_ASSET, GROWTH_ASSETS, download_prices


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CACHE = ROOT / "data" / "cache" / "etf_monthly_prices.csv"
PREDICTIONS = RESULTS / "walk_forward_predictions.csv"
REPORTED_START = pd.Timestamp("2021-01-01")

ASSET_NAMES = {
    "SPY": "US large-cap stocks",
    "EFA": "Developed international stocks",
    "EEM": "Emerging-market stocks",
    "AGG": "US investment-grade bonds",
    "TIP": "Inflation-protected Treasuries",
    "VNQ": "US real estate",
    "GLD": "Gold",
    "BIL": "Treasury bills / cash",
}


@st.cache_data(show_spinner=False)
def load_research_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Load the reproducible local cache and previously generated forecasts.

    prices = download_prices(CACHE)
    predictions = pd.read_csv(PREDICTIONS, index_col=0, parse_dates=True)
    predictions.index = pd.DatetimeIndex(predictions.index, name="decision_date")
    predictions.columns = [str(column).upper() for column in predictions.columns]
    return prices, predictions.reindex(columns=ALL_ASSETS)


def evaluation_metrics(result: object) -> pd.DataFrame:
    # Recompute summary metrics for the clearly labeled reported period.

    returns = result.returns.loc[result.returns.index >= REPORTED_START]
    turnover = result.turnover.reindex(returns.index)
    return metrics_table(returns, bil_column=CASH_ASSET, turnover=turnover)


def format_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "strategy": "Wealth Management ML Project",
        "static_profile": "Static profile",
        "60_40": "60/40",
        "SPY": "SPY",
        "equal_risky": "Equal risky",
        "BIL": "BIL / cash",
    }
    columns = [
        "cagr",
        "annualized_volatility",
        "bil_relative_sharpe",
        "max_drawdown",
        "annualized_turnover",
    ]
    shown = metrics.reindex(columns=columns).copy()
    shown.index = [labels.get(str(name), str(name)) for name in shown.index]
    shown.columns = ["CAGR", "Volatility", "Sharpe vs BIL", "Max drawdown", "Annual turnover"]
    for column in ["CAGR", "Volatility", "Max drawdown", "Annual turnover"]:
        shown[column] = shown[column].map(
            lambda value: "—" if pd.isna(value) else f"{float(value):.1%}"
        )
    shown["Sharpe vs BIL"] = shown["Sharpe vs BIL"].map(
        lambda value: "—" if pd.isna(value) else f"{float(value):.2f}"
    )
    return shown


def profile_form(defaults: dict[str, object], preset_name: str) -> ClientProfile | None:
    # Render the supported profile schema and return it after submission.

    suffix = preset_name.lower().replace(" ", "_")
    with st.form(f"profile_form_{suffix}"):
        st.markdown("#### Client facts")
        name = st.text_input("Scenario name", str(defaults["name"]), key=f"name_{suffix}")
        age = st.number_input("Age (context only)", 18, 100, int(defaults["age"]), key=f"age_{suffix}")
        investable_assets = st.number_input(
            "Investable assets ($)",
            1_000.0,
            100_000_000.0,
            float(defaults["investable_assets"]),
            step=10_000.0,
            key=f"assets_{suffix}",
        )
        annual_income = st.number_input(
            "Annual income ($; context only)",
            0.0,
            10_000_000.0,
            float(defaults["annual_income"]),
            step=5_000.0,
            key=f"income_{suffix}",
        )
        monthly_contribution = st.number_input(
            "Monthly contribution ($)",
            0.0,
            1_000_000.0,
            float(defaults["monthly_contribution"]),
            step=250.0,
            key=f"contribution_{suffix}",
        )
        goal_amount = st.number_input(
            "Goal amount ($)",
            1_000.0,
            1_000_000_000.0,
            float(defaults["goal_amount"]),
            step=25_000.0,
            key=f"goal_{suffix}",
        )
        horizon_years = st.slider(
            "Investment horizon (years)", 1, 50, int(defaults["horizon_years"]), key=f"horizon_{suffix}"
        )
        risk_tolerance = st.slider(
            "Risk tolerance", 1, 5, int(defaults["risk_tolerance"]),
            help="1 = lowest willingness; 5 = highest", key=f"risk_{suffix}"
        )
        max_drawdown = st.slider(
            "Maximum tolerable drawdown",
            5,
            60,
            int(round(float(defaults["max_drawdown_tolerance"]) * 100)),
            format="%d%%",
            help="A policy input, not a guarantee that losses cannot exceed it.",
            key=f"drawdown_{suffix}",
        )

        st.markdown("#### Liquidity and constraints")
        monthly_expenses = st.number_input(
            "Monthly expenses ($)", 0.0, 1_000_000.0, float(defaults["monthly_expenses"]),
            step=500.0, key=f"expenses_{suffix}"
        )
        liquidity_needed = st.number_input(
            "Other liquidity needed in 12 months ($)",
            0.0,
            100_000_000.0,
            float(defaults["liquidity_needed_12m"]),
            step=5_000.0,
            key=f"liquidity_{suffix}",
        )
        emergency_cash = st.number_input(
            "Emergency cash held outside this portfolio ($)",
            0.0,
            100_000_000.0,
            float(defaults["emergency_cash_held"]),
            step=5_000.0,
            key=f"cash_{suffix}",
        )
        stability_values = ["low", "medium", "high"]
        income_stability = st.selectbox(
            "Income stability",
            stability_values,
            index=stability_values.index(str(defaults["income_stability"])),
            key=f"stability_{suffix}",
        )
        objective_values = ["preservation", "income", "growth", "aggressive_growth"]
        objective = st.selectbox(
            "Objective",
            objective_values,
            index=objective_values.index(str(defaults["objective"])),
            format_func=lambda value: value.replace("_", " ").title(),
            key=f"objective_{suffix}",
        )
        excluded_assets = st.multiselect(
            "Exclude ETFs",
            list(ALL_ASSETS),
            default=list(defaults.get("excluded_assets", ())),
            key=f"exclusions_{suffix}",
        )
        position_cap = st.slider(
            "Maximum non-cash position",
            20,
            60,
            int(round(float(defaults["max_single_asset_weight"]) * 100)),
            format="%d%%",
            key=f"cap_{suffix}",
        )
        submitted = st.form_submit_button("Build recommendation", type="primary", width="stretch")

    if not submitted:
        return None
    return ClientProfile(
        name=name,
        age=int(age),
        investable_assets=float(investable_assets),
        annual_income=float(annual_income),
        monthly_contribution=float(monthly_contribution),
        goal_amount=float(goal_amount),
        horizon_years=float(horizon_years),
        risk_tolerance=int(risk_tolerance),
        max_drawdown_tolerance=float(max_drawdown) / 100.0,
        monthly_expenses=float(monthly_expenses),
        liquidity_needed_12m=float(liquidity_needed),
        emergency_cash_held=float(emergency_cash),
        income_stability=income_stability,
        objective=objective,
        excluded_assets=tuple(excluded_assets),
        max_single_asset_weight=float(position_cap) / 100.0,
    )


st.set_page_config(page_title="Wealth Management ML Project Lean", layout="wide")
st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 2rem;}
      [data-testid="stMetricValue"] {font-size: 1.55rem;}
      .small-note {color: #64748b; font-size: .9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Wealth Management ML Project")

if not CACHE.exists() or not PREDICTIONS.exists():
    st.info("Generate the local data, forecasts, and evaluation first: `python backtest.py` (or `python backtest.py --quick`).")
    st.stop()

try:
    prices, predictions = load_research_data()
except Exception as exc:  # A clear UI message is more useful than a Streamlit traceback.
    st.error(f"The saved research run could not be loaded: {exc}")
    st.stop()

with st.sidebar:
    st.header("Investor profile")
    preset_names = list(PROFILE_PRESETS)
    default_index = preset_names.index("Balanced") if "Balanced" in preset_names else 0
    preset_name = st.selectbox(
        "Start with a synthetic scenario",
        preset_names,
        index=default_index,
        format_func=str.title,
    )
    defaults = dict(PROFILE_PRESETS[preset_name])
    try:
        profile = profile_form(defaults, preset_name)
    except (TypeError, ValueError, InfeasibleProfileError) as exc:
        st.error(str(exc))
        profile = None

if profile is None:
    st.info("Choose a scenario or edit the inputs, then select **Build recommendation** in the sidebar.")
    image_path = RESULTS / "equity_curve.png"
    if image_path.exists():
        st.image(str(image_path), caption="Saved balanced-profile walk-forward comparison", width="stretch")
    st.stop()

try:
    policy = build_policy(profile)
    monthly_returns = prices.pct_change(fill_method=None)
    decision_date = pd.Timestamp(predictions.index.max())
    forecasts = predictions.loc[decision_date]
    weights, explanation = construct_portfolio(
        forecasts,
        monthly_returns.loc[:decision_date].tail(36),
        policy,
    )
    history = run_backtest(monthly_returns, predictions, profile)
    reported_metrics = evaluation_metrics(history)
except (TypeError, ValueError, InfeasibleProfileError) as exc:
    st.error(f"This profile cannot produce a compliant recommendation: {exc}")
    st.stop()

if not policy.actionable:
    st.warning(policy.abstention_reason or "The policy abstained from forecast-driven tilts.")
for warning in policy.warnings:
    st.warning(warning)

recommendation_tab, history_tab, method_tab = st.tabs(
    ["Recommendation", "Walk-forward history", "How it works"]
)

with recommendation_tab:
    st.caption(f"Uses information available through {decision_date:%B %Y}.")
    cards = st.columns(5)
    cards[0].metric("Risk label", policy.risk_label.replace("_", " ").title())
    cards[1].metric("Cash floor", f"{policy.cash_floor:.0%}")
    cards[2].metric("Growth target", f"{policy.growth_target:.0%}")
    cards[3].metric("Risk budget", f"{policy.target_volatility:.0%} vol")
    goal_value = "Unsolved" if policy.goal_required_return is None else f"{policy.goal_required_return:.1%}"
    cards[4].metric("Goal-implied return", goal_value)

    allocation_side, explanation_side = st.columns([1.05, 0.95])
    with allocation_side:
        st.markdown("#### Recommended allocation")
        nonzero = weights[weights > 0.0005]
        st.bar_chart(nonzero, color="#2563EB", height=330)
        allocation = pd.DataFrame(
            {
                "ETF": nonzero.index,
                "Role": [ASSET_NAMES[asset] for asset in nonzero.index],
                "Weight": [f"{value:.1%}" for value in nonzero.values],
            }
        )
        st.dataframe(allocation, hide_index=True, width="stretch")
    with explanation_side:
        st.markdown("#### Why")
        for rationale in policy.rationales:
            st.markdown(f"- {rationale}")
        if explanation.get("forecast_used"):
            blend = explanation.get("signal_blend", {})
            st.markdown(
                "- The tactical ranking blended "
                f"{float(blend.get('model_weight', .25)):.0%} neural signal with "
                f"{float(blend.get('momentum_weight', .75)):.0%} transparent momentum."
            )
            overweights = explanation.get("top_overweights", {})
            if overweights:
                changes = ", ".join(f"{asset} +{float(change):.1%}" for asset, change in overweights.items())
                st.markdown(f"- Largest tilts above strategic weights: {changes}.")
        else:
            st.markdown(f"- Forecast layer not used: {explanation.get('fallback_reason', 'policy fallback')}.")
        if explanation.get("volatility_scaled"):
            st.markdown("- Estimated volatility exceeded the risk budget, so exposure was shifted toward BIL.")

    strategic = pd.Series(policy.strategic_weights).reindex(ALL_ASSETS).fillna(0.0)
    comparison = pd.DataFrame({"Strategic": strategic, "Recommended": weights})
    comparison["Change"] = comparison["Recommended"] - comparison["Strategic"]
    st.markdown("#### Strategic policy versus recommendation")
    st.dataframe(
        comparison.style.format("{:.1%}"),
        width="stretch",
    )

with history_tab:
    st.markdown("#### Reported evaluation")
    st.caption(
        "January 2021 onward, 10 bps per dollar bought or sold. The static profile is the risk-matched "
        "primary comparison; SPY is market context."
    )
    st.dataframe(format_metrics(reported_metrics), width="stretch")

    reported_returns = history.returns.loc[history.returns.index >= REPORTED_START]
    wealth = (1.0 + reported_returns).cumprod()
    wealth = wealth.rename(columns={
        "strategy": "Wealth Management ML Project",
        "static_profile": "Static profile",
        "60_40": "60/40",
        "SPY": "SPY",
        "BIL": "BIL / cash",
        "equal_risky": "Equal risky",
    })
    st.line_chart(wealth, height=430)

    strategy = reported_metrics.loc["strategy"]
    static = reported_metrics.loc["static_profile"]
    if strategy["cagr"] > static["cagr"]:
        st.success(
            "For this synthetic profile and period, this project slightly exceeded its profile-matched static "
            "allocation. This is historical evidence, not a promise."
        )
    else:
        st.warning("For this profile and period, the active layer did not beat its static policy benchmark.")
    if "SPY" in reported_metrics and strategy["cagr"] <= reported_metrics.loc["SPY", "cagr"]:
        st.info("It did not beat SPY on raw return. That is expected to be a difficult and often unsuitable comparison for a diversified client portfolio.")

with method_tab:
    st.markdown("#### Four understandable steps")
    st.markdown(
        """
1. **Data:** Convert adjusted ETF prices to completed month-end observations and build only lagged features.
2. **Model:** A small PyTorch network predicts each risky ETF's next-month return above BIL.
3. **Suitability:** A separate transparent policy sets liquidity, growth, risk, exclusion, concentration, long-only, and no-leverage rules.
4. **Validation:** Each month-end forecast is applied to the first strictly later return; holdings drift and trading costs are deducted.
        """
    )
    st.markdown("#### Why keep the PyTorch model if it failed its gate?")
    st.write(
        "The project demonstrates a complete, testable ML workflow and an important model-risk decision: complexity "
        "should not be deployed merely because it exists. In this sample, momentum-only beat the ensemble and the "
        "neural-only forecast did not beat the static baseline, so the neural component should remain in shadow research."
    )
    st.markdown("#### Supported scope")
    st.write(
        "This supports synthetic profiles within the fields shown in the sidebar. It does not model taxes, account "
        "location, liabilities, current holdings, restricted securities, complex households, or guaranteed loss limits."
    )
