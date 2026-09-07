"""Investment risk assessment.

A direct adaptation of the `assess_risks` function from finalmodel.ipynb —
same thresholds, same decision logic (expected return vs. historical
volatility vs. P/E ratio) — generalized to take the live 7-day forecast and
any ticker's historical prices instead of the notebook's fixed HDB slice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_PE_RATIO = 20.0
DEFAULT_DEBT_EQUITY = 1.0

EXPECTED_RETURN_BUY_THRESHOLD = 0.05
EXPECTED_RETURN_AVOID_THRESHOLD = -0.05
VOLATILITY_AVOID_THRESHOLD = 0.20
PE_BUY_THRESHOLD = 15.0


@dataclass
class RiskAssessment:
    decision: str
    expected_return: float
    historical_volatility: float
    pe_ratio: float
    debt_equity: float
    rationale: str


def assess_risks(
    predicted_prices: np.ndarray,
    historical_prices: np.ndarray,
    financial_metrics: dict,
) -> str:
    """Unchanged decision logic from finalmodel.ipynb's assess_risks."""
    expected_return = (predicted_prices[-1] - predicted_prices[0]) / predicted_prices[0]

    historical_returns = [
        (historical_prices[i] - historical_prices[i - 1]) / historical_prices[i - 1]
        for i in range(1, len(historical_prices))
    ]
    historical_volatility = np.std(historical_returns)

    if expected_return > EXPECTED_RETURN_BUY_THRESHOLD:
        if financial_metrics["P/E ratio"] < PE_BUY_THRESHOLD:
            decision = "Invest"
        else:
            decision = "Hold"
    elif expected_return < EXPECTED_RETURN_AVOID_THRESHOLD or historical_volatility > VOLATILITY_AVOID_THRESHOLD:
        decision = "Avoid"
    else:
        decision = "Hold"

    return decision


def build_assessment(
    forecast_close: pd.Series,
    historical_close: pd.Series,
    pe_ratio: float | None,
    debt_equity: float | None,
    lookback_days: int = 90,
) -> RiskAssessment:
    predicted_prices = forecast_close.to_numpy()
    historical_prices = historical_close.tail(lookback_days).to_numpy()

    pe = pe_ratio if pe_ratio is not None else DEFAULT_PE_RATIO
    de = debt_equity if debt_equity is not None else DEFAULT_DEBT_EQUITY
    financial_metrics = {"P/E ratio": pe, "Debt/Equity ratio": de}

    decision = assess_risks(predicted_prices, historical_prices, financial_metrics)

    expected_return = (predicted_prices[-1] - predicted_prices[0]) / predicted_prices[0]
    historical_returns = np.diff(historical_prices) / historical_prices[:-1]
    historical_volatility = float(np.std(historical_returns))

    if decision == "Invest":
        rationale = (
            f"Expected 7-day return of {expected_return:+.1%} exceeds the "
            f"{EXPECTED_RETURN_BUY_THRESHOLD:.0%} threshold, and P/E ({pe:.1f}) "
            f"is below {PE_BUY_THRESHOLD:.0f}."
        )
    elif decision == "Avoid":
        if expected_return < EXPECTED_RETURN_AVOID_THRESHOLD:
            rationale = (
                f"Expected 7-day return of {expected_return:+.1%} is below the "
                f"{EXPECTED_RETURN_AVOID_THRESHOLD:.0%} threshold."
            )
        else:
            rationale = (
                f"Historical volatility ({historical_volatility:.1%}) exceeds the "
                f"{VOLATILITY_AVOID_THRESHOLD:.0%} threshold."
            )
    else:
        rationale = (
            f"Expected 7-day return of {expected_return:+.1%} and volatility of "
            f"{historical_volatility:.1%} fall between the Invest and Avoid thresholds"
            + (f", or P/E ({pe:.1f}) is above {PE_BUY_THRESHOLD:.0f}." if expected_return > EXPECTED_RETURN_BUY_THRESHOLD else ".")
        )

    return RiskAssessment(
        decision=decision,
        expected_return=expected_return,
        historical_volatility=historical_volatility,
        pe_ratio=pe,
        debt_equity=de,
        rationale=rationale,
    )
