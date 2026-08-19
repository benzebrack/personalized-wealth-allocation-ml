"""
The model predicts next-month excess returns for the seven risky ETFs. Model
selection is chronological such that the most recent labelled months are validation,
never shuffled into selection training. After early stopping chooses an epoch
count, a fresh network is refit on every outcome available at that decision
date. The walk-forward function repeats this process annually and scores each
month, enforcing target_date <= decision_date before every fit.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence, TypeAlias, overload

import numpy as np
import pandas as pd
import torch
from torch import nn

from data import ALL_ASSETS, CASH_ASSET, FEATURE_COLUMNS, RISKY_ASSETS, TARGET_COLUMN


DEFAULT_RANDOM_SEED = 42
PathLike: TypeAlias = str | os.PathLike[str]


def _set_seed(seed: int) -> None:
    # Seed every random source used by this CPU training routine.

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass(frozen=True)
class Standardizer:
    # Column standardisation learned from training data.

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] == 0:
            raise ValueError("Standardizer.fit expects a non-empty 2D array.")
        if not np.isfinite(array).all():
            raise ValueError("Cannot fit a Standardizer to non-finite values.")
        mean = array.mean(axis=0)
        scale = array.std(axis=0, ddof=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"Expected a 2D array with {self.mean.shape[0]} feature columns."
            )
        if not np.isfinite(array).all():
            raise ValueError("Model features must all be finite.")
        return ((array - self.mean) / self.scale).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, values: Mapping[str, Sequence[float]]) -> "Standardizer":
        return cls(
            mean=np.asarray(values["mean"], dtype=np.float64),
            scale=np.asarray(values["scale"], dtype=np.float64),
        )


class ReturnMLP(nn.Module):
    # Small network: 26 inputs -> 32 -> 16 -> one excess-return score.

    def __init__(self, input_size: int = len(FEATURE_COLUMNS)) -> None:
        super().__init__()
        if input_size < 1:
            raise ValueError("input_size must be positive.")
        self.input_size = int(input_size)
        self.network = nn.Sequential(
            nn.Linear(self.input_size, 32),
            nn.ReLU(),
            nn.Dropout(p=0.10),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(p=0.10),
            nn.Linear(16, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


@dataclass
class ModelBundle:
    # A fitted network plus its preprocessing and reproducibility metadata.

    model: ReturnMLP
    standardizer: Standardizer
    feature_columns: tuple[str, ...]
    metadata: dict[str, Any]

    def _feature_array(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            missing = [column for column in self.feature_columns if column not in data]
            if missing:
                raise ValueError("Missing model features: " + ", ".join(missing))
            values = data.loc[:, list(self.feature_columns)].to_numpy(dtype=np.float64)
        else:
            values = np.asarray(data, dtype=np.float64)
        return self.standardizer.transform(values)

    @overload
    def predict(self, data: pd.DataFrame) -> pd.Series: ...

    @overload
    def predict(self, data: np.ndarray) -> np.ndarray: ...

    def predict(self, data: pd.DataFrame | np.ndarray) -> pd.Series | np.ndarray:
        # Predict excess returns without changing network state.

        features = torch.from_numpy(self._feature_array(data))
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(features).squeeze(-1).cpu().numpy()
        if isinstance(data, pd.DataFrame):
            return pd.Series(
                prediction, index=data.index, name="predicted_excess_return"
            )
        return prediction

    def save(self, path: PathLike) -> Path:
        # Persist a portable CPU checkpoint and return its resolved path.

        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "feature_columns": list(self.feature_columns),
            "standardizer": self.standardizer.to_dict(),
            "metadata": self.metadata,
            "state_dict": {
                key: value.detach().cpu()
                for key, value in self.model.state_dict().items()
            },
        }
        torch.save(payload, destination)
        return destination.resolve()

    @classmethod
    def load(cls, path: PathLike) -> "ModelBundle":
        # Load a checkpoint onto the CPU.

        source = Path(path).expanduser()
        try:
            payload = torch.load(source, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch before the weights_only argument
            payload = torch.load(source, map_location="cpu")

        if payload.get("format_version") != 1:
            raise ValueError("Unsupported project model checkpoint format.")
        columns = tuple(str(column) for column in payload["feature_columns"])
        model = ReturnMLP(len(columns))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return cls(
            model=model,
            standardizer=Standardizer.from_dict(payload["standardizer"]),
            feature_columns=columns,
            metadata=dict(payload["metadata"]),
        )


def _loss_value(
    model: ReturnMLP,
    features: torch.Tensor,
    target: torch.Tensor,
    criterion: nn.Module,
) -> float:
    model.eval()
    with torch.no_grad():
        return float(criterion(model(features), target).item())


def _training_epoch(
    model: ReturnMLP,
    features: torch.Tensor,
    target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    *,
    batch_size: int,
    generator: torch.Generator,
) -> float:
    model.train()
    order = torch.randperm(features.shape[0], generator=generator)
    accumulated = 0.0
    for start in range(0, features.shape[0], batch_size):
        positions = order[start : start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(features[positions]), target[positions])
        loss.backward()
        optimizer.step()
        accumulated += float(loss.item()) * len(positions)
    return accumulated / features.shape[0]


def _labelled_training_frame(
    panel: pd.DataFrame,
    feature_columns: tuple[str, ...],
    target_column: str,
) -> tuple[pd.DataFrame, int]:
    required = {"feature_date", target_column, *feature_columns}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError("Training panel is missing columns: " + ", ".join(missing))

    frame = panel.copy()
    frame["feature_date"] = pd.to_datetime(frame["feature_date"], errors="coerce")
    if frame["feature_date"].isna().any():
        raise ValueError("feature_date contains invalid dates.")

    dropped = int(frame[target_column].isna().sum())
    frame = frame.dropna(subset=[target_column]).copy()
    if frame.empty:
        raise ValueError("Training panel contains no labelled observations.")
    sort_columns = ["feature_date"] + (["asset"] if "asset" in frame else [])
    frame = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    feature_values = frame.loc[:, list(feature_columns)].to_numpy(dtype=np.float64)
    target_values = frame[target_column].to_numpy(dtype=np.float64)
    if not np.isfinite(feature_values).all():
        raise ValueError("Training features contain missing or non-finite values.")
    if not np.isfinite(target_values).all():
        raise ValueError("Training targets contain non-finite values.")
    return frame, dropped


def train_model(
    panel: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
    validation_months: int = 12,
    max_epochs: int = 300,
    patience: int = 30,
    min_delta: float = 1e-6,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    huber_delta: float = 0.05,
    batch_size: int = 256,
    seed: int = DEFAULT_RANDOM_SEED,
) -> ModelBundle:
    # Choose an epoch chronologically, then refit on all supplied outcomes.

    columns = tuple(feature_columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("feature_columns must be non-empty and unique.")
    if validation_months < 1:
        raise ValueError("validation_months must be at least one.")
    if max_epochs < 1 or patience < 1 or batch_size < 1:
        raise ValueError("max_epochs, patience, and batch_size must be positive.")
    if learning_rate <= 0.0 or weight_decay < 0.0 or huber_delta <= 0.0:
        raise ValueError("Invalid optimizer or Huber-loss hyperparameters.")

    labelled, dropped_unlabelled = _labelled_training_frame(
        panel, columns, target_column
    )
    unique_dates = pd.DatetimeIndex(labelled["feature_date"].drop_duplicates())
    if len(unique_dates) <= validation_months:
        raise ValueError(
            f"Need more than {validation_months} labelled feature months for "
            "chronological validation."
        )

    validation_start = unique_dates[-validation_months]
    selection_train = labelled.loc[labelled["feature_date"] < validation_start]
    validation = labelled.loc[labelled["feature_date"] >= validation_start]

    x_train_raw = selection_train.loc[:, list(columns)].to_numpy(dtype=np.float64)
    y_train = selection_train[target_column].to_numpy(dtype=np.float32).reshape(-1, 1)
    x_validation_raw = validation.loc[:, list(columns)].to_numpy(dtype=np.float64)
    y_validation = validation[target_column].to_numpy(dtype=np.float32).reshape(-1, 1)

    selection_scaler = Standardizer.fit(x_train_raw)
    x_train = torch.from_numpy(selection_scaler.transform(x_train_raw))
    x_validation = torch.from_numpy(selection_scaler.transform(x_validation_raw))
    y_train_tensor = torch.from_numpy(y_train)
    y_validation_tensor = torch.from_numpy(y_validation)

    _set_seed(seed)
    selection_model = ReturnMLP(len(columns))
    criterion = nn.HuberLoss(delta=huber_delta)
    optimizer = torch.optim.AdamW(
        selection_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed + 1)

    best_loss = float("inf")
    best_epoch = 1
    stale_epochs = 0
    training_history: list[float] = []
    validation_history: list[float] = []

    for epoch in range(1, max_epochs + 1):
        train_loss = _training_epoch(
            selection_model,
            x_train,
            y_train_tensor,
            optimizer,
            criterion,
            batch_size=batch_size,
            generator=generator,
        )
        validation_loss = _loss_value(
            selection_model, x_validation, y_validation_tensor, criterion
        )
        training_history.append(train_loss)
        validation_history.append(validation_loss)

        if best_loss - validation_loss > min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    x_all_raw = labelled.loc[:, list(columns)].to_numpy(dtype=np.float64)
    y_all = labelled[target_column].to_numpy(dtype=np.float32).reshape(-1, 1)
    final_scaler = Standardizer.fit(x_all_raw)
    x_all = torch.from_numpy(final_scaler.transform(x_all_raw))
    y_all_tensor = torch.from_numpy(y_all)

    _set_seed(seed)
    final_model = ReturnMLP(len(columns))
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    final_generator = torch.Generator().manual_seed(seed + 1)
    refit_loss = float("nan")
    for _ in range(best_epoch):
        refit_loss = _training_epoch(
            final_model,
            x_all,
            y_all_tensor,
            final_optimizer,
            criterion,
            batch_size=batch_size,
            generator=final_generator,
        )
    final_model.eval()

    metadata: dict[str, Any] = {
        "architecture": [len(columns), 32, 16, 1],
        "dropout": 0.10,
        "feature_columns": list(columns),
        "target_column": target_column,
        "loss": "HuberLoss",
        "huber_delta": float(huber_delta),
        "optimizer": "AdamW",
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "validation_months": int(validation_months),
        "validation_start": validation_start.strftime("%Y-%m-%d"),
        "validation_end": unique_dates[-1].strftime("%Y-%m-%d"),
        "selection_train_observations": int(len(selection_train)),
        "validation_observations": int(len(validation)),
        "refit_observations": int(len(labelled)),
        "dropped_unlabelled_observations": int(dropped_unlabelled),
        "trained_through": unique_dates[-1].strftime("%Y-%m-%d"),
        "selected_epoch": int(best_epoch),
        "epochs_run_during_selection": int(len(validation_history)),
        "best_validation_loss": float(best_loss),
        "refit_final_training_loss": float(refit_loss),
        "selection_training_loss_history": [float(x) for x in training_history],
        "validation_loss_history": [float(x) for x in validation_history],
        "selection_scaler_fit": "pre-validation observations only",
        "final_scaler_fit": "all labelled observations",
    }
    return ModelBundle(
        model=final_model,
        standardizer=final_scaler,
        feature_columns=columns,
        metadata=metadata,
    )


def _month_end(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).to_period("M").to_timestamp("M")


def walk_forward_predictions(
    panel: pd.DataFrame,
    *,
    backtest_start: str | pd.Timestamp = "2015-01-01",
    retrain_every_months: int = 12,
    minimum_training_months: int = 60,
    max_epochs: int = 220,
    patience: int = 25,
    seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, ModelBundle]:
    """
    Return leakage-safe monthly predictions with annual expanding-window fits.
    The signal dated month t is fit only with labels whose outcomes have
    already finished by t. A model is retrained every twelve months by
    default but scores all seven risky assets at every decision month.  BIL is
    appended with a neutral score of zero because it is a policy/cash sleeve,
    not a model target.
    """

    if retrain_every_months < 1:
        raise ValueError("retrain_every_months must be positive.")
    if minimum_training_months <= 12:
        raise ValueError("minimum_training_months must exceed the validation window.")

    required = {"feature_date", "target_date", "asset", TARGET_COLUMN, *FEATURE_COLUMNS}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError("Feature panel is missing columns: " + ", ".join(missing))

    frame = panel.copy()
    frame["feature_date"] = pd.to_datetime(frame["feature_date"], errors="coerce")
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce")
    if frame[["feature_date", "target_date"]].isna().any(axis=None):
        raise ValueError("Feature panel contains invalid feature or target dates.")

    start = _month_end(backtest_start)
    decision_dates = pd.DatetimeIndex(
        sorted(frame.loc[frame["feature_date"] >= start, "feature_date"].unique())
    )
    if decision_dates.empty:
        raise ValueError("No feature dates fall inside the requested backtest period.")

    bundle: ModelBundle | None = None
    prediction_rows: list[pd.Series] = []
    events: list[dict[str, Any]] = []
    months_since_fit = retrain_every_months

    for decision_date in decision_dates:
        trainable = frame.loc[
            (frame["target_date"] <= decision_date) & frame[TARGET_COLUMN].notna()
        ].copy()
        labelled_months = int(trainable["feature_date"].nunique())
        if labelled_months < minimum_training_months:
            continue

        if bundle is None or months_since_fit >= retrain_every_months:
            bundle = train_model(
                trainable,
                max_epochs=max_epochs,
                patience=patience,
                seed=seed,
            )
            months_since_fit = 0
            metadata = bundle.metadata
            events.append(
                {
                    "decision_date": decision_date,
                    "trained_through_feature_date": pd.Timestamp(
                        metadata["trained_through"]
                    ),
                    "latest_observed_target_date": trainable["target_date"].max(),
                    "labelled_months": labelled_months,
                    "observations": int(metadata["refit_observations"]),
                    "selected_epoch": int(metadata["selected_epoch"]),
                    "best_validation_loss": float(metadata["best_validation_loss"]),
                    "validation_start": pd.Timestamp(metadata["validation_start"]),
                    "validation_end": pd.Timestamp(metadata["validation_end"]),
                    "seed": int(metadata["seed"]),
                }
            )

        scoring = frame.loc[frame["feature_date"] == decision_date].copy()
        if set(scoring["asset"]) != set(RISKY_ASSETS):
            raise ValueError(f"Incomplete scoring universe at {decision_date.date()}.")
        assert bundle is not None  # narrowed by the fit condition above
        predicted = bundle.predict(scoring)
        row = pd.Series(
            predicted.to_numpy(dtype=float),
            index=scoring["asset"].to_numpy(),
            name=decision_date,
        ).reindex(RISKY_ASSETS)
        row.loc[CASH_ASSET] = 0.0
        prediction_rows.append(row.reindex(ALL_ASSETS))
        months_since_fit += 1

    if bundle is None or not prediction_rows:
        raise ValueError("Not enough history to fit the first walk-forward model.")

    predictions = pd.DataFrame(prediction_rows)
    predictions.index = pd.DatetimeIndex(predictions.index, name="decision_date")
    predictions = predictions.reindex(columns=ALL_ASSETS)
    retraining_log = pd.DataFrame(events)
    return predictions, retraining_log, bundle

train_return_model = train_model

__all__ = [
    "ModelBundle",
    "ReturnMLP",
    "Standardizer",
    "train_model",
    "train_return_model",
    "walk_forward_predictions",
]
