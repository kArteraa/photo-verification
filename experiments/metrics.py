"""Metrics over the prediction table: per-requirement quality, error rates and timing."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from app.core.predicate import violation_score
from app.core.report import Status
from app.core.spec import Spec
from experiments.common import CLASS_TO_REQUIREMENT, CLEAN
from experiments.evaluate import measure_column, status_column


@dataclass(frozen=True)
class RequirementMetrics:
    """Detection quality of one requirement on one violation class against clean images."""

    requirement: str
    cls: str
    precision: float
    recall: float
    f1: float
    auc: float
    n_pos: int
    n_neg: int
    undefined_rate: float


@dataclass(frozen=True)
class ErrorRates:
    """Integral acceptance errors."""

    far: float
    frr: float
    per_class_far: dict[str, float]


def binary_prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Precision, recall and F1 of the positive class."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return float(precision), float(recall), float(f1)


def scores_for(frame: pd.DataFrame, spec: Spec, requirement_id: str) -> pd.Series:
    """Violation score of every row from the raw measurements, NaN when unavailable."""
    requirement = spec.by_id(requirement_id)
    predicate = requirement.predicate
    columns = {key: measure_column(requirement.measurer.name, key) for key in predicate.keys}
    values = frame[list(columns.values())]
    scores = pd.Series(np.nan, index=frame.index, dtype=float)
    complete = values.notna().all(axis=1)
    for index in frame.index[complete]:
        row = {key: float(frame.at[index, column]) for key, column in columns.items()}
        scores.at[index] = violation_score(predicate, row)
    return scores


def requirement_metrics(
    frame: pd.DataFrame, spec: Spec, requirement_id: str, cls: str
) -> RequirementMetrics:
    """Quality of one requirement on ``cls`` versus clean images."""
    subset = frame[frame["cls"].isin((cls, CLEAN))]
    y_true = (subset["cls"] == cls).to_numpy(dtype=int)
    statuses = subset[status_column(requirement_id)]
    y_pred = (statuses == Status.FAIL.value).to_numpy(dtype=int)
    precision, recall, f1 = binary_prf(y_true, y_pred)
    scores = scores_for(subset, spec, requirement_id)
    valid = scores.notna().to_numpy()
    auc = float("nan")
    if valid.any() and len(set(y_true[valid])) == 2:
        auc = float(roc_auc_score(y_true[valid], scores[valid]))
    return RequirementMetrics(
        requirement=requirement_id,
        cls=cls,
        precision=precision,
        recall=recall,
        f1=f1,
        auc=auc,
        n_pos=int(y_true.sum()),
        n_neg=int(len(y_true) - y_true.sum()),
        undefined_rate=float((statuses == Status.UNDEFINED.value).mean()),
    )


def all_requirement_metrics(frame: pd.DataFrame, spec: Spec) -> list[RequirementMetrics]:
    """Metrics for every violation class present in the table."""
    present = set(frame["cls"])
    return [
        requirement_metrics(frame, spec, requirement_id, cls)
        for cls, requirement_id in CLASS_TO_REQUIREMENT.items()
        if cls in present and requirement_id in spec.ids
    ]


def error_rates(frame: pd.DataFrame, accepted_column: str = "accepted") -> ErrorRates:
    """FAR over violating images and FRR over clean images."""
    accepted = frame[accepted_column].astype(bool)
    violating = frame["cls"] != CLEAN
    far = float(accepted[violating].mean()) if violating.any() else float("nan")
    frr = float((~accepted[~violating]).mean()) if (~violating).any() else float("nan")
    per_class = {
        cls: float(accepted[frame["cls"] == cls].mean())
        for cls in sorted(set(frame.loc[violating, "cls"]))
    }
    return ErrorRates(far, frr, per_class)


def timing(frame: pd.DataFrame) -> dict[str, float]:
    """Median and mean processing time per image in milliseconds."""
    return {
        "median_ms": float(frame["elapsed_ms"].median()),
        "mean_ms": float(frame["elapsed_ms"].mean()),
    }


def cross_trigger(frame: pd.DataFrame, requirement_ids: list[str]) -> pd.DataFrame:
    """Fail rate of every requirement on every class (rows: classes, columns: requirements)."""
    table = {
        requirement_id: frame.groupby("cls")[status_column(requirement_id)].apply(
            lambda s: float((s == Status.FAIL.value).mean())
        )
        for requirement_id in requirement_ids
    }
    return pd.DataFrame(table)


def metrics_table(items: list[RequirementMetrics]) -> pd.DataFrame:
    """Per-requirement metrics as a table."""
    return pd.DataFrame([asdict(item) for item in items])
