from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score

from src.config import TOP_K


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, Pearson r, and Spearman ρ between predictions and labels.

    Args:
        y_true: Ground-truth scores.
        y_pred: Predicted scores.

    Returns:
        Dict with keys MAE, RMSE, Pearson r, Spearman rho.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson, spearman = float("nan"), float("nan")
    else:
        pearson = float(pearsonr(y_true, y_pred).statistic)
        spearman = float(spearmanr(y_true, y_pred).statistic)
    return {"MAE": mae, "RMSE": rmse, "Pearson r": pearson, "Spearman rho": spearman}


def ranking_metrics(
    df: pd.DataFrame,
    pred_col: str,
    label_col: str = "matched_score",
    jd_group_col: str = "jd_group",
    k: int = TOP_K,
    relevance_threshold: Optional[float] = None,
) -> dict[str, float]:
    """Compute average NDCG@K, Precision@K, MRR across JD groups with ≥2 resumes.

    Args:
        df: DataFrame with at least pred_col, label_col, jd_group_col.
        pred_col: Column holding the predicted similarity score.
        label_col: Column holding ground-truth match scores.
        jd_group_col: Column identifying which rows share a JD.
        k: Cutoff for NDCG and Precision.
        relevance_threshold: Scores >= this count as relevant for Precision/MRR.
            Defaults to the dataset median of `label_col`.

    Returns:
        Dict with averaged metrics plus n_groups_evaluated and relevance_threshold.
    """
    if relevance_threshold is None:
        relevance_threshold = df[label_col].median()

    ndcgs, precisions, rrs = [], [], []
    eligible_groups = 0

    for _, group in df.groupby(jd_group_col):
        if len(group) < 2:
            continue
        eligible_groups += 1

        y_true_graded = group[label_col].to_numpy()[None, :]
        y_pred = group[pred_col].to_numpy()[None, :]
        ndcgs.append(ndcg_score(y_true_graded, y_pred, k=k))

        binary_relevance = (group[label_col] >= relevance_threshold).to_numpy()
        order = np.argsort(-group[pred_col].to_numpy())
        ranked_relevance = binary_relevance[order]

        top_k = ranked_relevance[:k]
        precisions.append(top_k.sum() / min(k, len(top_k)))

        first_hit = np.argmax(ranked_relevance) if ranked_relevance.any() else None
        rrs.append(1.0 / (first_hit + 1) if first_hit is not None else 0.0)

    return {
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else float("nan"),
        f"Precision@{k}": float(np.mean(precisions)) if precisions else float("nan"),
        "MRR": float(np.mean(rrs)) if rrs else float("nan"),
        "n_groups_evaluated": eligible_groups,
        "relevance_threshold": float(relevance_threshold),
    }
