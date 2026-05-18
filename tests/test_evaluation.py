import numpy as np
import pandas as pd
import pytest

from src.evaluation import ranking_metrics, regression_metrics


def test_regression_metrics_perfect():
    y = np.array([0.1, 0.5, 0.9, 0.3])
    result = regression_metrics(y, y)
    assert result["MAE"] == pytest.approx(0.0, abs=1e-9)
    assert result["RMSE"] == pytest.approx(0.0, abs=1e-9)
    assert result["Pearson r"] == pytest.approx(1.0, abs=1e-6)
    assert result["Spearman rho"] == pytest.approx(1.0, abs=1e-6)


def test_regression_metrics_returns_expected_keys():
    result = regression_metrics(np.array([0.5, 0.6]), np.array([0.4, 0.7]))
    assert set(result.keys()) == {"MAE", "RMSE", "Pearson r", "Spearman rho"}


def test_regression_metrics_zero_variance_pred():
    y_true = np.array([0.1, 0.5, 0.9])
    y_pred = np.array([0.5, 0.5, 0.5])
    result = regression_metrics(y_true, y_pred)
    assert np.isnan(result["Pearson r"])
    assert np.isnan(result["Spearman rho"])


def _make_ranking_df():
    return pd.DataFrame({
        "jd_group": [0, 0, 0, 1, 1, 1],
        "matched_score": [0.9, 0.5, 0.2, 0.8, 0.6, 0.3],
        "pred": [0.85, 0.55, 0.15, 0.75, 0.65, 0.25],
    })


def test_ranking_metrics_returns_expected_keys():
    df = _make_ranking_df()
    result = ranking_metrics(df, pred_col="pred")
    assert "NDCG@10" in result
    assert "Precision@10" in result
    assert "MRR" in result
    assert "n_groups_evaluated" in result
    assert "relevance_threshold" in result


def test_ranking_metrics_n_groups():
    df = _make_ranking_df()
    result = ranking_metrics(df, pred_col="pred")
    assert result["n_groups_evaluated"] == 2


def test_ranking_metrics_perfect_ranking():
    df = _make_ranking_df()
    result = ranking_metrics(df, pred_col="pred")
    assert result["NDCG@10"] == pytest.approx(1.0, abs=1e-6)


def test_ranking_metrics_no_eligible_groups():
    df = pd.DataFrame({
        "jd_group": [0, 1, 2],
        "matched_score": [0.9, 0.5, 0.2],
        "pred": [0.8, 0.6, 0.3],
    })
    result = ranking_metrics(df, pred_col="pred")
    assert result["n_groups_evaluated"] == 0
    assert np.isnan(result["NDCG@10"])
