import numpy as np
import pandas as pd

from src import config
from src.rankers.cross_encoder import (
    _model_exists,
    stratified_split,
    train_cross_encoder,
)


def _make_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(config.SEED)
    return pd.DataFrame(
        {
            "resume_text": [f"resume {i}" for i in range(n)],
            "jd_text": [f"job {i}" for i in range(n)],
            "matched_score": rng.uniform(0.0, 1.0, size=n),
        }
    )


def test_stratified_split_partitions_all_rows():
    df = _make_df()
    train_df, val_df, test_df = stratified_split(df)
    total = len(train_df) + len(val_df) + len(test_df)
    assert total == len(df)
    idx = set(train_df.index) | set(val_df.index) | set(test_df.index)
    assert idx == set(df.index)  # disjoint and exhaustive


def test_stratified_split_assigns_split_column():
    df = _make_df()
    stratified_split(df)
    assert set(df["split"].unique()) == {"train", "val", "test"}


def test_stratified_split_is_reproducible():
    a = _make_df()
    b = _make_df()
    stratified_split(a, seed=config.SEED)
    stratified_split(b, seed=config.SEED)
    assert a["split"].tolist() == b["split"].tolist()


def test_model_exists_detects_saved_model(tmp_path):
    assert _model_exists(tmp_path) is False
    (tmp_path / "model.safetensors").write_bytes(b"\x00")
    assert _model_exists(tmp_path) is True


def test_train_cross_encoder_reuses_cached_model(tmp_path):
    # Cache-reuse branch must short-circuit before any heavy import / training.
    (tmp_path / "model.safetensors").write_bytes(b"\x00")
    out = train_cross_encoder(pd.DataFrame(), pd.DataFrame(), output_dir=tmp_path)
    assert out == tmp_path
