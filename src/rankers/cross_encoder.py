from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def stratified_split(
    df: pd.DataFrame,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows 70/15/15 into train/val/test, stratified on a 5-bin score.

    Mirrors notebook 01 §9.1. Adds a ``split`` column to ``df`` in place
    (``"train"``/``"val"``/``"test"``) so downstream code can mask the test
    rows, and returns the three subframes.

    Args:
        df: DataFrame with a ``matched_score`` column.
        seed: Random seed for reproducible splits.

    Returns:
        (train_df, val_df, test_df).
    """
    score_bin = pd.cut(df["matched_score"], bins=5, labels=False)
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed, stratify=score_bin)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=score_bin.loc[temp_df.index],
    )
    df["split"] = "train"
    df.loc[val_df.index, "split"] = "val"
    df.loc[test_df.index, "split"] = "test"
    return train_df, val_df, test_df


def _model_exists(output_dir: Path) -> bool:
    """True if ``output_dir`` already holds a saved transformer model."""
    if not output_dir.is_dir():
        return False
    return any((output_dir / name).exists() for name in ("model.safetensors", "pytorch_model.bin"))


def train_cross_encoder(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: Union[str, Path] = config.CE_OUTPUT_DIR,
    *,
    force: bool = False,
) -> Path:
    """Fine-tune a cross-encoder on ``matched_score`` with MSE loss.

    Mirrors notebook 01 §9.2. If a saved model already exists at
    ``output_dir`` and ``force`` is False, training is skipped and the
    existing path is returned (decision: train, reuse if cached).

    Args:
        train_df: Training rows (needs ``resume_text``, ``jd_text``,
            ``matched_score``).
        val_df: Validation rows (same columns).
        output_dir: Where to save / look for the model.
        force: Retrain even if a cached model exists.

    Returns:
        Path to the saved (or reused) model directory.
    """
    output_dir = Path(output_dir)
    if _model_exists(output_dir) and not force:
        print(f"  cache hit: reusing cross-encoder at {output_dir}")
        return output_dir

    import torch
    from datasets import Dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import (
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
        losses,
    )
    from sentence_transformers.cross_encoder.evaluation import (
        CrossEncoderCorrelationEvaluator,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def to_hf(d: pd.DataFrame) -> "Dataset":
        return Dataset.from_dict(
            {
                "resume": d["resume_text"].tolist(),
                "job": d["jd_text"].tolist(),
                "label": d["matched_score"].astype(float).tolist(),
            }
        )

    train_ds, val_ds = to_hf(train_df), to_hf(val_df)

    model = CrossEncoder(
        config.CE_BASE_MODEL,
        num_labels=1,
        max_length=config.CE_MAX_LEN,
        device=device,
    )
    loss = losses.MSELoss(model)

    val_evaluator = CrossEncoderCorrelationEvaluator(
        sentence_pairs=list(zip(val_ds["resume"], val_ds["job"])),
        scores=val_ds["label"],
        name="resume-jd-val",
    )

    args = CrossEncoderTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.CE_EPOCHS,
        per_device_train_batch_size=config.CE_BATCH_SIZE,
        per_device_eval_batch_size=config.CE_BATCH_SIZE,
        learning_rate=config.CE_LR,
        warmup_ratio=0.1,
        fp16=(device == "cuda"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_resume-jd-val_pearson",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=config.SEED,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        loss=loss,
        evaluator=val_evaluator,
    )
    trainer.train()
    model.save_pretrained(str(output_dir))
    print(f"  saved cross-encoder -> {output_dir}")
    return output_dir


def score_cross_encoder(
    model_dir: Union[str, Path],
    resume_texts: list[str],
    jd_texts: list[str],
) -> np.ndarray:
    """Score resume/JD pairs with a (fine-tuned) cross-encoder.

    Mirrors notebook 01 §9.3. The pair model attends jointly across both
    texts; scores are clipped to ``[0, 1]`` to match the label range.

    Args:
        model_dir: Directory of the saved cross-encoder.
        resume_texts: Cleaned resume texts.
        jd_texts: Cleaned JD texts (same length as ``resume_texts``).

    Returns:
        1-D float array of clipped scores, one per pair.
    """
    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(str(model_dir), max_length=config.CE_MAX_LEN, device=device)
    scores = model.predict(
        list(zip(resume_texts, jd_texts)),
        batch_size=32,
        show_progress_bar=True,
    )
    return np.clip(scores, 0.0, 1.0)
