from __future__ import annotations

from collections import Counter
from typing import Iterable, Literal

import numpy as np
import pandas as pd


def _get_sequence_columns(df: pd.DataFrame) -> list[str]:
    seq_cols = [col for col in df.columns if str(col).startswith("feature_")]
    seq_cols.sort(key=lambda name: int(str(name).split("_")[1]))
    if not seq_cols:
        raise ValueError("Khong tim thay cot sequence dang feature_*.")
    return seq_cols


def _clean_sequence(values: Iterable[float]) -> list[int]:
    seq: list[int] = []
    for x in values:
        if pd.isna(x):
            continue
        try:
            v = int(float(x))
        except Exception:
            continue
        if v != 0:
            seq.append(v)
    return seq


def _calculate_entropy(sequence: list[int]) -> float:
    if not sequence:
        return 0.0
    counts = Counter(sequence)
    probs = np.array(list(counts.values()), dtype=float) / len(sequence)
    return float(-np.sum(probs * np.log2(probs + 1e-9)))


def _count_rollbacks(sequence: list[int]) -> tuple[int, int, list[str], list[str]]:
    n = len(sequence)
    rollback_3 = 0
    rollback_4 = 0
    rb3_actions: list[str] = []
    rb4_actions: list[str] = []

    if n < 3:
        return 0, 0, [], []

    # A-B-A
    for i in range(n - 2):
        if sequence[i] == sequence[i + 2] and sequence[i] != sequence[i + 1]:
            rollback_3 += 1
            rb3_actions.append(str(sequence[i]))

    # A-B-C-A
    for i in range(n - 3):
        if (
            sequence[i] == sequence[i + 3]
            and sequence[i] != sequence[i + 1]
            and sequence[i] != sequence[i + 2]
        ):
            rollback_4 += 1
            rb4_actions.append(str(sequence[i]))

    return rollback_3, rollback_4, rb3_actions, rb4_actions


def _most_common_anchor(actions: list[str]) -> str:
    if not actions:
        return "-"
    return Counter(actions).most_common(1)[0][0]


def _row_to_action_sequence(sequence: list[int]) -> str:
    return "-".join(str(x) for x in sequence)


def _extract_row_features(row: pd.Series, seq_cols: list[str]) -> pd.Series:
    seq = _clean_sequence(row[seq_cols].values)
    length = len(seq)
    unique_count = len(set(seq))
    entropy = _calculate_entropy(seq)

    rb3, rb4, rb3_actions, rb4_actions = _count_rollbacks(seq)

    all_rb_actions = rb3_actions + rb4_actions
    first_action_rb = sorted(set(all_rb_actions))
    anchor_action = _most_common_anchor(all_rb_actions)

    return pd.Series(
        {
            "length": length,
            "unique_count": unique_count,
            "entropy": entropy,
            "rb_3_steps": rb3,
            "rb_4_steps": rb4,
            "rb3_actions": rb3_actions,
            "rb4_actions": rb4_actions,
            "first_action_rb": first_action_rb,
            "anchor_action": anchor_action,
            "action_sequence": _row_to_action_sequence(seq),
        }
    )


def featuring_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    seq_cols = _get_sequence_columns(df)

    feature_df = df.apply(lambda row: _extract_row_features(row, seq_cols), axis=1)
    return pd.concat([df, feature_df], axis=1)


def generate_edge_case_report(
    df: pd.DataFrame,
    mode: Literal["or", "and"] = "or",
    min_rb3: int = 2,
    min_rb4: int = 1,
) -> pd.DataFrame:
    if mode not in {"or", "and"}:
        raise ValueError("mode phai la 'or' hoac 'and'.")

    required_cols = {
        "id",
        "action_sequence",
        "anchor_action",
        "first_action_rb",
        "rb3_actions",
        "rb4_actions",
        "rb_3_steps",
        "rb_4_steps",
        "length",
        "unique_count",
        "entropy",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Thieu cot dau vao cho report: {sorted(missing)}")

    cond_rb3 = df["rb_3_steps"] >= min_rb3
    cond_rb4 = df["rb_4_steps"] >= min_rb4
    mask = (cond_rb3 | cond_rb4) if mode == "or" else (cond_rb3 & cond_rb4)
    edge_cases = df[mask].copy()

    def create_fact(row: pd.Series) -> str:
        rb3 = int(row["rb_3_steps"])
        rb4 = int(row["rb_4_steps"])
        length = int(row["length"])
        unique_count = int(row["unique_count"])
        entropy = round(float(row["entropy"]), 2)
        anchor = row["anchor_action"]
        return (
            f"Anchor={anchor}. "
            f"Phat hien {rb3} lan lap 3 buoc (A-B-A), "
            f"{rb4} lan lap 4 buoc (A-B-C-A). "
            f"Do dai chuoi: {length} thao tac. "
            f"So action duy nhat: {unique_count}. "
            f"Chi so hon loan (Entropy): {entropy}."
        )

    def create_edge_rule(row: pd.Series) -> str:
        hit_rb3 = int(row["rb_3_steps"]) >= min_rb3
        hit_rb4 = int(row["rb_4_steps"]) >= min_rb4
        if hit_rb3 and hit_rb4:
            return f"rb3>={min_rb3}+rb4>={min_rb4}"
        if hit_rb3:
            return f"rb3>={min_rb3}"
        return f"rb4>={min_rb4}"

    report_cols = [
        "id",
        "action_sequence",
        "anchor_action",
        "first_action_rb",
        "rb3_actions",
        "rb4_actions",
        "rb_3_steps",
        "rb_4_steps",
        "length",
        "unique_count",
        "entropy",
    ]
    report_df = edge_cases[report_cols].copy()
    report_df["edge_rule"] = edge_cases.apply(create_edge_rule, axis=1)
    report_df["fact"] = edge_cases.apply(create_fact, axis=1)

    return report_df.reset_index(drop=True)