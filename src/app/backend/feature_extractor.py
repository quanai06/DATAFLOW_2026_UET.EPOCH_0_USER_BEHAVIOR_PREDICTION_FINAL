from __future__ import annotations

import math
from collections import Counter
from typing import Any

from src.app.backend.config import DEFAULT_MAX_SEQUENCE_LENGTH, DEFAULT_TOP_HUBS


def parse_sequence(sequence_input: Any) -> list[int]:
    if isinstance(sequence_input, str):
        raw_parts = sequence_input.replace("\n", ",").split(",")
        values = [part.strip() for part in raw_parts if part.strip()]
    elif isinstance(sequence_input, (list, tuple)):
        values = list(sequence_input)
    else:
        raise ValueError("Sequence phai la danh sach so nguyen hoac chuoi comma-separated.")

    parsed: list[int] = []
    for value in values:
        try:
            parsed_value = int(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Gia tri sequence khong hop le: {value!r}") from exc
        if parsed_value > 0:
            parsed.append(parsed_value)

    if not parsed:
        raise ValueError("Sequence khong duoc rong va khong duoc chi gom gia tri 0.")

    return parsed


def normalize_sequence(sequence: list[int], max_length: int = DEFAULT_MAX_SEQUENCE_LENGTH) -> tuple[list[int], bool]:
    clipped = False
    if len(sequence) > max_length:
        sequence = sequence[:max_length]
        clipped = True
    return sequence, clipped


def pad_sequence(sequence: list[int], max_length: int = DEFAULT_MAX_SEQUENCE_LENGTH) -> list[int]:
    padded = list(sequence[:max_length])
    if len(padded) < max_length:
        padded.extend([0] * (max_length - len(padded)))
    return padded


def _entropy(sequence: list[int]) -> float:
    if not sequence:
        return 0.0
    counts = Counter(sequence)
    total = len(sequence)
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability + 1e-12)
    return entropy


def _rollback_counts(sequence: list[int]) -> tuple[int, int, list[int]]:
    rollback_3_count = 0
    rollback_4_count = 0
    anchors: list[int] = []
    for index in range(len(sequence) - 2):
        a, b, c = sequence[index : index + 3]
        if a == c and a != b:
            rollback_3_count += 1
            anchors.append(a)
    for index in range(len(sequence) - 3):
        a, b, c, d = sequence[index : index + 4]
        if a == d and a not in {b, c}:
            rollback_4_count += 1
            anchors.append(a)
    return rollback_3_count, rollback_4_count, anchors


def _mode_value(counts: Counter[int]) -> tuple[int, int]:
    max_count = max(counts.values())
    mode_value = min(action for action, count in counts.items() if count == max_count)
    return mode_value, max_count


def extract_features(
    sequence_input: Any,
    *,
    max_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    top_hubs: tuple[int, ...] = DEFAULT_TOP_HUBS,
) -> dict[str, Any]:
    sequence = parse_sequence(sequence_input)
    sequence, clipped = normalize_sequence(sequence, max_length=max_length)

    counts = Counter(sequence)
    sequence_length = len(sequence)
    unique_action_count = len(counts)
    most_common_action, most_common_count = _mode_value(counts)
    first_action = sequence[0]
    last_action = sequence[-1]

    rollback_3_count, rollback_4_count, rollback_anchors = _rollback_counts(sequence)
    anchor_action = rollback_anchors and Counter(rollback_anchors).most_common(1)[0][0] or most_common_action

    transition_count = sum(1 for left, right in zip(sequence, sequence[1:]) if left != right)
    transition_ratio = transition_count / max(sequence_length - 1, 1)
    duplicate_ratio = 1.0 - (unique_action_count / max(sequence_length, 1))
    repeat_density = sum(count - 1 for count in counts.values()) / max(sequence_length, 1)
    action_dominance = most_common_count / max(sequence_length, 1)
    rare_action_count = sum(1 for value in sequence if value not in top_hubs)
    rare_action_ratio = rare_action_count / max(sequence_length, 1)
    entropy_value = _entropy(sequence)

    top_actions = [
        {"action": str(action), "count": count}
        for action, count in counts.most_common(5)
    ]

    hub_counts = [float(counts.get(int(hub), 0)) for hub in top_hubs[:10]]
    if len(hub_counts) < 10:
        hub_counts.extend([0.0] * (10 - len(hub_counts)))

    # Match the original notebook training pipeline exactly:
    # [length, nunique, first_item, last_item, mode_val] + top-10 hub counts.
    wide_features = [
        float(sequence_length),
        float(unique_action_count),
        float(first_action),
        float(last_action),
        float(most_common_action),
        *hub_counts,
    ]

    return {
        "sequence": sequence,
        "padded_sequence": pad_sequence(sequence, max_length=max_length),
        "sequence_length": sequence_length,
        "unique_action_count": unique_action_count,
        "anchor_action": str(anchor_action),
        "rollback_3_count": rollback_3_count,
        "rollback_4_count": rollback_4_count,
        "action_frequency_top": top_actions,
        "repeat_density": round(repeat_density, 4),
        "transition_count": transition_count,
        "transition_ratio": round(transition_ratio, 4),
        "entropy": round(entropy_value, 4),
        "rare_action_count": rare_action_count,
        "rare_action_ratio": round(rare_action_ratio, 4),
        "duplicate_ratio": round(duplicate_ratio, 4),
        "first_action": str(first_action),
        "last_action": str(last_action),
        "mode_action": str(most_common_action),
        "wide_features": wide_features,
        "was_truncated_to_max_length": clipped,
    }


def process_sequences(
    sequence_inputs: list[Any],
    *,
    max_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    top_hubs: tuple[int, ...] = DEFAULT_TOP_HUBS,
) -> list[dict[str, Any]]:
    return [
        extract_features(sequence_input, max_length=max_length, top_hubs=top_hubs)
        for sequence_input in sequence_inputs
    ]


def create_features(processed_sequences: list[dict[str, Any]]) -> list[list[float]]:
    return [list(item["wide_features"]) for item in processed_sequences]


def build_input_summary(feature_bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "sequence_length": feature_bundle["sequence_length"],
        "unique_action_count": feature_bundle["unique_action_count"],
        "anchor_action": feature_bundle["anchor_action"],
        "rollback_3_count": feature_bundle["rollback_3_count"],
        "rollback_4_count": feature_bundle["rollback_4_count"],
        "action_frequency_top": feature_bundle["action_frequency_top"],
        "repeat_density": feature_bundle["repeat_density"],
        "transition_ratio": feature_bundle["transition_ratio"],
        "entropy": feature_bundle["entropy"],
        "rare_action_ratio": feature_bundle["rare_action_ratio"],
        "was_truncated_to_max_length": feature_bundle["was_truncated_to_max_length"],
    }
