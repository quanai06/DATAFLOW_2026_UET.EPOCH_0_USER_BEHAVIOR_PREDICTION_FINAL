from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd


@dataclass
class RowFeature:
    row_id: str
    sequence_action: str
    anchor: str
    aba_count: int
    b_unique: int
    rb4: int
    length: int
    unique_count: int
    entropy: float
    nent: float
    ent_level: str
    signature: str


_ROW_BY_SEQUENCE: Dict[str, RowFeature] = {}
_GROUP_INFO_BY_SIG: Dict[str, Dict[str, str]] = {}
_CACHE: Dict[Tuple[str, str], str] = {}
_ENT_THRESHOLDS: Tuple[float, float] = (0.30, 0.70)

_TAG_HUMAN = {
    "ANCHOR_LOOP": "Lap quanh hanh dong truc (A-B-A)",
    "RB4": "Rollback 4 buoc (A-B-C-A)",
    "LONG": "Chuoi dai bat thuong",
    "HIGH_VAR": "Do bien thien hanh vi cao",
    "MIXED": "Mau hanh vi hon hop",
}

_PRIORITY_HUMAN = {
    "HIGH": "Cao",
    "MED": "Trung binh",
    "LOW": "Thap",
}


def _normalized_entropy(entropy: float, unique_count: int) -> float:
    if unique_count <= 1:
        return 0.0
    return float(entropy / (math.log2(unique_count) + 1e-12))


def _ent_level(nent: float, low_thr: float, high_thr: float) -> str:
    if nent >= high_thr + 1e-6:
        return "HIGH"
    if nent <= low_thr - 1e-6:
        return "LOW"
    return "MID"


def _len_bucket(length: int) -> str:
    if length >= 21:
        return "LONG"
    if length >= 12:
        return "MID"
    return "SHORT"


def _build_signature(
    anchor: str,
    aba_count: int,
    b_unique: int,
    rb4: int,
    length: int,
    ent_level: str,
) -> str:
    return (
        f"A{anchor}|ABA{aba_count}|BU{b_unique}|RB4{1 if rb4 > 0 else 0}|"
        f"L{_len_bucket(length)}|E{ent_level}"
    )


def _rarity_boost_from_group_size(group_size: int) -> int:
    if group_size >= 2000:
        return 4   # cực lớn, kiểu 105
    if group_size >= 300:
        return 3   # lớn, kiểu 102
    if group_size >= 100:
        return 2   # trung bình khá, kiểu 697 / 975 / 1068
    if group_size >= 30:
        return 1   # có đủ volume để đáng chú ý
    return 0


def _rule_tag_priority(row: RowFeature, group_size: int = 1) -> Tuple[str, str]:
    if row.rb4 > 0:
        tag = "RB4"
    elif row.aba_count > 0:
        tag = "ANCHOR_LOOP"
    elif row.length >= 21:
        tag = "LONG"
    elif row.ent_level == "HIGH":
        tag = "HIGH_VAR"
    else:
        tag = "MIXED"

    # =========================
    # 1) Severity score
    # =========================
    severity_score = 0

    # rollback 4 bước = tín hiệu khó hơn
    if row.rb4 >= 2:
        severity_score += 3
    elif row.rb4 == 1:
        severity_score += 2

    # rollback 3 bước
    if row.aba_count >= 4:
        severity_score += 3
    elif row.aba_count >= 3:
        severity_score += 2
    elif row.aba_count >= 2:
        severity_score += 1

    # số biến thể B trong A-B-A
    if row.b_unique >= 5:
        severity_score += 2
    elif row.b_unique >= 3:
        severity_score += 1

    # độ dài chuỗi
    if row.length >= 30:
        severity_score += 3
    elif row.length >= 21:
        severity_score += 2
    elif row.length >= 16:
        severity_score += 1

    # độ đa dạng hành vi
    if row.unique_count >= 20:
        severity_score += 3
    elif row.unique_count >= 14:
        severity_score += 2
    elif row.unique_count >= 10:
        severity_score += 1

    # entropy
    if row.ent_level == "HIGH":
        severity_score += 2
    elif row.ent_level == "MID":
        severity_score += 1

    # =========================
    # 2) Impact score
    # =========================
    impact_score = _rarity_boost_from_group_size(int(group_size))

    # =========================
    # 3) Final score
    # =========================
    final_score = severity_score + impact_score

    # =========================
    # 4) Rule chốt priority
    # =========================
    # Case rất lớn => không được LOW
    if group_size >= 2000:
        if severity_score >= 2:
            priority = "HIGH"
        else:
            priority = "MED"

    # Case lớn + có độ khó nhất định => HIGH
    elif group_size >= 300:
        if severity_score >= 3:
            priority = "HIGH"
        else:
            priority = "MED"

    # Severity rất cao thì dù volume không lớn vẫn HIGH
    elif severity_score >= 7:
        priority = "HIGH"

    elif final_score >= 5:
        priority = "MED"

    else:
        priority = "LOW"

    return tag, priority


def _parse_list_like(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x if str(v).strip() not in {"", "0", "nan", "None"}]
    s = str(x).strip()
    if not s or s in {"[]", "nan", "None"}:
        return []
    s = s.strip("[]")
    parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
    return [p for p in parts if p]


def _make_feature_from_row(row: pd.Series) -> RowFeature:
    anchor = str(row.get("anchor_action", "-"))
    aba_count = int(row.get("rb_3_steps", 0))
    rb4 = int(row.get("rb_4_steps", 0))
    length = int(row.get("length", 0))
    unique_count = int(row.get("unique_count", 1))
    entropy = float(row.get("entropy", 0.0))

    rb3_actions = _parse_list_like(row.get("rb3_actions", []))
    b_unique = len(set(rb3_actions))

    nent = _normalized_entropy(entropy, unique_count)
    low_thr, high_thr = _ENT_THRESHOLDS
    ent_level = _ent_level(nent, low_thr, high_thr)

    signature = _build_signature(
        anchor=anchor,
        aba_count=aba_count,
        b_unique=b_unique,
        rb4=rb4,
        length=length,
        ent_level=ent_level,
    )

    return RowFeature(
        row_id=str(row.get("id", "NA")),
        sequence_action=str(row.get("action_sequence", "")),
        anchor=anchor,
        aba_count=aba_count,
        b_unique=b_unique,
        rb4=rb4,
        length=length,
        unique_count=unique_count,
        entropy=entropy,
        nent=nent,
        ent_level=ent_level,
        signature=signature,
    )


def prime_group_context(df: pd.DataFrame):
    _ROW_BY_SEQUENCE.clear()
    _GROUP_INFO_BY_SIG.clear()
    _CACHE.clear()

    required_cols = {
        "id",
        "action_sequence",
        "anchor_action",
        "rb_3_steps",
        "rb_4_steps",
        "length",
        "unique_count",
        "entropy",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Thieu cot dau vao cho SLM: {sorted(missing)}")

    sig_to_rows = defaultdict(list)

    for _, row in df.iterrows():
        feat = _make_feature_from_row(row)
        _ROW_BY_SEQUENCE[feat.sequence_action] = feat
        sig_to_rows[feat.signature].append(feat)

    gid_counter = 1
    for sig, rows in sig_to_rows.items():
        gid = f"G{gid_counter:03d}"
        gid_counter += 1

        sample = rows[0]
        tag, priority = _rule_tag_priority(sample, group_size=len(rows))

        _GROUP_INFO_BY_SIG[sig] = {
            "gid": gid,
            "tag": tag,
            "p": priority,
            "count": str(len(rows)),
            "rep": str(sample.row_id),
            "anchor": sample.anchor,
            "aba": str(sample.aba_count),
            "buniq": str(sample.b_unique),
        }


def _fallback_line(sequence_action: str, fact_text: str) -> str:
    length_match = re.search(r"Do dai chuoi:\s*(\d+)", str(fact_text))
    rb3_match = re.search(r"Phat hien\s+(\d+)\s+lan lap 3 buoc", str(fact_text))
    rb4_match = re.search(r"(\d+)\s+lan lap 4 buoc", str(fact_text))
    entropy_match = re.search(r"Entropy\)\s*:\s*([0-9]+(?:\.[0-9]+)?)", str(fact_text))
    anchor_match = re.search(r"Anchor=([^\.\s]+)", str(fact_text))

    length = int(length_match.group(1)) if length_match else 0
    aba_count = int(rb3_match.group(1)) if rb3_match else 0
    rb4 = int(rb4_match.group(1)) if rb4_match else 0
    entropy = float(entropy_match.group(1)) if entropy_match else 0.0
    anchor = anchor_match.group(1) if anchor_match else "-"

    unique_count = max(1, min(length, 10))
    nent = _normalized_entropy(entropy, unique_count)
    ent_level = _ent_level(nent, *_ENT_THRESHOLDS)

    rf = RowFeature(
        row_id="NA",
        sequence_action=str(sequence_action),
        anchor=anchor,
        aba_count=aba_count,
        b_unique=0,
        rb4=rb4,
        length=length,
        unique_count=unique_count,
        entropy=entropy,
        nent=nent,
        ent_level=ent_level,
        signature=_build_signature(anchor, aba_count, 0, rb4, length, ent_level),
    )
    tag, priority = _rule_tag_priority(rf)
    return (
        f"G=G000|TAG={tag}|P={priority}|COUNT=1|REP=NA|"
        f"A={rf.anchor}|ABA={rf.aba_count}|Buniq={rf.b_unique}"
    )


def get_slm_analysis(fact_text, sequence_action):
    key = (str(fact_text), str(sequence_action))
    if key in _CACHE:
        return _CACHE[key]

    seq_key = str(sequence_action)
    rf = _ROW_BY_SEQUENCE.get(seq_key)
    if rf is None:
        out = _fallback_line(seq_key, str(fact_text))
        _CACHE[key] = out
        return out

    info = _GROUP_INFO_BY_SIG.get(rf.signature)
    if info is None:
        out = _fallback_line(seq_key, str(fact_text))
        _CACHE[key] = out
        return out

    out = (
        f"G={info['gid']}|TAG={info['tag']}|P={info['p']}|COUNT={info['count']}|REP={info['rep']}|"
        f"A={info['anchor']}|ABA={info['aba']}|Buniq={info['buniq']}"
    )
    _CACHE[key] = out
    return out


def parse_analysis_line(line: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for part in str(line).split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _as_int(value: Optional[str], default: int = 0) -> int:
    try:
        return int(str(value))
    except Exception:
        return default


def analysis_line_to_dict(line: str) -> Dict[str, Any]:
    parsed = parse_analysis_line(line)

    group_id = parsed.get("G", "G000")
    tag = parsed.get("TAG", "MIXED")
    priority = parsed.get("P", "LOW")
    group_size = _as_int(parsed.get("COUNT"), default=1)
    representative_id = parsed.get("REP", "NA")
    anchor_action = parsed.get("A", "-")
    aba_count = _as_int(parsed.get("ABA"), default=0)
    b_unique = _as_int(parsed.get("Buniq"), default=0)

    tag_human = _TAG_HUMAN.get(tag, _TAG_HUMAN["MIXED"])
    priority_human = _PRIORITY_HUMAN.get(priority, _PRIORITY_HUMAN["LOW"])

    anchor_text = "khong xac dinh action truc" if anchor_action == "-" else f"action truc A={anchor_action}"

    human_text = (
        f"Nhom {group_id} co {group_size} truong hop (mau dai dien: {representative_id}). "
        f"Kieu bat thuong: {tag_human}. "
        f"Thong tin rollback: {anchor_text}, so vong A-B-A={aba_count}, so bien the B={b_unique}. "
        f"Muc uu tien xu ly: {priority_human}."
    )

    return {
        "group_id": group_id,
        "tag": tag,
        "tag_human": tag_human,
        "priority": priority,
        "priority_human": priority_human,
        "group_size": group_size,
        "representative_id": representative_id,
        "anchor_action": anchor_action,
        "aba_count": aba_count,
        "b_unique": b_unique,
        "human_text": human_text,
        "raw": str(line),
    }


def explain_analysis_line(line: str) -> str:
    return str(analysis_line_to_dict(line)["human_text"])


def enrich_analysis_dataframe(
    df,
    analysis_column: str = "ai_assistance",
    human_column: str = "ai_assistance_human",
):
    if analysis_column not in df.columns:
        raise ValueError(f"Khong tim thay cot '{analysis_column}' trong DataFrame.")

    out = df.copy()
    parsed_rows = out[analysis_column].fillna("").map(analysis_line_to_dict)

    out["triage_group_id"] = parsed_rows.map(lambda x: x["group_id"])
    out["triage_tag"] = parsed_rows.map(lambda x: x["tag"])
    out["triage_tag_human"] = parsed_rows.map(lambda x: x["tag_human"])
    out["triage_priority"] = parsed_rows.map(lambda x: x["priority"])
    out["triage_priority_human"] = parsed_rows.map(lambda x: x["priority_human"])
    out["triage_group_size"] = parsed_rows.map(lambda x: x["group_size"])
    out["triage_representative_id"] = parsed_rows.map(lambda x: x["representative_id"])
    out["triage_anchor_action"] = parsed_rows.map(lambda x: x["anchor_action"])
    out["triage_aba_count"] = parsed_rows.map(lambda x: x["aba_count"])
    out["triage_b_unique"] = parsed_rows.map(lambda x: x["b_unique"])
    out[human_column] = parsed_rows.map(lambda x: x["human_text"])
    return out