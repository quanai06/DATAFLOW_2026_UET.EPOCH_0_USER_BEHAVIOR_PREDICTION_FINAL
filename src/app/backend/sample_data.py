from __future__ import annotations


SAMPLE_ORDERS = [
    {
        "sample_id": "ORD-001",
        "title": "Stable Short Loop",
        "description": "Chuoi ngan, lap lai nhe, phu hop de demo profile on dinh.",
        "raw_sequence": [105, 102, 105, 103, 105, 606, 105],
        "tags": ["stable", "short", "light-workload"],
        "expected_profile": "stable",
    },
    {
        "sample_id": "ORD-002",
        "title": "Stable Medium Order",
        "description": "Don tam trung voi tan suat quay lai deu, thich hop muc maintain.",
        "raw_sequence": [102, 105, 103, 102, 105, 103, 606, 760, 606, 760, 606],
        "tags": ["stable", "medium", "balanced"],
        "expected_profile": "balanced",
    },
    {
        "sample_id": "ORD-003",
        "title": "Rollback 3-Step Order",
        "description": "Co nhieu A-B-A nen thay ro rollback 3 buoc.",
        "raw_sequence": [105, 102, 105, 103, 105, 606, 105, 603, 105, 709, 105],
        "tags": ["rollback-3", "volatile"],
        "expected_profile": "risky",
    },
    {
        "sample_id": "ORD-004",
        "title": "Rollback 4-Step Order",
        "description": "Co A-B-C-A lap lai, gay ap luc dieu phôi kho va cong suat.",
        "raw_sequence": [775, 929, 867, 775, 697, 975, 1068, 697, 1265, 1353, 1072, 1265],
        "tags": ["rollback-4", "urgent", "complex"],
        "expected_profile": "high-risk",
    },
    {
        "sample_id": "ORD-005",
        "title": "Bursty Repeat Heavy",
        "description": "Mot so action lap lai dam dac, hop de xem warehouse pressure.",
        "raw_sequence": [105, 105, 105, 102, 105, 105, 105, 103, 105, 105, 606, 105],
        "tags": ["repeat-heavy", "warehouse"],
        "expected_profile": "pressure-heavy",
    },
    {
        "sample_id": "ORD-006",
        "title": "High Unique Action Order",
        "description": "Do phan manh cao, nhieu hub khac nhau trong 4 tuan.",
        "raw_sequence": [4347, 697, 975, 1068, 109, 1265, 1353, 1072, 979, 8615, 603, 709, 685, 621],
        "tags": ["high-unique", "complex"],
        "expected_profile": "complex",
    },
    {
        "sample_id": "ORD-007",
        "title": "Long Complex Order",
        "description": "Chuoi dai, vua nhieu diem quay lai vua nhieu chuyen trang thai.",
        "raw_sequence": [
            102, 105, 103, 606, 760, 606, 603, 709, 603, 685, 621, 685, 102, 105, 103,
            697, 975, 1068, 697, 1265, 1353, 1072, 1265, 979, 8615, 979,
        ],
        "tags": ["long", "complex", "high-workload"],
        "expected_profile": "high-capacity",
    },
    {
        "sample_id": "ORD-008",
        "title": "Unstable Risky Order",
        "description": "Dao chieu lien tuc, hop de demo warning zone.",
        "raw_sequence": [105, 102, 105, 775, 929, 867, 775, 105, 603, 105, 697, 975, 697, 105],
        "tags": ["risky", "warning", "unstable"],
        "expected_profile": "red-risk",
    },
    {
        "sample_id": "ORD-009",
        "title": "Low Risk High Volume",
        "description": "Nhieu ban ghi nhung mau hanh vi tuong doi co quy luat.",
        "raw_sequence": [102, 103, 606, 760, 603, 709, 685, 621, 102, 103, 606, 760, 603, 709, 685, 621],
        "tags": ["volume", "low-risk", "maintain"],
        "expected_profile": "high-volume",
    },
    {
        "sample_id": "ORD-010",
        "title": "High Volatility Edge Case",
        "description": "Profile goc canh, vua dai vua nhieu rollback va action hiem.",
        "raw_sequence": [
            4347, 105, 102, 105, 775, 929, 867, 775, 20278, 1059, 103, 20278, 1027,
            760, 1165, 105, 946, 103, 164, 760, 889, 105, 15342, 760,
        ],
        "tags": ["edge-case", "volatile", "red-risk"],
        "expected_profile": "edge-case",
    },
]


def get_sample_orders() -> list[dict]:
    return SAMPLE_ORDERS


def get_demo_batch() -> list[dict]:
    return [
        {"order_id": item["sample_id"], "sequence": item["raw_sequence"]}
        for item in SAMPLE_ORDERS[:6]
    ]

