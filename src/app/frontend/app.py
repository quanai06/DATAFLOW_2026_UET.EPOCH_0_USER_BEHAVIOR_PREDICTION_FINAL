from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.frontend.api_client import (
    ApiClientError,
    get_dataset_overview,
    get_health,
    get_order,
    get_orders,
    get_planning_overview,
    predict_order,
    predict_order_live,
)
from src.app.frontend.charts import (
    action_frequency_chart,
    batch_distribution_chart,
    gauge_chart,
    output_bar_chart,
    selection_frontier_chart,
    top_orders_chart,
    tradeoff_scatter_chart,
)

from src.app.frontend.components import badge, inject_styles, metric_card, recommendation_panel, section_header, warning_panel
from src.app.frontend.utils import init_session_state, order_label, sequence_to_text


st.set_page_config(page_title="Risk-Aware Dynamic Scheduler", page_icon="R", layout="wide")
inject_styles(st)
init_session_state(st)


def _safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ApiClientError as exc:
        st.error(str(exc))
        return None


def _load_health() -> dict:
    return _safe_call(get_health) or {
        "status": "offline",
        "predictor_type": "unavailable",
        "runtime_ready": False,
        "default_source_mode": "precomputed",
        "precomputed_ready": False,
        "live_ready": False,
    }


def _load_dataset_overview() -> dict | None:
    if st.session_state.dataset_overview is None:
        st.session_state.dataset_overview = _safe_call(get_dataset_overview)
    return st.session_state.dataset_overview


def _load_planning_overview(force_refresh: bool = False) -> dict | None:
    if force_refresh or st.session_state.planning_overview is None:
        planning_table_start = max(1, int(st.session_state.planning_table_start))
        planning_table_end = max(planning_table_start, int(st.session_state.planning_table_end))
        planning_table_limit = min(100, planning_table_end - planning_table_start + 1)
        latest_planning = _safe_call(
            get_planning_overview,
            limit=10,
            capacity_budget_pct=float(st.session_state.capacity_budget_pct),
            warehouse_budget_pct=float(st.session_state.warehouse_budget_pct),
            planning_table_offset=planning_table_start - 1,
            planning_table_limit=planning_table_limit,
        )
        if latest_planning is not None:
            st.session_state.planning_overview = latest_planning
    return st.session_state.planning_overview


def _search_orders(query: str) -> None:
    cleaned_query = query.strip()
    if not cleaned_query:
        st.warning("Hay nhap order_id hoac mot phan order_id de tim.")
        return
    st.session_state.orders_cache = _safe_call(get_orders, query=cleaned_query, limit=20) or []


def _load_selected_order(order_id: str, *, use_live_mode: bool = False) -> None:
    detail = _safe_call(get_order, order_id)
    if detail is None:
        return

    st.session_state.selected_order_id = order_id
    st.session_state.selected_order_detail = detail
    st.session_state.sequence_text = sequence_to_text(detail["raw_sequence"])

    prediction = _safe_call(predict_order_live if use_live_mode else predict_order, order_id)
    if prediction is not None:
        st.session_state.single_prediction = prediction


def _format_minutes(value: float | int | None) -> str:
    if value is None:
        return "0m"
    total_minutes = max(float(value), 0.0)
    hours = int(total_minutes // 60)
    minutes = int(round(total_minutes - (hours * 60)))
    if minutes == 60:
        hours += 1
        minutes = 0
    if hours and minutes:
        return f"{hours}h {minutes}m"
    if hours:
        return f"{hours}h"
    return f"{minutes}m"


def _render_header(health: dict, overview: dict | None) -> None:
    st.markdown('<div class="dashboard-shell">', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Risk-Aware Dynamic Scheduler</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Hanh vi bien dong 4 tuan cua khach -> 6 Outputs -> quyet dinh san xuat va phan bo kho bai ngay hom nay</div>',
        unsafe_allow_html=True,
    )

    left, middle, right, extra = st.columns([1.1, 1.3, 1.5, 2.8])
    with left:
        badge(st, str(health.get("status", "unknown")).upper())
    with middle:
        badge(st, str(health.get("default_source_mode", "precomputed")).upper())
    with right:
        badge(st, str(health.get("predictor_type", "unknown")).replace("-", " ").upper())
    with extra:
        generated_from = overview.get("generated_at") if overview else None
        st.markdown(
            f'<div class="mini-note">Precomputed ready: {health.get("precomputed_ready")} | Live ready: {health.get("live_ready")} | Generated from real X_test at: {generated_from or "N/A"}</div>',
            unsafe_allow_html=True,
        )


def _render_chain_section() -> None:
    section_header(st, "Decision Chain")
    st.markdown(
        """
        <div class="chain-panel">
            <strong>Step 1:</strong> Hanh vi bien dong 4 tuan cua khach duoc doc tu X_test.<br>
            <strong>Step 2:</strong> Ensemble model that suy ra 6 outputs attr_1..attr_6.<br>
            <strong>Step 3:</strong> Scheduler tinh nguoc tu completion window va workload de ra ty le nen chay hom nay va ap luc kho cho xuat.<br>
            <strong>Step 4:</strong> Planning engine uoc tinh so phut xu ly tu attr_3 va attr_6, sau do xep lan luot cac order vao workday 480 phut va giu percent load/kho de theo doi.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_overview(overview: dict | None, health: dict) -> None:
    section_header(st, "Dataset Overview")
    if overview is None:
        st.warning("Chua tai duoc thong tin precomputed dataset.")
        return

    cards = [
        ("Total Orders", str(overview["total_orders"]), "precomputed from real X_test"),
        ("Avg Length", f"{overview['average_sequence_length']:.1f}", "behavior depth"),
        ("P95 Length", str(overview["p95_sequence_length"]), "tail sequence length"),
        ("Max Length", str(overview["max_sequence_length"]), "longest order"),
        ("Model Count", str(health.get("model_count", 0)), "ensemble artifacts"),
    ]
    columns = st.columns(5)
    for column, (label, value, caption) in zip(columns, cards):
        with column:
            metric_card(st, label, value, caption)


def _planning_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows,
        columns=[
            "order_id",
            "priority_level",
            "recommended_action",
            "planning_day_bucket",
            "start_date",
            "end_date",
            "estimated_processing_minutes",
            "planned_order_sequence",
            "planned_start_minute",
            "planned_end_minute",
            "cumulative_selected_processing_minutes",
            "today_production_pct",
            "warehouse_waiting_pressure_pct",
            "risk_score",
            "risk_level",
            "planning_rank_score",
            "behavior_diversity_score",
            "sequence_length",
            "unique_action_count",
            "plan_status",
            "anchor_action",
            "rollback_3_count",
            "rollback_4_count",
            "entropy",
            "rare_action_ratio",
            "capacity_band",
            "completion_urgency_band",
            "warehouse_stress_zone",
        ],
    )
    if not frame.empty:
        frame = frame.rename(
            columns={
                "today_production_pct": "production_pct",
                "warehouse_waiting_pressure_pct": "warehouse_pct",
            }
        )
    return frame


def _render_planning_engine(planning: dict | None) -> None:
    section_header(st, "Planning Level: Compare Orders Across The Portfolio")
    budget_left, budget_mid, budget_right = st.columns([1.2, 1.2, 0.8])
    with budget_left:
        st.session_state.capacity_budget_pct = st.slider(
            "Legacy factory load reference",
            min_value=20.0,
            max_value=400.0,
            value=float(st.session_state.capacity_budget_pct),
            step=5.0,
        )
    with budget_mid:
        st.session_state.warehouse_budget_pct = st.slider(
            "Legacy warehouse stress reference",
            min_value=20.0,
            max_value=500.0,
            value=float(st.session_state.warehouse_budget_pct),
            step=5.0,
        )
    with budget_right:
        st.write("")
        if st.button("Rebuild Plan", type="primary", use_container_width=True):
            planning = _load_planning_overview(force_refresh=True)

    if planning is None:
        st.warning("Chua tai duoc planning overview.")
        return

    st.markdown(
        '<div class="mini-note"><strong>Time-aware planner:</strong> daily selection is now driven by a fixed <strong>480-minute</strong> workday. The two sliders remain as legacy reference gauges so the existing load and warehouse charts stay comparable.</div>',
        unsafe_allow_html=True,
    )

    planning_table_total_count = int(planning.get("planning_table_total_count", 0))
    max_row_number = max(1, planning_table_total_count)

    st.session_state.planning_table_start = min(
        max_row_number,
        max(1, int(st.session_state.planning_table_start)),
    )
    st.session_state.planning_table_end = min(
        max_row_number,
        max(int(st.session_state.planning_table_start), int(st.session_state.planning_table_end)),
    )
    if st.session_state.planning_table_end - st.session_state.planning_table_start >= 100:
        st.session_state.planning_table_end = min(
            max_row_number,
            int(st.session_state.planning_table_start) + 99,
        )

    range_left, range_mid, range_right = st.columns([1.1, 1.1, 0.8])
    with range_left:
        st.session_state.planning_table_start = int(
            st.number_input(
                "Start row",
                min_value=1,
                max_value=max_row_number,
                value=int(st.session_state.planning_table_start),
                step=100,
            )
        )
    with range_mid:
        max_end_value = min(max_row_number, int(st.session_state.planning_table_start) + 99)
        st.session_state.planning_table_end = int(
            st.number_input(
                "End row",
                min_value=int(st.session_state.planning_table_start),
                max_value=max_end_value,
                value=min(int(st.session_state.planning_table_end), max_end_value),
                step=10,
            )
        )
    with range_right:
        st.write("")
        if st.button("Load Range", use_container_width=True):
            planning = _load_planning_overview(force_refresh=True)

    if planning is None:
        st.warning("Khong the cap nhat planning overview luc nay.")
        return

    selected_frame = _planning_frame(planning["selected_orders_for_today"])
    deferred_frame = _planning_frame(planning["deferred_orders"])
    priority_frame = _planning_frame(planning["top_priority_orders"])
    accelerate_frame = _planning_frame(planning["top_accelerate_orders"])
    hold_frame = _planning_frame(planning["top_hold_orders"])
    warehouse_frame = _planning_frame(planning["top_warehouse_pressure_orders"])
    risk_frame = _planning_frame(planning["top_risk_orders"])
    planning_table = _planning_frame(planning["planning_table"])
    planning_table_offset = int(planning.get("planning_table_offset", 0))
    planning_table_limit = int(planning.get("planning_table_limit", len(planning_table)))
    planning_table_end = min(planning_table_total_count, planning_table_offset + planning_table_limit)

    selected_minutes = float(planning.get("cumulative_selected_processing_minutes", 0.0))
    day_time_budget = float(planning.get("daily_time_budget_minutes", 480.0))
    primary_summary_cols = st.columns(5)
    primary_summary_cards = [
        ("Orders Today", str(planning.get("selected_orders_count", 0)), "so lenh can xu ly trong ca hom nay"),
        ("Deferred", str(planning.get("deferred_orders_count", 0)), "don chua vao duoc ke hoach hom nay"),
        ("Planned Time", _format_minutes(selected_minutes), "tong thoi gian xu ly da xep lich"),
        ("Remaining Time", _format_minutes(planning.get("remaining_time_budget_minutes", 0.0)), "thoi gian con lai trong ca"),
        ("Day Fill", f"{planning.get('day_time_utilization_pct', 0.0):.1f}%", f"muc lap day cua ca {_format_minutes(day_time_budget)}"),
    ]
    for column, (label, value, caption) in zip(primary_summary_cols, primary_summary_cards):
        with column:
            metric_card(st, label, value, caption)

    st.markdown(
        f'<div class="mini-note"><strong>Planning logic:</strong> Today duoc co dinh la <strong>01/01</strong>. Planner uu tien cao nhat cho order co <code>start_date = end_date = 01/01</code>, sau do den nhom <code>start_date = 01/01</code>, roi moi toi cac order o nhung ngay xa hon. Neu cung nhom ngay, planner van sap xep <strong>HIGH -&gt; MEDIUM -&gt; LOW</strong> theo ranking hien tai. Moi order duoc gan <strong>estimated processing minutes</strong> tu <code>attr_3</code> va <code>attr_6</code>, sau do planner xep lan luot cho toi khi day <strong>{_format_minutes(day_time_budget)}</strong>. <br><strong>Cut-off logic:</strong> {planning["cutoff_reason"]}</div>',
        unsafe_allow_html=True, 
    )
    st.markdown(
        """
        <div class="mini-note">
            <strong>How to read the board:</strong>
            Chart mau xanh va mau cam giup so sanh truc tiep muc tai san xuat va muc ap luc kho theo tung order.
            Donut chart cho biet co cau lenh dieu do, con budget chart ben phai giu vai tro theo doi cac percent budget legacy sau khi planner da chot theo gio.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="mini-note"><strong>Planning table range:</strong> dang hien thi dong {planning_table_offset + 1:,} den {planning_table_end:,} tren tong {planning_table_total_count:,} dong. Ban co the nhap Start va End trong toan bo du lieu, nhung moi luot chi xem toi da 100 dong.</div>',
        unsafe_allow_html=True,
    )

        # ===== Planning charts =====
    st.plotly_chart(
        tradeoff_scatter_chart(planning_table, "Production vs Warehouse Load by Order"),
        use_container_width=True,
    )

    chart_mid_left, chart_mid_right = st.columns(2)
    with chart_mid_left:
        st.plotly_chart(
            selection_frontier_chart(selected_frame, "Cumulative Load of Selected Orders"),
            use_container_width=True,
        )
    with chart_mid_right:
        st.plotly_chart(
            batch_distribution_chart(
                planning_table,
                "recommended_action",
                "Recommended Action Distribution",
                {
                    "ACCELERATE": "#8f1d1d",
                    "MAINTAIN": "#315c8a",
                    "SLOW_DOWN": "#8a5600",
                    "HOLD": "#5a4a42",
                },
            ),
            use_container_width=True,
        )

    chart_bottom_left, chart_bottom_right = st.columns(2)
    with chart_bottom_left:
        st.plotly_chart(
            top_orders_chart(
                priority_frame,
                "planning_rank_score",
                "Top Orders Admitted Into Today's Plan",
                "#0d3b66",
            ),
            use_container_width=True,
        )
    with chart_bottom_right:
        st.plotly_chart(
            top_orders_chart(
                warehouse_frame,
                "warehouse_pct",
                "Orders Creating Highest Warehouse Stress",
                "#c46b48",
            ),
            use_container_width=True,
        )

    section_header(st, "Today's Execution Queue")
    st.markdown(
        '<div class="mini-note">Danh sach duoi day la toan bo don da duoc xep vao ca hom nay, theo dung thu tu planner khuyen nghi thuc hien tren xuyen suot 480 phut.</div>',
        unsafe_allow_html=True,
    )
    if selected_frame.empty:
        st.info("Hom nay chua co don nao duoc dua vao execution queue.")
    else:
        execution_queue = selected_frame.copy()
        execution_queue["step"] = execution_queue["planned_order_sequence"].fillna(0).astype(int)
        execution_queue["processing_time"] = execution_queue["estimated_processing_minutes"].map(_format_minutes)
        execution_queue["time_window"] = execution_queue.apply(
            lambda row: (
                f"{_format_minutes(row['planned_start_minute'])} -> {_format_minutes(row['planned_end_minute'])}"
                if pd.notna(row["planned_start_minute"]) and pd.notna(row["planned_end_minute"])
                else "-"
            ),
            axis=1,
        )
        execution_queue = execution_queue.loc[
            :,
            [
                "step",
                "order_id",
                "priority_level",
                "recommended_action",
                "processing_time",
                "time_window",
                "start_date",
                "end_date",
            ],
        ].rename(
            columns={
                "step": "Seq",
                "order_id": "Order ID",
                "priority_level": "Priority",
                "recommended_action": "Action",
                "processing_time": "Processing Time",
                "time_window": "Planned Window",
                "start_date": "Start Date",
                "end_date": "End Date",
            }
        )
        st.dataframe(execution_queue, use_container_width=True, hide_index=True)

    section_header(st, "Planning Drill-down")
    inspect_candidates = planning_table["order_id"].tolist() if not planning_table.empty else []
    if inspect_candidates:
        st.session_state.planning_focus_order_id = st.selectbox(
            "Inspect one order from the planning board",
            options=inspect_candidates,
            index=0 if st.session_state.planning_focus_order_id not in inspect_candidates else inspect_candidates.index(st.session_state.planning_focus_order_id),
        )
        inspect_left, inspect_right = st.columns(2)
        with inspect_left:
            if st.button("Inspect Selected Planning Order", use_container_width=True):
                _load_selected_order(st.session_state.planning_focus_order_id, use_live_mode=False)
        with inspect_right:
            if st.button("Inspect Selected Planning Order Live", use_container_width=True):
                _load_selected_order(st.session_state.planning_focus_order_id, use_live_mode=True)

    section_header(st, "Planning Table")
    st.dataframe(planning_table, use_container_width=True, hide_index=True)


def _render_order_loader() -> None:
    section_header(st, "Order Level: Load One Order Only When Needed")
    st.markdown(
        '<div class="mini-note">Lúc app mo len khong co chart nao cua order duoc load san. Ban chi fetch detail/order-level chart khi can inspect mot order cu the.</div>',
        unsafe_allow_html=True,
    )

    search_left, search_mid, search_right = st.columns([2.2, 1, 1])
    with search_left:
        order_query = st.text_input(
            "Order search or direct order_id",
            key="loaded_order_id_input",
            placeholder="Vi du: n6r61",
        )
    with search_mid:
        if st.button("Search", use_container_width=True):
            _search_orders(order_query)
    with search_right:
        if st.button("Load Exact Order", type="primary", use_container_width=True) and order_query.strip():
            _load_selected_order(order_query.strip(), use_live_mode=False)

    if st.session_state.orders_cache:
        selected_candidate = st.selectbox(
            "Search results",
            options=[item["order_id"] for item in st.session_state.orders_cache],
            format_func=lambda order_id: order_label(next(item for item in st.session_state.orders_cache if item["order_id"] == order_id)),
        )
        choose_left, choose_right = st.columns(2)
        with choose_left:
            if st.button("Load Selected Precomputed", use_container_width=True):
                _load_selected_order(selected_candidate, use_live_mode=False)
        with choose_right:
            if st.button("Run Live Inference For Selected", use_container_width=True):
                _load_selected_order(selected_candidate, use_live_mode=True)


def _render_raw_behavior(order_detail: dict) -> None:
    section_header(st, "Section 1: Raw 4-week Behavior")
    left, right = st.columns([1.2, 2.4])
    with left:
        metric_card(st, "Order ID", order_detail["order_id"], "loaded on demand")
        metric_card(st, "Sequence Length", str(order_detail["sequence_length"]), "after preprocessing")
    with right:
        st.code(sequence_to_text(order_detail["raw_sequence"]), language="text")

    summary = order_detail["input_summary"]
    columns = st.columns(6)
    cards = [
        ("Unique Actions", str(summary["unique_action_count"]), "behavior variety"),
        ("Anchor Action", summary["anchor_action"], "dominant return action"),
        ("Rollback 3", str(summary["rollback_3_count"]), "A-B-A"),
        ("Rollback 4", str(summary["rollback_4_count"]), "A-B-C-A"),
        ("Entropy", f"{summary.get('entropy', 0):.2f}", "behavior spread"),
        ("Rare Action Ratio", f"{summary.get('rare_action_ratio', 0):.2f}", "uncommon hubs"),
    ]
    for column, (label, value, caption) in zip(columns, cards):
        with column:
            metric_card(st, label, value, caption)

    st.plotly_chart(action_frequency_chart(summary["action_frequency_top"]), use_container_width=True)


def _render_outputs(payload: dict) -> None:
    section_header(st, "Section 2: 6 Outputs From Model")
    st.markdown(
        "- `attr_3`: estimated factory workload signal used to size today's production effort\n"
        "- `attr_4`, `attr_5`: completion timing signals used to estimate urgency and delivery window\n"
        "- `attr_6`: volatility / uncertainty proxy used by the risk layer and as a time buffer in processing estimates"
    )

    columns = st.columns(6)
    for column, (key, value) in zip(columns, payload["predicted_outputs"].items()):
        with column:
            metric_card(st, key.upper(), str(value), "model output")

    st.plotly_chart(output_bar_chart(payload["predicted_outputs"]), use_container_width=True)


def _render_scheduler(payload: dict) -> None:
    section_header(st, "Section 3: Dynamic Scheduler Today")
    decision = payload["scheduler_decision"]

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            gauge_chart(decision["today_production_pct"], title="Today Production %", bar_color="#0d3b66"),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            gauge_chart(
                decision["warehouse_waiting_pressure_pct"],
                title="Warehouse Waiting Pressure %",
                bar_color="#c46b48",
            ),
            use_container_width=True,
        )

    info_cols = st.columns(6)
    metrics = [
        ("Capacity Band", decision.get("capacity_band", "-"), "workload zone"),
        ("Urgency Band", decision.get("completion_urgency_band", "-"), "completion window"),
        ("Window Days", str(decision.get("completion_window_days", "-")), "predicted window"),
        ("Est. Time", _format_minutes(decision.get("estimated_processing_minutes", 0.0)), "expected processing time"),
        ("Warehouse Zone", decision.get("warehouse_stress_zone", "-"), "inventory stress"),
        ("Priority", decision["priority_level"], "execution priority"),
    ]
    for column, (label, value, caption) in zip(info_cols, metrics):
        with column:
            metric_card(st, label, value, caption)

    badge_col, action_col, text_col = st.columns([1, 1, 2])
    with badge_col:
        badge(st, decision["priority_level"])
    with action_col:
        badge(st, decision["recommended_action"])
    with text_col:
        st.markdown(f'<div class="mini-note">{decision["explanation"]}</div>', unsafe_allow_html=True)

    recommendation_panel(st, decision, payload["risk_assessment"])


def _render_risk(payload: dict) -> None:
    section_header(st, "Section 4: Risk")
    risk = payload["risk_assessment"]
    columns = st.columns(4)
    cards = [
        ("Risk Score", f"{risk['risk_score']:.2f}", "0 -> 1"),
        ("Risk Level", risk["risk_level"], "traffic-light warning"),
        ("Virtual Inventory", "YES" if risk["virtual_inventory_warning"] else "NO", "warehouse risk"),
        ("Confidence Proxy", f"{risk['confidence_proxy']:.2f}", "lower = noisier"),
    ]
    for column, (label, value, caption) in zip(columns, cards):
        with column:
            metric_card(st, label, value, caption)

    if risk["risk_level"] == "RED":
        warning_panel(st, "Warning", "Recommend slowing down production to avoid virtual inventory risk.")

    st.markdown("**Risk reasons**")
    for reason in risk["risk_reasons"]:
        st.markdown(f"- {reason}")


def _render_model_info(payload: dict, overview: dict | None) -> None:
    section_header(st, "Scaling And Real-World Adoption")
    artifacts = payload.get("model_artifacts") or {}
    total_orders = (overview or {}).get("total_orders", "N/A")
    st.markdown(
        f"""
        - Demo mode hien tai la **{payload.get('source_mode', 'unknown')}** de UI lookup nhanh tren tap **{total_orders}** order.
        - Ensemble models: **{artifacts.get('model_count', 0)}** | Aggregation: **{artifacts.get('aggregation', 'unknown')}**
        - Planning engine khong rerun model. No chi dung precomputed outputs de xay dung daily scenario theo 480 phut xu ly trong ngay, trong khi van giu percent load va warehouse budget de doi chieu.
        - Thuc te co the chay offline precompute theo lich gio/ngay, sau do dashboard chi doc artifact, phu hop cho ERP/WMS planning va scaling.
        - Live mode van ton tai de doi chieu model runtime, nhung khong phai duong chay mac dinh.
        """
    )


health_payload = _load_health()
overview_payload = _load_dataset_overview()
_load_planning_overview()

_render_header(health_payload, overview_payload)
_render_chain_section()
_render_overview(overview_payload, health_payload)
st.write("")
_render_planning_engine(st.session_state.planning_overview)
st.write("")
_render_order_loader()

if st.session_state.selected_order_detail and st.session_state.single_prediction:
    st.write("")
    _render_raw_behavior(st.session_state.selected_order_detail)
    st.write("")
    _render_outputs(st.session_state.single_prediction)
    st.write("")
    _render_scheduler(st.session_state.single_prediction)
    st.write("")
    _render_risk(st.session_state.single_prediction)
    st.write("")
    _render_model_info(st.session_state.single_prediction, overview_payload)

st.markdown(
    """
    <div class="mini-note" style="margin-top:1.2rem;">
        Default mode cua app la precomputed-first. Planning engine chay tren artifact da precompute, con order-level chart chi duoc fetch khi ban inspect mot order cu the.
    </div>
    </div>
    """,
    unsafe_allow_html=True,
)
