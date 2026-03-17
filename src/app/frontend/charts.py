from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PAPER_BG = "#FDFDFD"
PLOT_BG = "#F1EDED"
GRID = "#d9e1ea"
TEXT = "#233142"
MUTED = "#5b6b7a"
BLUE = "#1f4e79"
BLUE_LIGHT = "#4f86c6"
ORANGE = "#c96f3b"
RED = "#9f1f1f"
GOLD = "#a66a00"
GRAY = "#7d8a99"


def _apply_enterprise_layout(
    figure: go.Figure,
    *,
    title: str,
    height: int,
    xaxis_title: str = "",
    yaxis_title: str = "",
    showlegend: bool = True,
) -> go.Figure:
    figure.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=18, color=TEXT)),
        height=height,
        margin=dict(l=28, r=24, t=64, b=24),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT, size=13),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#d7dee7",
            borderwidth=1,
            font=dict(color=TEXT, size=12),
        ),
        showlegend=showlegend,
    )
    figure.update_xaxes(
        title=xaxis_title,
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        linecolor=GRID,
        tickfont=dict(color=TEXT),
        title_font=dict(color=MUTED),
    )
    figure.update_yaxes(
        title=yaxis_title,
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        linecolor=GRID,
        tickfont=dict(color=TEXT),
        title_font=dict(color=MUTED),
    )
    figure.update_annotations(font_color=TEXT)
    return figure


def output_bar_chart(predicted_outputs: dict[str, int]) -> Any:
    label_map = {
        "attr_1": "Start Month",
        "attr_2": "Start Day",
        "attr_3": "Factory Load",
        "attr_4": "End Month",
        "attr_5": "End Day",
        "attr_6": "Volatility",
    }

    frame = pd.DataFrame(
        {
            "Output": [label_map.get(k, k) for k in predicted_outputs.keys()],
            "Value": list(predicted_outputs.values()),
        }
    )

    figure = px.bar(
        frame,
        x="Value",
        y="Output",
        orientation="h",
        color="Value",
        color_continuous_scale=["#c9d9ef", BLUE],
        title="Model Output Signals",
        text="Value",
    )
    figure.update_traces(textposition="outside", marker_line_color="#d7dee7", marker_line_width=1)
    figure.update_layout(coloraxis_showscale=False)
    figure.update_yaxes(categoryorder="total ascending")
    return _apply_enterprise_layout(
        figure,
        title="Model Output Signals",
        height=360,
        xaxis_title="Predicted value",
        yaxis_title="",
        showlegend=False,
    )


def action_frequency_chart(action_frequency_top: list[dict[str, Any]]) -> Any:
    frame = pd.DataFrame(action_frequency_top)
    if frame.empty:
        frame = pd.DataFrame({"action": ["N/A"], "count": [0]})

    figure = px.bar(
        frame,
        x="count",
        y="action",
        orientation="h",
        color="count",
        color_continuous_scale=["#f7d9c7", ORANGE],
        title="Most Frequent Actions In Sequence",
        text="count",
    )
    figure.update_traces(textposition="outside", marker_line_color="#eaded4", marker_line_width=1)
    figure.update_layout(coloraxis_showscale=False)
    figure.update_yaxes(categoryorder="total ascending")
    return _apply_enterprise_layout(
        figure,
        title="Most Frequent Actions In Sequence",
        height=320,
        xaxis_title="Frequency",
        yaxis_title="Action ID",
        showlegend=False,
    )


def gauge_chart(value: float, *, title: str, bar_color: str, max_value: float = 100.0) -> Any:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "font": {"color": TEXT, "size": 34}},
            title={"text": "", "font": {"color": TEXT, "size": 18}},
            gauge={
                "axis": {"range": [0, max_value], "tickcolor": MUTED, "tickfont": {"color": MUTED}},
                "bar": {"color": bar_color},
                "bgcolor": "#eef3f8",
                "bordercolor": "#d7dee7",
                "borderwidth": 1,
                "steps": [
                    {"range": [0, max_value * 0.35], "color": "#d8f3dc"},
                    {"range": [max_value * 0.35, max_value * 0.70], "color": "#fff3bf"},
                    {"range": [max_value * 0.70, max_value], "color": "#ffd6d6"},
                ],
            },
        )
    )
    return _apply_enterprise_layout(
        figure,
        title=title,
        height=270,
        showlegend=False,
    )


def batch_distribution_chart(frame: pd.DataFrame, column: str, title: str, color_map: dict[str, str]) -> Any:
    if frame.empty or column not in frame:
        frame = pd.DataFrame({column: []})

    counts = frame[column].value_counts().rename_axis(column).reset_index(name="count")
    if counts.empty:
        counts = pd.DataFrame({column: ["No Data"], "count": [1]})

    figure = px.pie(
        counts,
        names=column,
        values="count",
        title=title,
        hole=0.58,
        color=column,
        color_discrete_map=color_map,
    )
    figure.update_traces(
        textposition="inside",
        textinfo="percent+label",
        textfont=dict(color="#ffffff", size=13),
        marker=dict(line=dict(color=PLOT_BG, width=2)),
        sort=False,
    )
    return _apply_enterprise_layout(
        figure,
        title=title,
        height=340,
        showlegend=True,
    )


def top_orders_chart(frame: pd.DataFrame, value_column: str, title: str, color: str) -> Any:
    if frame.empty or value_column not in frame:
        frame = pd.DataFrame({"order_id": [], value_column: []})

    trimmed = frame.sort_values(value_column, ascending=False).head(8)
    figure = px.bar(
        trimmed,
        x=value_column,
        y="order_id",
        orientation="h",
        title=title,
        text=value_column,
    )
    figure.update_traces(marker_color=color, textposition="outside")
    figure.update_yaxes(categoryorder="total ascending")
    return _apply_enterprise_layout(
        figure,
        title=title,
        height=380,
        xaxis_title="Score / Percentage",
        yaxis_title="",
        showlegend=False,
    )


def tradeoff_scatter_chart(frame: pd.DataFrame, title: str) -> Any:
    plot_frame = frame.copy()

    if plot_frame.empty:
        plot_frame = pd.DataFrame(
            {
                "production_pct": [],
                "warehouse_pct": [],
                "recommended_action": [],
                "risk_score": [],
                "order_id": [],
                "planning_rank_score": [],
            }
        )

    if "planning_rank_score" in plot_frame.columns:
        plot_frame = plot_frame.sort_values("planning_rank_score", ascending=False)
    plot_frame = plot_frame.head(12)

    melted = plot_frame.melt(
        id_vars=["order_id"],
        value_vars=["production_pct", "warehouse_pct"],
        var_name="metric",
        value_name="value",
    )
    melted["metric"] = melted["metric"].map(
        {
            "production_pct": "Production %",
            "warehouse_pct": "Warehouse %",
        }
    )

    figure = px.bar(
        melted,
        x="value",
        y="order_id",
        color="metric",
        barmode="group",
        orientation="h",
        title=title,
        color_discrete_map={
            "Production %": BLUE,
            "Warehouse %": ORANGE,
        },
        text="value",
    )
    figure.update_traces(texttemplate="%{text:.1f}", textposition="outside", marker_line_width=0)
    figure.update_yaxes(categoryorder="total ascending")
    return _apply_enterprise_layout(
        figure,
        title=title,
        height=430,
        xaxis_title="Percent load",
        yaxis_title="Order ID",
        showlegend=True,
    )


def budget_consumption_chart(
    *,
    capacity_budget_pct: float,
    warehouse_budget_pct: float,
    cumulative_selected_production_load_pct: float,
    cumulative_selected_warehouse_stress_pct: float,
) -> Any:
    frame = pd.DataFrame(
        {
            "Metric": ["Capacity budget", "Warehouse budget"],
            "Used": [
                cumulative_selected_production_load_pct,
                cumulative_selected_warehouse_stress_pct,
            ],
            "Budget": [capacity_budget_pct, warehouse_budget_pct],
        }
    )
    frame["Remaining"] = (frame["Budget"] - frame["Used"]).clip(lower=0.0)

    fig = go.Figure()
    fig.add_bar(
        y=frame["Metric"],
        x=frame["Used"],
        name="Used",
        orientation="h",
        marker_color=BLUE,
        text=[f"{value:.1f}" for value in frame["Used"]],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="#ffffff"),
    )
    fig.add_bar(
        y=frame["Metric"],
        x=frame["Remaining"],
        name="Remaining",
        orientation="h",
        marker_color="#cfdceb",
        text=[f"{value:.1f}" for value in frame["Remaining"]],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="#111111"),
    )
    fig.update_layout(barmode="stack")
    return _apply_enterprise_layout(
        fig,
        title="Budget Consumption vs Remaining Capacity",
        height=320,
        xaxis_title="Budget load",
        yaxis_title="",
        showlegend=True,
    )


def selection_frontier_chart(frame: pd.DataFrame, title: str = "Selection Frontier") -> Any:
    if frame.empty:
        frame = pd.DataFrame(
            {
                "order_id": [],
                "production_pct": [],
                "warehouse_pct": [],
            }
        )

    working = frame.copy().reset_index(drop=True)

    # Hỗ trợ cả tên cột cũ và mới
    production_col = "today_production_pct" if "today_production_pct" in working.columns else "production_pct"
    warehouse_col = (
        "warehouse_waiting_pressure_pct"
        if "warehouse_waiting_pressure_pct" in working.columns
        else "warehouse_pct"
    )

    # fallback để tránh crash
    if production_col not in working.columns:
        working[production_col] = 0.0
    if warehouse_col not in working.columns:
        working[warehouse_col] = 0.0

    working["selection_rank"] = range(1, len(working) + 1)
    working["cum_production"] = working[production_col].cumsum()
    working["cum_warehouse"] = working[warehouse_col].cumsum()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=working["selection_rank"],
            y=working["cum_production"],
            mode="lines+markers",
            name="Cumulative production load",
            line=dict(color=BLUE, width=3),
            fill="tozeroy",
            fillcolor="rgba(31, 78, 121, 0.08)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=working["selection_rank"],
            y=working["cum_warehouse"],
            mode="lines+markers",
            name="Cumulative warehouse stress",
            line=dict(color=ORANGE, width=3),
            fill="tozeroy",
            fillcolor="rgba(201, 111, 59, 0.08)",
        )
    )
    return _apply_enterprise_layout(
        fig,
        title=title,
        height=360,
        xaxis_title="Selected order rank",
        yaxis_title="Cumulative load",
        showlegend=True,
    )
