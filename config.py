from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

# ============================================================================
# SINGLE SOURCE OF TRUTH for the column mapping. Every function takes `cfg`.
# frozen=True protects against accidental mutation.
# ============================================================================

@dataclass(frozen=True)
class HierLevel:
    code_col: str
    name_col: str | None
    label: str


@dataclass(frozen=True)
class EventLogConfig:
    case_id_col: str
    activity_code_col: str
    activity_name_col: str | None
    timestamp_col: str | None          # point-event log if not None
    start_timestamp_col: str | None    # interval-event log (Start/Finish) if not None
    end_timestamp_col: str | None
    hierarchy_levels: tuple[HierLevel, ...]

    @property
    def has_interval(self) -> bool:
        """True if each row has both a start and an end time (durations can be computed)."""
        return self.start_timestamp_col is not None and self.end_timestamp_col is not None

    @property
    def ts_col(self) -> str | None:
        """The timestamp used to order operations inside an order."""
        return self.timestamp_col if self.timestamp_col else self.start_timestamp_col

    @property
    def hier_code_cols(self) -> tuple[str, ...]:
        return tuple(lvl.code_col for lvl in self.hierarchy_levels)
    


# ============================================================================
# Sidebar: column mapping UI (the Streamlit counterpart of the notebook's
# "Column mapping" settings cell).
# ============================================================================
_DEFAULT_LABELS = ("Workcenter", "Workstation")


def _pick(cols, *names, optional=False):
    """Index of the first candidate name found in cols (to pre-select sensible defaults)."""
    options = (["(none)"] if optional else []) + list(cols)
    for n in names:
        if n in options:
            return options.index(n)
    return 0


def build_config_ui(df: pd.DataFrame) -> EventLogConfig:
    cols = list(df.columns)

    st.sidebar.subheader("2. Columns")
    case_id_col = st.sidebar.selectbox("Order / Case ID", cols, index=_pick(cols, "OrderId"))
    activity_code_col = st.sidebar.selectbox("Operation code", cols, index=_pick(cols, "OperationCode"))
    activity_name_col = st.sidebar.selectbox(
        "Operation name (optional)", ["(none)"] + cols,
        index=_pick(cols, "OperationName", optional=True))

    ts_mode = st.sidebar.radio("Time model", ["Interval (Start/Finish)", "Point (single timestamp)"])
    if ts_mode.startswith("Interval"):
        timestamp_col = None
        start_col = st.sidebar.selectbox("Start time", cols, index=_pick(cols, "StartTime"))
        end_col = st.sidebar.selectbox("Finish time", cols, index=_pick(cols, "FinishTime"))
    else:
        timestamp_col = st.sidebar.selectbox("Timestamp", cols, index=_pick(cols, "timestamp"))
        start_col = end_col = None

    st.sidebar.subheader("3. Hierarchy (outer to inner)")
    n_levels = int(st.sidebar.number_input("Number of levels", min_value=1, max_value=5, value=2))
    code_defaults = (("WorkCenterCode",), ("WorkStationCode",))
    name_defaults = (("WorkCenterName",), ("WorkStationName",))
    levels: list[HierLevel] = []
    for i in range(n_levels):
        st.sidebar.markdown(f"**Level {i}**")
        code_col = st.sidebar.selectbox(f"Code column #{i}", cols, key=f"lvl_code_{i}",
                                        index=_pick(cols, *(code_defaults[i] if i < 2 else ())))
        name_col = st.sidebar.selectbox(f"Name column #{i} (optional)", ["(none)"] + cols, key=f"lvl_name_{i}",
                                        index=_pick(cols, *(name_defaults[i] if i < 2 else ()), optional=True))
        label = st.sidebar.text_input(f"Label #{i}", value=_DEFAULT_LABELS[i] if i < 2 else f"Level {i}", key=f"lvl_label_{i}")
        levels.append(HierLevel(code_col, None if name_col == "(none)" else name_col, label))

    return EventLogConfig(
        case_id_col=case_id_col,
        activity_code_col=activity_code_col,
        activity_name_col=None if activity_name_col == "(none)" else activity_name_col,
        timestamp_col=timestamp_col,
        start_timestamp_col=start_col,
        end_timestamp_col=end_col,
        hierarchy_levels=tuple(levels),
    )
