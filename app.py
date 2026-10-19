from __future__ import annotations

import pandas as pd
import streamlit as st
from analysis import mismatch_table, run_analysis, share_change
from config import build_config_ui
from data_prep import load_event_log, prepare_dataframe
from floorplan import FloorPlanSettings, build_floor_plan_views, default_level_hints

st.set_page_config(page_title="Floor Plan Process Map", layout="wide")
st.title("Floor Plan Process Map")


def _show_svg(svg: str, height: int = 800):
    """Show SVG in an iframe (so zoom/pan/hover JS works)."""
    st.iframe("<html><body style='margin:0;background:#fff'>" + svg + "</body></html>", height=height)


def _arrow_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert mixed-type (e.g., "All" + years) object columns to strings; prevents Arrow serialization warnings."""
    out = frame.copy()
    for c in out.columns:
        if out[c].dtype == object and out[c].map(type).nunique() > 1:
            out[c] = out[c].astype(str)
    return out


def _table(frame: pd.DataFrame, file_name: str):
    st.download_button(f"Download {file_name} CSV", frame.to_csv(index=False), file_name=file_name.replace(" ", "_") + ".csv", mime="text/csv")
    st.dataframe(_arrow_safe(frame), width="stretch")

# ============================================================================
# 1) FLES
# ============================================================================
st.sidebar.subheader("1. Files")

if "log" not in st.session_state:
    st.session_state.log = None
if "svg" not in st.session_state:
    st.session_state.svg = None

log_file = st.sidebar.file_uploader("Event Log CSV or Excel File (required)", type=["csv", "xlsx", "xls"])
svg_file = st.sidebar.file_uploader("Process Floor Plan SVG (required)", type=["svg"], key="floorplan_svg")

if log_file is not None:
    try:
        st.session_state.log = load_event_log(log_file.getvalue(), log_file.name)
        st.sidebar.success("Event Log File\n Successfully Uploaded!")
    except (ValueError, TypeError, OSError, UnicodeError) as e:
        st.error(f"File could not be read: {e}")
        st.stop()

if svg_file is not None:
    try:
        st.session_state.svg = svg_file.getvalue().decode("utf-8", errors="replace")
        st.sidebar.success("Process Floor Plan SVG File\n Successfully Uploaded!")
    except (ValueError, TypeError, OSError, UnicodeError) as e:
        st.error(f"File could not be read: {e}")
        st.stop()

if st.sidebar.button("Load Sample Data", type="primary"):
    try:
        st.session_state.log = pd.read_excel("KNS.xlsx", engine="openpyxl", dtype=str)
        with open("KNS.svg", "r", encoding="utf-8", errors="replace") as f:
            st.session_state.svg = f.read()
        st.sidebar.info("Sample data loaded.")
    except (ValueError, TypeError, OSError, UnicodeError) as e:
        st.error(f"Could not read sample data: {e}")

if st.session_state.log is None or st.session_state.svg is None:
    missing = [n for n, f in (("Event Log", st.session_state.log), ("Process Floor Plan", st.session_state.svg)) if f is None]
    st.info("Upload the " + " and the ".join(missing) + " from the left sidebar to begin.")
    st.stop()

# ============================================================================
# 2) SETTINGS
# ============================================================================
raw_df = st.session_state.log.copy()
svg_text = st.session_state.svg

cfg = build_config_ui(raw_df) # type: ignore

st.sidebar.subheader("4. Floor plan")
fp_scale = st.sidebar.number_input("Scale: 1 SVG px = ? meters", 
                                   min_value=0.001, value=0.06,
                                   step=0.005, format="%.3f")
fp_wall = st.sidebar.number_input("Wall thickness (px)", 
                                  min_value=1.0, value=8.0, step=1.0)
st.sidebar.caption("Zones: border rectangle + door IDs. Walls are crossed only through doors. "
                   "Leave the table empty if there are no zones.")
zones_df = st.sidebar.data_editor(
    pd.DataFrame([
        {"Area": "Assembly", "Border id": "AssemblyBorder", "Door ids (comma-separated)": "AssemblyDoor_1, AssemblyDoor_2"},
        {"Area": "Mechanic", "Border id": "MechanicBorder", "Door ids (comma-separated)": "MechanicDoor_1"},
    ]),
    num_rows="dynamic", key="fp_zones", width="stretch",
)
zones = {}
for _, r in zones_df.iterrows():
    name, border = str(r.get("Area") or "").strip(), str(r.get("Border id") or "").strip()
    if name and border and name.lower() != "nan" and border.lower() != "nan":
        doors_raw = str(r.get("Door ids (comma-separated)") or "")
        zones[name] = {"border_id": border,
                       "door_ids": [d.strip() for d in doors_raw.split(",")
                                    if d.strip() and d.strip().lower() != "nan"]}

fp_top_n = st.sidebar.number_input("Top N most frequent flows (0 = all)", min_value=0, value=0)
fp_min_count = st.sidebar.number_input("Minimum flow count", min_value=1, value=1)
with st.sidebar.expander("Floor plan — advanced"):
    fp_label_n = st.number_input("Always label the top N flows", min_value=0, value=40)
    fp_ref = st.selectbox("Line/arrow size reference", ["max", "p90", "median"], index=0)
    fp_min_size = st.number_input("Minimum box size to count as an area (px)", min_value=1.0, value=12.0)
    area_hint = st.text_input("Fallback — area id keyword", value="area")
    hints = []
    for i, lvl in enumerate(cfg.hierarchy_levels):
        raw = st.text_input(f"Fallback — '{lvl.label}' id hints (comma-separated)",
                            value=", ".join(default_level_hints(lvl.label)), key=f"fp_hint_{i}_{lvl.label}")
        hints.append(tuple(h.strip() for h in raw.split(",") if h.strip()))

fp_settings = FloorPlanSettings(
    area_hint=area_hint, level_hints=tuple(hints),
    top_n_edges=int(fp_top_n) or None, min_edge_count=int(fp_min_count),
    label_top_n=int(fp_label_n), px_to_m=float(fp_scale), wall_thickness=float(fp_wall),
    zones=zones, dom_min_size=float(fp_min_size), ref_stat=fp_ref,
)

# ============================================================================
# 3) RUN
# ============================================================================
if st.sidebar.button("Generate", type="primary"):
    res: dict = {}
    try:
        df = prepare_dataframe(raw_df, cfg)
    except ValueError as e:
        st.error(f"Data preparation error: {e}")
        st.stop()

    floors = {}
    for fmode, flabel in (("freq", "Frequency"), ("perf", "Performance")):
        try:
            with st.spinner(f"Calculating floor plan ({flabel}) routes..."):
                floors[fmode] = build_floor_plan_views(svg_text, df, cfg, fp_settings, mode=fmode)
        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as e:
            st.warning(f"Floor plan ({flabel}) could not be generated: {e}")
    res["floors"] = floors

    try:
        res["analysis"] = run_analysis(df, cfg)
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as e:
        st.warning(f"Analysis tables could not be generated: {e}")
    st.session_state["res"] = res


# ============================================================================
# 4) VİEW
# ============================================================================
def _show_floor(floors: dict):
    ref = floors.get("freq") or floors.get("perf")
    for w in ref["warnings"]: # type: ignore
        st.warning(w)
    if not ref["levels"]: # type: ignore
        st.info("No floor plan diagram could be generated.")
        return
    lv_tabs = st.tabs([lv["label"] for lv in ref["levels"]]) # type: ignore
    for lt, lv0 in zip(lv_tabs, ref["levels"]): # type: ignore
        with lt:
            mode_tabs = st.tabs(["📊 Frequency", "⏱️ Performance (avg. waiting)"])
            for mt, fmode in zip(mode_tabs, ("freq", "perf")):
                with mt:
                    fl_mode = floors.get(fmode)
                    lv = next((x for x in (fl_mode or {}).get("levels", [])
                               if x["level_idx"] == lv0["level_idx"]), None)
                    if lv is None:
                        st.info("This view could not be generated for this level.")
                        continue
                    inf = lv["info"]
                    st.caption(f"{inf['n_edges']} flows + {inf['n_loops']} self-loops | "
                               f"{inf['door_cross']} door crossings | {inf['failed']} unresolved | "
                               f"box violations {inf['leaf_viol']}, wall violations {inf['wall_viol']} | "
                               f"{inf['seconds']} s")
                    st.download_button("Download SVG", lv["svg"],
                                       file_name=f"floorplan_{fmode}_{lv['label'].replace(' ', '_')}.svg",
                                       mime="image/svg+xml", 
                                       key=f"dl_floorplan_{fmode}_{lv['level_idx']}"
                                       )
                    _show_svg(lv["svg"])


def _show_analysis(an: dict, floors: dict):
    floor = floors.get("freq") or floors.get("perf")
    st.subheader("Yearly Summary")
    _table(an["yearly"], "floorplan Yearly Summary")
    if not an["transfer"].empty:
        st.subheader("Transfer Waiting Time — by separation level")
        _table(an["transfer"], "floorplan Transfer Waiting Time by separation level")

    years = [int(y) for y in an["yearly"].index]
    floor_by_idx = {lv["level_idx"]: lv for lv in (floor or {}).get("levels", [])}
    tabs = st.tabs([lv["label"] for lv in an["levels"]])
    for tab, lv in zip(tabs, an["levels"]):
        with tab:
            i = lv["level_idx"]
            st.markdown("**Ordered by operation events (descending)**")
            _table(lv["ranking"], "floorplan " + lv["label"] + " Ordered by Operation Events (descending)")
            if "Avg op. duration (sec)" in lv["ranking"].columns:
                slow = lv["ranking"][lv["ranking"]["Operation events"] >= 30]
                st.markdown("**The Slowest**")
                _table(slow.sort_values("Avg op. duration (sec)", ascending=False),
                       "floorplan " + lv["label"] + " The Slowest")
            if len(years) >= 2:
                st.markdown(f"**Share change ({years[0]} → {years[-1]})**")
                _table(share_change(lv["year_share"], years[0], years[-1]).head(20), 
                       "floorplan " + lv["label"] + " Share Change")
            st.markdown("**SLP: which pairs should be close?**")
            _table(lv["slp"].head(30), "floorplan " + lv["label"] + " SLP which pairs should be close")
            fl = floor_by_idx.get(i)
            if fl is not None:
                mm = mismatch_table(lv["slp"], fl["positions"], floor["geo"], floor["px_to_m"]) # type: ignore
                if not mm.empty:
                    st.markdown("**Current layout: Flow × Distance** (high flow + far apart = relocation candidate)")
                    _table(mm.head(30), "floorplan " + lv["label"] + " Current layout Flow × Distance")


res = st.session_state.get("res") # type: ignore
if not res:
    st.info("Map the columns and click **Generate**.")
    st.stop()

labels = []
if res.get("floors"):
    labels.append("🗺️ Floor plan")
if res.get("analysis"):
    labels.append("📈 Analysis")
if not labels:
    st.stop()
tabs = dict(zip(labels, st.tabs(labels)))
if "🗺️ Floor plan" in tabs:
    with tabs["🗺️ Floor plan"]:
        _show_floor(res["floors"])
if "📈 Analysis" in tabs:
    with tabs["📈 Analysis"]:
        _show_analysis(res["analysis"], res.get("floors") or {})
