from __future__ import annotations

import colorsys
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from data_prep import safe_sort
from floorplan_routing import (
    build_factory_geometry,
    build_floor_plan_overlay,
    dom_positions,
    parse_svg_dom,
)

# ============================================================================
# Floor plan (Figma SVG) renderer — draws flow arrows on the floor plan — ONE diagram per hierarchy level.
# The number of levels is taken from cfg.hierarchy_levels (0..N).
# Area boxes are found first from the DOM (label text + containment depth);
# fallback mapping is attempted for unmatched ones using ID hints. The routing engine
# (door/wall aware, orthogonal, grid-based) is implemented in floorplan_routing.py.
# ============================================================================


@dataclass
class FloorPlanSettings:
    # --- fallback (ID-based) mapping
    area_hint: str = "area"                        # a common word in the ID of the area rectangle
    level_hints: tuple[tuple[str, ...], ...] = ()  # in the SAME order as cfg.hierarchy_levels
    # --- Flows will be drawn
    top_n_edges: int | None = None                 # None = all edges; int = most frequent N edges
    min_edge_count: int = 1
    self_loop_min_count: int = 1
    label_top_n: int = 40                          # The most frequent N edge labels to always display
    # --- scale and factory limit
    px_to_m: float = 0.06                          # 1 px = 6 cm in SVG Plan (Figma) coordinates
    wall_thickness: float = 8.0                    # Wall thickness (px)
    zones: dict = field(default_factory=dict)      # {name: {"border_id": str, "door_ids": [str, ...]}}
    # --- DOM area detection / style
    dom_min_size: float = 12.0                     # (px) Boxes smaller than this area are filtered out.
    dom_max_canvas_frac: float = 0.6               # The area of ​​boxes (background) larger than this proportion of the canvas is eliminated.
    ref_stat: str = "max"                          #  line/arrow size reference: max | p90 | median
    margin_px: float | None = None                 # box margin; None = automatic


def default_level_hints(label: str) -> tuple[str, ...]:
    """Derives SVG ID hints from the level label (e.g., 'Work Center'): 
    such as 'work center', 'workcenter', 'work_center', 'wc'."""
    words = re.findall(r"[A-Za-z0-9]+", label)
    cand = [label.lower(), "".join(words).lower(), "_".join(words).lower()]
    if len(words) > 1:
        cand.append("".join(w[0] for w in words).lower())
    out, seen = [], set()
    for c in cand:
        n = re.sub(r"[^a-z0-9]", "", c)
        if n and n not in seen:
            seen.add(n)
            out.append(c)
    return tuple(out)


def sanitize_svg(content: str) -> str:
    """Sanitizes the loaded SVG before embedding it in an iframe: script,
    foreignObject, on* event attributes and javascript: links."""
    content = re.sub(r"<script\b.*?</script\s*>", "", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"<script\b[^>]*/>", "", content, flags=re.IGNORECASE)
    content = re.sub(r"<foreignObject\b.*?</foreignObject\s*>", "", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"\son\w+\s*=\s*(\"[^\"]*\"|'[^']*')", "", content, flags=re.IGNORECASE)
    content = re.sub(r"(href\s*=\s*[\"'])\s*javascript:[^\"']*", r"\1#", content, flags=re.IGNORECASE)
    return content


# ----------------------------------------------------------------------------
# Level transitions ("From-To") matrix
# ----------------------------------------------------------------------------
def build_level_transition_matrix(ops: pd.DataFrame, cfg, code_col: str, name_col: str | None = None):
    """Sequential events with the same level values are counted once; only
    actual movements between different locations are counted (SLP 'From-To' table).
    Returns: (matrix_df, pairs_df[from,to,count(,from_name,to_name)], name_lookup)."""
    ts_col = cfg.ts_col
    cols = [cfg.case_id_col, code_col, ts_col] + ([name_col] if name_col else [])
    frame = ops[cols].dropna(subset=[code_col, ts_col]).copy()
    frame[code_col] = frame[code_col].astype(str)
    frame = safe_sort(frame, [cfg.case_id_col, ts_col])
    name_lookup = {}
    if name_col:
        name_lookup = frame.drop_duplicates(code_col).set_index(code_col)[name_col].astype(str).to_dict()
    frame["prev_code"] = frame.groupby(cfg.case_id_col, sort=False)[code_col].shift(1)
    moves = frame[frame["prev_code"].notna() & (frame["prev_code"] != frame[code_col])]
    if moves.empty:
        empty = pd.DataFrame(columns=["from", "to", "count"])
        return pd.DataFrame(), empty, name_lookup
    pairs = (moves.groupby(["prev_code", code_col]).size().rename("count").reset_index()
                  .rename(columns={"prev_code": "from", code_col: "to"}))
    pairs = pairs.iloc[np.argsort(-pairs["count"].to_numpy(), kind="stable")].reset_index(drop=True)
    codes = sorted(set(pairs["from"]) | set(pairs["to"]))
    matrix = pd.DataFrame(0, index=codes, columns=codes, dtype=int)
    for r in pairs.itertuples(index=False):
        matrix.loc[r[0], r[1]] = r[2]
    if name_lookup:
        pairs["from_name"] = pairs["from"].map(name_lookup)
        pairs["to_name"] = pairs["to"].map(name_lookup)
    return matrix, pairs, name_lookup


def build_level_transition_perf(ops: pd.DataFrame, cfg, code_col: str):
    """Average waiting time (in seconds) for transitions between levels.
    Waiting time = End of previous operation -> Start of next operation (if Start/Finish is available),
    otherwise the difference between events. Negative (overlapping) gaps are clipped to 0.
    Returns: (pairs_perf[from,to,mean_sec,min_sec,max_sec], {code: self-loop average time})"""
    ts_s = cfg.ts_col
    ts_e = cfg.end_timestamp_col if cfg.has_interval else None
    cols = [cfg.case_id_col, code_col, ts_s] + ([ts_e] if ts_e else [])
    fr = ops[cols].dropna(subset=[code_col, ts_s]).copy()
    fr[code_col] = fr[code_col].astype(str)
    fr = safe_sort(fr, [cfg.case_id_col, ts_s])
    g = fr.groupby(cfg.case_id_col, sort=False)
    fr["prev_code"] = g[code_col].shift(1)
    prev_ref = g[ts_s].shift(1)
    if ts_e:
        prev_ref = g[ts_e].shift(1).fillna(prev_ref)
    fr["_gap"] = (fr[ts_s] - prev_ref).dt.total_seconds().clip(lower=0)
    mv = fr[fr["prev_code"].notna() & fr["_gap"].notna()]
    cross = mv[mv["prev_code"] != mv[code_col]]
    same = mv[mv["prev_code"] == mv[code_col]]
    perf = (cross.groupby(["prev_code", code_col])["_gap"]
                 .agg(mean_sec="mean", min_sec="min", max_sec="max").reset_index()
                 .rename(columns={"prev_code": "from", code_col: "to"}))
    loop_perf = same.groupby(code_col)["_gap"].mean().to_dict()
    return perf, loop_perf


def compute_self_loop_counts(ops: pd.DataFrame, cfg, code_col: str) -> dict:
    """Events with the same code occurring consecutively within a case."""
    ts_col = cfg.ts_col
    frame = ops[[cfg.case_id_col, code_col, ts_col]].dropna(subset=[code_col, ts_col]).copy()
    frame[code_col] = frame[code_col].astype(str)
    frame = safe_sort(frame, [cfg.case_id_col, ts_col])
    frame["prev_code"] = frame.groupby(cfg.case_id_col, sort=False)[code_col].shift(1)
    loops = frame[frame["prev_code"].notna() & (frame["prev_code"] == frame[code_col])]
    return loops.groupby(code_col).size().to_dict()


# ----------------------------------------------------------------------------
# Fallback: SVG ID <-> code mapping (for codes not found via DOM tag)
# ----------------------------------------------------------------------------
def _norm(s) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def match_area_ids_for_level(svg_ids, hints, area_hint, codes):
    """Returns: ({code: svg_id}, unmatched_codes). First exact matches, then (for remaining
    codes) fuzzy matches — order dependency not considered."""
    area_n = _norm(area_hint)
    hints_n = sorted({_norm(h) for h in hints if _norm(h)}, key=len, reverse=True)
    candidates = [i for i in svg_ids if area_n and area_n in _norm(i)]
    hinted = [i for i in candidates if any(h in _norm(i) for h in hints_n)]
    pool = hinted if hinted else candidates
    norm_codes = {_norm(c): c for c in codes}

    def core_of(svg_id):
        core = _norm(svg_id).replace(area_n, "")
        for h in hints_n:
            core = core.replace(h, "")
        return core

    result, used_ids = {}, set()
    for svg_id in pool:                          # 1) exact match
        core = core_of(svg_id)
        if core and core in norm_codes and norm_codes[core] not in result:
            result[norm_codes[core]] = svg_id
            used_ids.add(svg_id)
    for svg_id in pool:                          # 2) fuzzy match (core substring)
        if svg_id in used_ids:
            continue
        core = core_of(svg_id)
        if not core:
            continue
        for cn, orig in norm_codes.items():
            if orig not in result and cn and (cn in core or core in cn):
                result[orig] = svg_id
                used_ids.add(svg_id)
                break
    missing = [c for c in codes if c not in result]
    return result, missing


# ----------------------------------------------------------------------------
# Color Assignment
# ----------------------------------------------------------------------------
def assign_node_colors(codes, saturation=0.65, value=0.85):
    codes = sorted(set(codes))
    n = max(1, len(codes))
    colors = {}
    for i, code in enumerate(codes):
        r, g, b = colorsys.hsv_to_rgb(i / n, saturation, value)
        colors[code] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    return colors


def recolor_floor_plan_areas(svg_content, id_map, node_colors, fill_opacity=0.30, stroke_width=2.5):
    """Area kutularını node rengine göre boyar (var olan fill/stroke silinir)."""
    content = svg_content
    for code, svg_id in id_map.items():
        color = node_colors.get(code)
        if not color or not svg_id:        # DOM eşlemesinde id'siz kutu olabilir -> boyanamaz
            continue
        pattern = re.compile(rf'(<[a-zA-Z]+\b[^>]*\bid="{re.escape(svg_id)}"[^>]*?)(/?>)')
        m = pattern.search(content)
        if not m:
            continue
        tag_open, tag_close = m.group(1), m.group(2)
        tag_open = re.sub(r'\sfill="[^"]*"', "", tag_open)
        tag_open = re.sub(r'(\sstyle=")([^"]*)(")',
                          lambda mm: mm.group(1) + re.sub(r'fill\s*:\s*[^;]+;?', '', mm.group(2)) + mm.group(3),
                          tag_open)
        tag_open = re.sub(r'\sstroke="[^"]*"', "", tag_open)
        tag_open = re.sub(r'\sstroke-width="[^"]*"', "", tag_open)
        new_open = (f'{tag_open} fill="{color}" fill-opacity="{fill_opacity}" '
                    f'stroke="{color}" stroke-width="{stroke_width}" stroke-opacity="0.9"')
        content = content[:m.start()] + new_open + tag_close + content[m.end():]
    return content


# ----------------------------------------------------------------------------
# SINGLE ENTRY POINT — floor plan diagram for each hierarchy level
# ----------------------------------------------------------------------------
def build_floor_plan_views(svg_content: str, df: pd.DataFrame, cfg, fp: FloorPlanSettings, 
                           mode: str = "freq") -> dict:
    """Returns: {"levels": [{label, level_idx, svg, info, centers, positions, unmatched}],
    "geo": factory wall/door geometry, "px_to_m": float, "warnings": [...], "log": [...]}"""
    svg_content = sanitize_svg(svg_content)
    nodes, id_bboxes, _canvas = parse_svg_dom(svg_content, fp.dom_min_size, fp.dom_max_canvas_frac)
    ops = df
    warnings, log, levels = [], [], []

    geo, geo_warn = build_factory_geometry(id_bboxes, fp.zones, fp.wall_thickness)
    warnings += geo_warn
    for name, g in geo.items():
        log.append(f"Zone {name}: doors={[d['id'] for d in g['doors']]} | wall segments={len(g['walls'])}")

    level_codes = [ops[lvl.code_col].astype(str).unique().tolist() for lvl in cfg.hierarchy_levels]

    def _id_fallback(lvl_idx, missing):
        label = cfg.hierarchy_levels[lvl_idx].label
        hints = fp.level_hints[lvl_idx] if lvl_idx < len(fp.level_hints) else default_level_hints(label)
        id_map, _ = match_area_ids_for_level(list(id_bboxes), hints, fp.area_hint, missing)
        return ({c: id_bboxes[s] for c, s in id_map.items() if s in id_bboxes}, dict(id_map))

    pos_by_level, ids_by_level, missing_by_level = dom_positions(nodes, level_codes, _id_fallback, log)

    for i, lvl in enumerate(cfg.hierarchy_levels):
        positions, id_map, missing = pos_by_level[i], ids_by_level[i], missing_by_level[i]
        if not positions:
            warnings.append(f"{lvl.label}: No field matched in the SVG (label text and ID hints were tried; "
                            f"area keyword: '{fp.area_hint}').")
            continue
        if missing:
            warnings.append(f"{lvl.label}: {len(missing)} codes did not match in the SVG: "
                            f"{missing[:10]}{' ...' if len(missing) > 10 else ''}")
        no_id = [c for c in positions if not id_map.get(c)]
        if no_id:
            warnings.append(f"{lvl.label}: {len(no_id)} The area does not have an SVG ID; these areas could not be colored. "
                            f"(Flows are mapped out again.).")
        _, pairs, names = build_level_transition_matrix(ops, cfg, lvl.code_col, lvl.name_col)
        perf_df, loop_perf = (build_level_transition_perf(ops, cfg, lvl.code_col)
                              if mode == "perf" else (None, None))
        colors = assign_node_colors(positions.keys())
        # do not paint in perf mode: edge colors will carry duration scalar
        recolored = recolor_floor_plan_areas(svg_content, id_map, colors) if mode == "freq" else svg_content
        containers = {j: pos_by_level[j] for j in range(i)}
        suffix = " — Average Waiting Time" if mode == "perf" else ""
        try:
            svg, info = build_floor_plan_overlay(
                recolored, positions, pairs,
                title=f"{lvl.label} Flow (On Floor Plan){suffix}",
                node_colors=colors,
                self_loop_counts=compute_self_loop_counts(ops, cfg, lvl.code_col),
                factory_geo=geo, container_positions=containers, node_names=names,
                wall_t=fp.wall_thickness, min_count=fp.min_edge_count, top_n=fp.top_n_edges,
                label_top_n=fp.label_top_n, margin_px=fp.margin_px, ref_stat=fp.ref_stat,
                self_loop_min_count=fp.self_loop_min_count,
                mode=mode, perf_df=perf_df, loop_perf=loop_perf)
        except ValueError as e:
            warnings.append(f"{lvl.label}: {e}")
            continue
        levels.append({
            "label": lvl.label,
            "level_idx": i,
            "svg": svg,
            "info": info,
            "centers": {c: (x + w / 2, y + h / 2) for c, (x, y, w, h) in positions.items()},
            "positions": positions,
            "unmatched": missing,
            "mode": mode,
        })
    return {"levels": levels, "geo": geo, "px_to_m": fp.px_to_m, "warnings": warnings, "log": log}
