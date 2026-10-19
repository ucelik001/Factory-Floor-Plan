from __future__ import annotations

import numpy as np
import pandas as pd
from config import EventLogConfig
from data_prep import format_duration, safe_sort
from floorplan import build_level_transition_matrix
from floorplan_routing import layout_distance

SLP_LABELS = ("A - Absolutely necessary", "E - Especially important", "I - Important",
              "O - Ordinary closeness OK", "U - Unimportant")


def _duration_sec(ops: pd.DataFrame, cfg) -> pd.Series | None:
    if not cfg.has_interval:
        return None
    return (ops[cfg.end_timestamp_col] - ops[cfg.start_timestamp_col]).dt.total_seconds()


def level_ranking(ops: pd.DataFrame, cfg, level_idx: int, pairs: pd.DataFrame, names: dict) -> pd.DataFrame:
    lvl = cfg.hierarchy_levels[level_idx]
    code = lvl.code_col
    d = ops.copy()
    d[code] = d[code].astype(str)
    dur = _duration_sec(d, cfg)
    if dur is not None:
        d["_dur"] = dur
    g = d.groupby(code, observed=True)
    out = pd.DataFrame({
        "Operation events": g.size(),
        "Distinct orders": g[cfg.case_id_col].nunique(),
        "Distinct operations": g[cfg.activity_code_col].nunique(),
    })
    if dur is not None:
        out["Avg op. duration (sec)"] = g["_dur"].mean()
        out["Avg op. duration"] = out["Avg op. duration (sec)"].map(
            lambda v: format_duration(v) if pd.notna(v) else "-")
    partners = {}
    for a, b in zip(pairs["from"], pairs["to"]):
        partners.setdefault(a, set()).add(b)
        partners.setdefault(b, set()).add(a)
    out[f"Connected to # other {lvl.label}s"] = [len(partners.get(c, ())) for c in out.index]
    out.insert(0, "Name", [names.get(c, "") for c in out.index])
    if level_idx > 0:
        parent = cfg.hierarchy_levels[level_idx - 1].code_col
        d[parent] = d[parent].astype(str)
        out.insert(1, f"Parent {cfg.hierarchy_levels[level_idx - 1].label}",
                   d.groupby(code, observed=True)[parent].agg(lambda s: s.mode().iat[0]))
    out.index.name = lvl.label
    out = out.iloc[np.argsort(-out["Operation events"].to_numpy(), kind="stable")]
    return out.reset_index()


def transfer_time_summary(ops: pd.DataFrame, cfg) -> pd.DataFrame:
    """The gap between consecutive operations (the end of the previous -> the start of the next),
    grouped by the level at which they diverge:
    'Same <inner level>' ... 'Different <outer level>'. The outermost divergence
    (e.g., different work center) is a strong layout signal."""
    levels = cfg.hierarchy_levels
    if not cfg.has_interval or not levels:
        return pd.DataFrame()
    code_cols = list(cfg.hier_code_cols)
    cols = [cfg.case_id_col, cfg.start_timestamp_col, cfg.end_timestamp_col] + code_cols
    seq = ops[cols].dropna(subset=[cfg.start_timestamp_col, cfg.end_timestamp_col]).copy()
    for c in code_cols:
        seq[c] = seq[c].astype(str)
    seq = safe_sort(seq, [cfg.case_id_col, cfg.start_timestamp_col])
    grp = seq.groupby(cfg.case_id_col, sort=False)
    nxt_start = grp[cfg.start_timestamp_col].shift(-1)
    valid = nxt_start.notna().to_numpy().copy()
    gap = (nxt_start - seq[cfg.end_timestamp_col]).dt.total_seconds().to_numpy()
    valid = valid & (gap >= 0)
    n = len(levels)
    first_diff = np.full(len(seq), n)
    for k in reversed(range(n)):
        nxt_code = grp[code_cols[k]].shift(-1).to_numpy()
        first_diff = np.where(seq[code_cols[k]].to_numpy() != nxt_code, k, first_diff)
    first_diff, gap = first_diff[valid], gap[valid]

    def bucket(k):
        if k == n:
            return f"Same {levels[-1].label}"
        if k == 0:
            return f"Different {levels[0].label}"
        return f"Same {levels[k - 1].label}, different {levels[k].label}"

    tmp = pd.DataFrame({"k": first_diff, "gap": gap})
    agg = tmp.groupby("k")["gap"].agg(["count", "mean", "median"])
    agg.columns = ["Moves", "Mean", "Median"]
    res = agg.copy()
    res.insert(0, "Transfer type", [bucket(k) for k in res.index])
    res["Avg wait"] = res["Mean"].map(format_duration)
    res["Median wait (sec)"] = res["Median"]
    return res.drop(columns=["Mean", "Median"]).reset_index(drop=True)


def yearly_overview(ops: pd.DataFrame, cfg) -> pd.DataFrame:
    ts = cfg.ts_col
    d = ops.dropna(subset=[ts]).copy()
    d["year"] = d[ts].dt.year
    agg = {"Operation events": (cfg.case_id_col, "size"),
           "Distinct orders": (cfg.case_id_col, "nunique")}
    for lvl in cfg.hierarchy_levels:
        agg[f"Distinct {lvl.label}"] = (lvl.code_col, "nunique")
    out = d.groupby("year").agg(**agg)
    dur = _duration_sec(d, cfg)
    if dur is not None:
        d["_dur"] = dur
        out["Avg. op duration"] = d.groupby("year")["_dur"].mean().map(format_duration)
    return out


def level_year_share(ops: pd.DataFrame, cfg, level_idx: int) -> pd.DataFrame:
    """Row: code, Column: year, Value: total activity share in that year."""
    code = cfg.hierarchy_levels[level_idx].code_col
    d = ops.dropna(subset=[cfg.ts_col]).copy()
    d["year"] = d[cfg.ts_col].dt.year
    ct = pd.crosstab(d[code].astype(str), d["year"])
    return ct.div(ct.sum(axis=0), axis=1)


def share_change(year_share: pd.DataFrame, year_from: int, year_to: int) -> pd.DataFrame:
    a = year_share[year_from] if year_from in year_share.columns else 0.0
    b = year_share[year_to] if year_to in year_share.columns else 0.0
    ya, yb = str(year_from), str(year_to)
    out = pd.DataFrame({ya: a, yb: b}).fillna(0.0)
    out["Change (pp)"] = (out[yb] - out[ya]) * 100
    return out.iloc[np.argsort(-out["Change (pp)"].to_numpy(), kind="stable")].round(3)


def slp_table(pairs: pd.DataFrame, names: dict) -> pd.DataFrame:
    """A->B and B->A flows are combined; based on the distribution of their own level,
    they are rated A/E/I/O/U according to quantiles (0.5/0.75/0.9/0.97).
    (In the notebook, WS rating was incorrectly using WC quantiles;
    here each level uses its own quantiles.)"""
    cols: list[str] = ["A", "A name", "B", "B name", "Combined flow", "SLP rating"]
    if pairs.empty:
        return pd.DataFrame(columns=cols)
    f, t = pairs["from"].to_numpy(), pairs["to"].to_numpy()
    lo, hi = np.where(f < t, f, t), np.where(f < t, t, f)
    sym = (pd.DataFrame({"A": lo, "B": hi, "Combined flow": pairs["count"].to_numpy()})
             .groupby(["A", "B"], as_index=False)["Combined flow"].sum())
    sym = sym.iloc[np.argsort(-sym["Combined flow"].to_numpy(), kind="stable")].reset_index(drop=True)
    v = sym["Combined flow"].to_numpy()
    q = np.quantile(v, [0.5, 0.75, 0.9, 0.97])
    sym["SLP rating"] = np.select([v >= q[3], v >= q[2], v >= q[1], v >= q[0]],
                                  list(SLP_LABELS[:4]), default=SLP_LABELS[4])
    sym["A name"] = sym["A"].map(names)
    sym["B name"] = sym["B"].map(names)
    return pd.DataFrame(sym.loc[:, cols].copy())


def mismatch_table(sym: pd.DataFrame, positions: dict, geo: dict | None = None,
                   px_to_m: float = 1.0) -> pd.DataFrame:
    """Based on the REAL locations in the floor plan: high flow x distant distance =
    the most valuable pairs to relocate. Distance is in meters (px_to_m scale);
    'Walking via doors' measures factory walls only through doors (vertical-horizontal)."""
    if sym.empty or not positions:
        return pd.DataFrame()
    rows = []
    for r in sym.itertuples(index=False):
        a, b = str(r.A), str(r.B)
        if a not in positions or b not in positions:
            continue
        eu_m, walk_m, za, zb = layout_distance(a, b, positions, geo or {}, px_to_m)
        rows.append({
            "A": a, "A name": r[1], "B": b, "B name": r[3],
            "Zone A": za or "outside", "Zone B": zb or "outside",
            "Combined flow": r[4], "SLP rating": r[5],
            "Straight-line (m)": round(eu_m, 1),
            "Walking via doors (m)": round(walk_m, 1),
            "Flow x Walking distance (m)": round(r[4] * walk_m, 1),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    order = np.argsort(-out["Flow x Walking distance (m)"].to_numpy(), kind="stable")
    return out.iloc[order].reset_index(drop=True)


# ----------------------------------------------------------------------------
# ONLY ONE ENTRY POINT
# ----------------------------------------------------------------------------
def run_analysis(df: pd.DataFrame, cfg: EventLogConfig) -> dict:
    levels = []
    for i, lvl in enumerate(cfg.hierarchy_levels):
        _, pairs, names = build_level_transition_matrix(df, cfg, lvl.code_col, lvl.name_col)
        levels.append({
            "label": lvl.label,
            "level_idx": i,
            "pairs": pairs,
            "ranking": level_ranking(df, cfg, i, pairs, names),
            "year_share": level_year_share(df, cfg, i),
            "slp": slp_table(pairs, names),
        })
    return {
        "transfer": transfer_time_summary(df, cfg),
        "yearly": yearly_overview(df, cfg),
        "levels": levels,
    }
