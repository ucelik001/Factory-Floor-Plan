from __future__ import annotations

import heapq
import html
import itertools
import math
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# Pure helpers: no function here reads cfg/global state.


def _segment_intersects_rect(p0, p1, rect, margin=3.0):
    x0, y0 = p0
    x1, y1 = p1
    rx, ry, rw, rh = rect
    xmin, ymin = rx - margin, ry - margin
    xmax, ymax = rx + rw + margin, ry + rh + margin
    dx, dy = x1 - x0, y1 - y0
    tmin, tmax = 0.0, 1.0
    for p, q in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)):
        if abs(p) < 1e-9:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            if t > tmax:
                return False
            tmin = max(tmin, t)
        else:
            if t < tmin:
                return False
            tmax = min(tmax, t)
    return tmin <= tmax


def manhattan_fallback(exit_a, exit_b):
    _, y0 = exit_a
    x1, _ = exit_b
    mid = (x1, y0)
    if mid == exit_a or mid == exit_b:
        return [exit_a, exit_b]
    return [exit_a, mid, exit_b]


def fmt_dur(sec):
    sec = abs(float(sec))
    if sec < 60:
        return f"{sec:.0f} sec"
    if sec < 3600:
        return f"{sec / 60:.0f} min"
    if sec < 86400:
        return f"{sec / 3600:.1f} hr"
    return f"{sec / 86400:.1f} day"


def duration_to_color(value, vmin, vmax):
    if vmax <= vmin:
        t = 0.5
    else:
        t = min(max((value - vmin) / (vmax - vmin), 0.0), 1.0)
    green, yellow, red = (46, 204, 113), (241, 196, 15), (231, 76, 60)
    if t < 0.5:
        t2 = t / 0.5
        r = int(green[0] + (yellow[0] - green[0]) * t2)
        g = int(green[1] + (yellow[1] - green[1]) * t2)
        b = int(green[2] + (yellow[2] - green[2]) * t2)
    else:
        t2 = (t - 0.5) / 0.5
        r = int(yellow[0] + (red[0] - yellow[0]) * t2)
        g = int(yellow[1] + (red[1] - yellow[1]) * t2)
        b = int(yellow[2] + (red[2] - yellow[2]) * t2)
    return f"#{r:02x}{g:02x}{b:02x}"


# ============================================================================
# Floor plan engine
# Wall thickness, scale, zone definitions, etc., are all passed as parameters;
# diagnostic outputs are written to the `log` list instead of using `print`.
# Safe for use in multi-user environments.
# ============================================================================

EPS = 1e-6
BEND_PEN = 60.0                        # each 90° turn = 60 px length penalty
DOOR_BALANCE_PX = 0.0                  # >0: door usage penalty (distributes congestion to doors)
_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIRV = {"E": (1, 0), "W": (-1, 0), "S": (0, 1), "N": (0, -1)}   # SVG: y down, x right
_DNAMES = "EWSN"                       # indices: E0 W1 S2 N3  (reverse direction = idx ^ 1)
_NORMAL = {"N": (0, -1), "S": (0, 1), "W": (-1, 0), "E": (1, 0)}
_VEC2DIR = {(0, -1): "N", (0, 1): "S", (-1, 0): "W", (1, 0): "E"}
_NUM = r'-?\d*\.?\d+(?:[eE][-+]?\d+)?'

# ============================================================================
# 1) SVG DOM: transform chain, area boxes, labels, id -> bbox
# ============================================================================
def _local(tag):
    return tag.split("}")[-1] if isinstance(tag, str) else ""


def _parse_transform(tstr):
    """Turns 3x3 matrix into a 3x3 affine matrix (rotate/translate/scale/matrix)."""
    M = np.eye(3)
    for name, args in re.findall(r'(\w+)\s*\(([^)]*)\)', tstr or ""):
        a = [float(v) for v in re.findall(_NUM, args)]
        T = np.eye(3)
        if name == "translate" and a:
            T[0, 2], T[1, 2] = a[0], (a[1] if len(a) > 1 else 0.0)
        elif name == "scale" and a:
            T[0, 0], T[1, 1] = a[0], (a[1] if len(a) > 1 else a[0])
        elif name == "rotate" and a:
            r = math.radians(a[0])
            c, s = math.cos(r), math.sin(r)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
            if len(a) >= 3:
                T1 = np.eye(3)
                T1[0, 2], T1[1, 2] = a[1], a[2]
                T2 = np.eye(3)
                T2[0, 2], T2[1, 2] = -a[1], -a[2]
                T = T1 @ R @ T2
            else:
                T = R
        elif name == "matrix" and len(a) == 6:
            T = np.array([[a[0], a[2], a[4]], [a[1], a[3], a[5]], [0, 0, 1.0]])
        M = M @ T
    return M


def _style_fill(el):
    f = el.get("fill")
    if f:
        return f.strip().lower()
    m = re.search(r"fill\s*:\s*([^;]+)", el.get("style", ""))
    return m.group(1).strip().lower() if m else ""


def _raw_bbox(el):
    t = _local(el.tag)
    if t == "rect":
        try:
            return (float(el.get("x", 0)), float(el.get("y", 0)),
                    float(el.get("width")), float(el.get("height")))
        except (TypeError, ValueError):
            return None
    if t == "line":
        try:
            x1, y1, x2, y2 = (float(el.get(k, 0)) for k in ("x1", "y1", "x2", "y2"))
        except (TypeError, ValueError):
            return None
        return (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
    src = el.get("points") if t in ("polygon", "polyline") else el.get("d") if t == "path" else None
    if not src:
        return None
    nums = [float(v) for v in re.findall(_NUM, src)]
    xs, ys = nums[0::2], nums[1::2]
    n = min(len(xs), len(ys))
    xs, ys = xs[:n], ys[:n]
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)) if xs and ys else None


def _apply_M(bb, M):
    x, y, w, h = bb
    P = np.array([[x, y, 1], [x + w, y, 1], [x, y + h, 1], [x + w, y + h, 1]], float) @ M.T
    return (float(P[:, 0].min()), float(P[:, 1].min()),
            float(P[:, 0].max() - P[:, 0].min()), float(P[:, 1].max() - P[:, 1].min()))


def parse_svg_dom(svg_text, min_size=12.0, max_canvas_frac=0.6, is_area=None):
    """Applies the <g transform> chain to (1) area boxes + <text> labels,
    (2) the ROOT coordinates of each rect/line/polygon/path with an id.
    Returns: (nodes, id_bboxes, (canvas_w, canvas_h)).
    Default area definition: rect/polygon/path with a fill other than 'none', not too small/large.
    For custom rules, provide is_area(el, bbox) -> bool."""
    if re.search(r"<!ENTITY", svg_text, re.IGNORECASE):
        raise ValueError("there is a <!ENTITY> definition in the SVG; it has been rejected for security reasons.")
    try:
        root = ET.fromstring(svg_text.encode("utf-8"))
    except ET.ParseError as e:
        raise ValueError(f"SVG could not be parsed: {e}") from e
    vb = root.get("viewBox")
    try:
        if vb:
            cw, ch = [float(v) for v in vb.replace(",", " ").split()[2:4]]
        else:
            width_attr = root.get("width")
            height_attr = root.get("height")
            if width_attr is None or height_attr is None:
                raise ValueError("SVG width/height is missing.")
            cw = float(re.sub(r"[^\d.]", "", width_attr))
            ch = float(re.sub(r"[^\d.]", "", height_attr))
    except (TypeError, ValueError, IndexError) as e:
        raise ValueError("SVG viewBox / width-height could not be read.") from e
    nodes, texts, id_bboxes = [], [], {}

    def walk(el, M):
        if el.get("transform"):
            M = M @ _parse_transform(el.get("transform"))
        t = _local(el.tag)
        if t in ("rect", "polygon", "polyline", "path", "line"):
            bb = _raw_bbox(el)
            if bb:
                bb = _apply_M(bb, M)
                if el.get("id"):
                    id_bboxes[el.get("id")] = bb
                if t in ("rect", "polygon", "path"):
                    ok = (bb[2] >= min_size and bb[3] >= min_size
                          and bb[2] * bb[3] < max_canvas_frac * cw * ch
                          and _style_fill(el) != "none")
                    if is_area is not None:
                        ok = ok and is_area(el, bb)
                    if ok:
                        nodes.append({"bbox": bb, "id": el.get("id"), "labels": [], "area": bb[2] * bb[3]})
        elif t == "text":
            tsp = [c for c in el.iter() if _local(c.tag) == "tspan" and c.get("x") is not None]
            if tsp:
                items = [(" ".join("".join(c.itertext()).split()), c) for c in tsp]
            else:
                items = [(" ".join("".join(el.itertext()).split()), el)]
            for s, src in items:
                if s:
                    try:
                        px = float(src.get("x", 0) or 0) + 1.0
                        py = float(src.get("y", 0) or 0) - 3.0
                    except ValueError:
                        continue
                    p = M @ np.array([px, py, 1.0])
                    texts.append((s, (float(p[0]), float(p[1]))))
        for c in el:
            walk(c, M)

    walk(root, np.eye(3))
    # --- containment tree: parent = the smallest LARGER box that contains the center
    for i, a in enumerate(nodes):
        cx, cy = a["bbox"][0] + a["bbox"][2] / 2, a["bbox"][1] + a["bbox"][3] / 2
        cands = [j for j, b in enumerate(nodes)
                 if j != i and b["area"] > a["area"] * 1.001
                 and b["bbox"][0] <= cx <= b["bbox"][0] + b["bbox"][2]
                 and b["bbox"][1] <= cy <= b["bbox"][1] + b["bbox"][3]]
        a["parent"] = min(cands, key=lambda j: nodes[j]["area"]) if cands else None

    def depth(i):
        d = 0
        while nodes[i]["parent"] is not None:
            i = nodes[i]["parent"]
            d += 1
        return d

    for i, a in enumerate(nodes):
        a["depth"] = depth(i)
    # --- labels: are written in the SMALLEST box that contains every point of the <text>
    for s, (px, py) in texts:
        cands = [a for a in nodes if a["bbox"][0] <= px <= a["bbox"][0] + a["bbox"][2]
                 and a["bbox"][1] <= py <= a["bbox"][1] + a["bbox"][3]]
        if cands:
            min(cands, key=lambda a: a["area"])["labels"].append(s)
    return nodes, id_bboxes, (cw, ch)


def dom_positions(nodes, level_codes, fallback=None, log=None):
    """level_codes: [[level0 codes], [level1 codes], ...].
    Returns: (positions[level]{code:bbox}, ids[level]{code:svg_id}, missing[level][code...]).
    Level = the containment depth of the box (mode of the depths of candidate boxes); ID is not considered.
    For codes that cannot be found by labeling, `fallback(level, missing)` (ID-based fallback) is called."""
    log = log if log is not None else []
    out_pos, out_ids, out_missing = [], [], []
    for lvl, codes in enumerate(level_codes):
        pats = {c: re.compile(rf"(?<![\w-]){re.escape(str(c))}(?![\w-])", re.IGNORECASE) for c in codes}
        hits = [(nd, c) for nd in nodes for c, p in pats.items() if any(p.search(l) for l in nd["labels"])]
        pos, ids = {}, {}
        if hits:
            depths = [nd["depth"] for nd, _ in hits]
            mode = max(set(depths), key=depths.count)
            for nd, c in hits:
                if nd["depth"] == mode and (c not in pos or nd["area"] > pos[c][2] * pos[c][3]):
                    pos[c], ids[c] = nd["bbox"], nd["id"]
            log.append(f"Level {lvl}: containment depth={mode}, {len(pos)}/{len(codes)} codes matched from DOM")
        missing = [c for c in codes if c not in pos]
        if missing and fallback is not None:
            fb_pos, fb_ids = fallback(lvl, missing)
            if fb_pos:
                pos.update(fb_pos)
                ids.update(fb_ids)
                log.append(f"Level {lvl}: {len(fb_pos)} codes did not match labels, found with ID fallback "
                           f"(level information should be checked)")
        out_pos.append(pos)
        out_ids.append(ids)
        out_missing.append([c for c in codes if c not in pos])
    return out_pos, out_ids, out_missing


# ============================================================================
# 2) Factory border / wall / door geometry
# ============================================================================
def build_factory_geometry(id_bboxes, zones, wall_t):
    """zones: {name: {"border_id": str, "door_ids": [str, ...]}}.
    Returns: ({zone: {'bbox','doors','walls'}}, warnings).
    Walls = the 4 sides of the border rectangle, with door intervals removed."""
    geo, warns = {}, []
    for name, z in (zones or {}).items():
        bb = id_bboxes.get(z.get("border_id"))
        if bb is None:
            warns.append(f"'{z.get('border_id')}' could not be found, {name} zone skipped.")
            continue
        bx, by, bw, bh = bb
        doors = []
        for did in z.get("door_ids", []):
            db = id_bboxes.get(did)
            if db is None:
                warns.append(f"{name}: door '{did}' could not be found.")
                continue
            dx, dy, dw, dh = db
            dcx, dcy = dx + dw / 2, dy + dh / 2
            dist = {"N": abs(dcy - by), "S": abs(dcy - (by + bh)),
                    "W": abs(dcx - bx), "E": abs(dcx - (bx + bw))}
            side = min(dist, key=lambda d: dist[d])
            if side in "NS":
                lo, hi, line, c = dx, dx + dw, (by if side == "N" else by + bh), dcx
            else:
                lo, hi, line, c = dy, dy + dh, (bx if side == "W" else bx + bw), dcy
            doors.append({"id": did, "side": side, "lo": lo, "hi": hi, "line": line, "center": c})
        walls = []
        for side in "NSWE":
            ds = sorted((d["lo"], d["hi"]) for d in doors if d["side"] == side)
            if side in "NS":
                a0, a1, line = bx, bx + bw, (by if side == "N" else by + bh)
            else:
                a0, a1, line = by, by + bh, (bx if side == "W" else bx + bw)
            segs, cur = [], a0
            for lo, hi in ds:
                if lo > cur:
                    segs.append((cur, lo))
                cur = max(cur, hi)
            if cur < a1:
                segs.append((cur, a1))
            for s0, s1 in segs:
                if side in "NS":
                    walls.append((s0, line - wall_t / 2, s1 - s0, wall_t))
                else:
                    walls.append((line - wall_t / 2, s0, wall_t, s1 - s0))
        geo[name] = {"bbox": bb, "doors": doors, "walls": walls}
    return geo, warns


def zone_of_point(pt, geo):
    for name, g in (geo or {}).items():
        x, y, w, h = g["bbox"]
        if x < pt[0] < x + w and y < pt[1] < y + h:
            return name
    return None


def door_passage(door, lane_off, off):
    """Door passage points. For orthogonal passage, both points are on the same lane."""
    nx, ny = _NORMAL[door["side"]]
    c = door["center"] + lane_off
    p = (c, door["line"]) if door["side"] in "NS" else (door["line"], c)
    return (p[0] - nx * off, p[1] - ny * off), (p[0] + nx * off, p[1] + ny * off)


def _pick_doors(A, B, zA, zB, geo, used, door_off, balance_px=None):
    """The most shortest (Manhattan) door-door pair. If equal length, the door closest to the source is selected."""
    balance_px = DOOR_BALANCE_PX if balance_px is None else balance_px

    def man(p, q):
        return abs(p[0] - q[0]) + abs(p[1] - q[1])

    docsA = geo[zA]["doors"] if zA in geo else []
    docsB = geo[zB]["doors"] if zB in geo else []
    best = None
    for dA in (docsA or [None]):
        for dB in (docsB or [None]):
            chain, tie = [A], 0.0
            if dA:
                i, o = door_passage(dA, 0.0, door_off)
                chain += [i, o]
                tie += man(A, i)
            if dB:
                i, o = door_passage(dB, 0.0, door_off)
                chain += [o, i]
                tie += man(B, i)
            chain.append(B)
            cost = sum(man(p, q) for p, q in itertools.pairwise(chain))
            cost += balance_px * sum(used.get(d["id"], 0) for d in (dA, dB) if d)
            cost += 1e-3 * tie
            if best is None or cost < best[0]:
                best = (cost, dA, dB)
    if best is None:
        return None, None
    return best[1], best[2]


# ---- real distance (meter)
def _center_of(bbox):
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def door_path_length_px(pA, zA, pB, zB, geo):
    """Vertical-horizontal (forklift/transportation) distance; shortest door(s) across different zones."""
    def man(p, q):
        return abs(p[0] - q[0]) + abs(p[1] - q[1])

    if zA == zB or not geo:
        return man(pA, pB)
    docsA = geo[zA]["doors"] if zA in geo else []
    docsB = geo[zB]["doors"] if zB in geo else []
    best = math.inf
    for dA in (docsA or [None]):
        for dB in (docsB or [None]):
            chain = [pA]
            if dA:
                chain.append(door_passage(dA, 0.0, 0.0)[0])
            if dB:
                chain.append(door_passage(dB, 0.0, 0.0)[0])
            chain.append(pB)
            best = min(best, sum(man(p, q) for p, q in itertools.pairwise(chain)))
    return best


def layout_distance(a, b, positions, geo, px_to_m):
    """Returns: (straight-line m, door-based m, zone_a, zone_b)."""
    ca, cb = _center_of(positions[a]), _center_of(positions[b])
    za, zb = zone_of_point(ca, geo), zone_of_point(cb, geo)
    eu = math.hypot(cb[0] - ca[0], cb[1] - ca[1])
    walk = door_path_length_px(ca, za, cb, zb, geo)
    return eu * px_to_m, walk * px_to_m, za, zb


# ============================================================================
# 3) Level style (line width, arrow, font... derived from box width)
# ============================================================================
def derive_level_style(positions, ref_stat="max", overrides=None):
    ov = dict(overrides or {})
    thick = np.array([min(w, h) for (_x, _y, w, h) in positions.values() if w > 0 and h > 0], dtype=float)
    if thick.size == 0:
        u = 50.0
    elif ref_stat == "max":
        u = float(thick.max())
    elif ref_stat == "p90":
        u = float(np.percentile(thick, 90))
    elif ref_stat == "median":
        u = float(np.median(thick))
    else:
        raise ValueError("ref_stat: 'max' | 'p90' | 'median'")

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    st = {
        "ref": u,
        "max_width": clamp(u * 0.06, 1.2, 10.0),
        "min_width": clamp(u * 0.02, 0.5, 2.0),
        "font_size": clamp(u * 0.12, 6.0, 16.0),
        "arrow_min": clamp(u * 0.05, 3.0, 10.0),
        "arrow_k": 3.2,
        "arrow_half_ratio": 0.45,
        "loop_size": clamp(u * 0.10, 6.0, 40.0),
        "loop_gap": clamp(u * 0.08, 4.0, 30.0),
        "label_halo": None,
        "title_font": None,
        "cell_size": None,
        "corridor_margin": None,
        "buffer_radius": None,
        "buffer_weight": 0.35,
        "port_gap_min": None,
        "port_gap_comfort": None,
        "stub": None,
        "stub_step": None,
    }
    st.update(ov)
    if st["label_halo"] is None:
        st["label_halo"] = st["font_size"] * 0.35
    if st["title_font"] is None:
        st["title_font"] = clamp(st["font_size"] * 1.5, 10.0, 24.0)
    if st["cell_size"] is None:
        st["cell_size"] = int(clamp(round(u / 40.0), 2, 10))
    if st["corridor_margin"] is None:
        st["corridor_margin"] = max(clamp(u * 0.05, 3.0, 24.0), st["max_width"] * 0.75 + 1.0)
    if st["buffer_radius"] is None:
        st["buffer_radius"] = int(clamp(math.ceil(st["max_width"] / st["cell_size"]), 1, 3))
    cs = st["cell_size"]
    arrow_len_max = max(st["arrow_min"], st["arrow_k"] * st["max_width"])
    if st["port_gap_min"] is None:
        st["port_gap_min"] = max(3.2 * st["max_width"], 2.0 * cs, 4.0)
    if st["port_gap_comfort"] is None:
        st["port_gap_comfort"] = st["port_gap_min"] * 2.0
    if st["stub_step"] is None:
        st["stub_step"] = max(2.0 * cs, st["max_width"] * 2.0)
    if st["stub"] is None:
        st["stub"] = max(arrow_len_max * 1.4, st["corridor_margin"] + cs * 1.5)
    return st


# ============================================================================
# 4) Arrow / label / self-loop drawing
# ============================================================================
def _arrow_for(path, w, st):
    """path: world-coordinate path. Returns: (trimmed_path, arrow_polygon_points_str)."""
    L = max(st["arrow_min"], st["arrow_k"] * w)
    hw = L * st["arrow_half_ratio"]
    (x0, y0), (x1, y1) = path[-2], path[-1]
    dx, dy = x1 - x0, y1 - y0
    seg = math.hypot(dx, dy)
    ux, uy = (1.0, 0.0) if seg < 1e-9 else (dx / seg, dy / seg)
    bx, by = x1 - ux * L, y1 - uy * L
    if seg > L:
        trimmed = list(path[:-1]) + [(bx, by)]
    else:
        trimmed = list(path[:-1])
        if len(trimmed) < 2:
            trimmed = [path[0], (bx, by)]
    px, py = -uy, ux
    poly = (f"{x1:.1f},{y1:.1f} {bx + px * hw:.1f},{by + py * hw:.1f} "
            f"{bx - px * hw:.1f},{by - py * hw:.1f}")
    return trimmed, poly


def _label_anchor(path, w, st):
    """Places the label at the center of the longest segment of the path. Returns: (x, y, text_anchor)."""
    if not path:
        return 0.0, 0.0, "middle"
    if len(path) == 1:
        x, y = path[0]
        f = st["font_size"]
        return x, y - w / 2.0 - f * 0.3, "middle"

    best, best_len = None, -1.0
    for i in range(len(path) - 1):
        (xa, ya), (xb, yb) = path[i], path[i + 1]
        ln = math.hypot(xb - xa, yb - ya)
        if ln > best_len:
            best, best_len = (xa, ya, xb, yb), ln

    if best is None:
        x, y = path[0]
        f = st["font_size"]
        return x, y - w / 2.0 - f * 0.3, "middle"

    xa, ya, xb, yb = best
    mx, my = (xa + xb) / 2.0, (ya + yb) / 2.0
    f = st["font_size"]
    if abs(xb - xa) >= abs(yb - ya):
        return mx, my - w / 2.0 - f * 0.3, "middle"
    return mx + w / 2.0 + f * 0.3, my + f * 0.35, "start"


def _label_svg(x, y, anchor, text, st):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial" font-size="{st["font_size"]:.1f}" '
            f'font-weight="bold" fill="#1B2631" text-anchor="{anchor}" '
            f'style="paint-order: stroke; stroke:#FFFFFF; stroke-width:{st["label_halo"]:.1f}px;">{text}</text>')


def _self_loop_geometry(bbox, cnt, st, stroke_w, label_text=None):
    x, y, w, _h = bbox
    L = max(st["arrow_min"], st["arrow_k"] * stroke_w)
    hw = L * st["arrow_half_ratio"]
    gap = max(min(st["loop_gap"], w * 0.30), stroke_w * 2.5)
    loop_h = max(st["loop_size"], L * 1.6)
    p_out = (x + w - gap * 2, y)
    p_in = (x + w - gap, y)
    mid = (p_out[0] + p_in[0]) / 2.0
    f = st["font_size"]
    label_y = y - 0.75 * loop_h - stroke_w / 2.0 - f * 0.35
    txt_hw = len(label_text or f"{cnt:,}") * f * 0.36 + st["label_halo"]
    pad = f * 0.3
    rx0 = min(p_out[0] - hw, mid - txt_hw) - pad
    rx1 = max(p_in[0] + hw, mid + txt_hw) + pad
    ry0 = min(y - loop_h - hw, label_y - f) - pad
    return {"L": L, "hw": hw, "gap": gap, "loop_h": loop_h, "p_out": p_out, "p_in": p_in,
            "mid": mid, "label_y": label_y, "rect": (rx0, ry0, rx1 - rx0, y - ry0)}


def draw_self_loop_svg(bbox, count, color, st, stroke_w, label_text=None):
    geo = _self_loop_geometry(bbox, count, st, stroke_w, label_text)
    y = bbox[1]
    L, hw, loop_h = geo["L"], geo["hw"], geo["loop_h"]
    p_out, p_in = geo["p_out"], geo["p_in"]
    end = (p_in[0], y - L)
    path = (f'<path d="M {p_out[0]:.1f},{p_out[1]:.1f} '
            f'C {p_out[0]:.1f},{y - loop_h:.1f} {p_in[0]:.1f},{y - loop_h:.1f} {end[0]:.1f},{end[1]:.1f}" '
            f'fill="none" stroke="{color}" stroke-width="{stroke_w:.2f}" '
            f'stroke-linecap="round" opacity="0.9"/>')
    arrow = (f'<polygon points="{p_in[0]:.1f},{p_in[1]:.1f} '
             f'{p_in[0] - hw:.1f},{y - L:.1f} {p_in[0] + hw:.1f},{y - L:.1f}" '
             f'fill="{color}" opacity="0.9"/>')
    label = _label_svg(geo["mid"], geo["label_y"], "middle", label_text or f"{count:,}", st)
    return path, arrow, label


def _loop_rect_for_side(bbox, cnt, st, stroke_w, side, label_text=None):
    rx, ry, rw, rh = _self_loop_geometry(bbox, cnt, st, stroke_w, label_text)["rect"]
    if side == "N":
        return (rx, ry, rw, rh)
    return (rx, bbox[1] + bbox[3], rw, rh)


def draw_self_loop_svg_side(bbox, count, color, st, stroke_w, side="N", label_text=None):
    lp, la, ll = draw_self_loop_svg(bbox, count, color, st, stroke_w, label_text)
    if side == "N":
        return lp, la, ll
    _x, y, _w, h = bbox
    geo = _self_loop_geometry(bbox, count, st, stroke_w, label_text)
    f = st["font_size"]
    label = _label_svg(geo["mid"], (2 * y + h) - geo["label_y"] + f * 0.75, "middle",
                       label_text or f"{count:,}", st)
    g = f'<g transform="translate(0,{2 * y + h:.2f}) scale(1,-1)">{lp}{la}</g>'
    return g, "", label


# ============================================================================
# 5) Ortogonal route helpers
# ============================================================================
def _orthogonalize(points):
    out = [points[0]]
    for p in points[1:]:
        q = out[-1]
        if abs(p[0] - q[0]) > 1e-6 and abs(p[1] - q[1]) > 1e-6:
            out.append((p[0], q[1]))
        out.append(p)
    return out


def _simplify_points(pts, eps=1e-6):
    """Removes duplicate and collinear intermediate points; leaves endpoints untouched."""
    dedup = []
    for p in pts:
        if dedup and abs(p[0] - dedup[-1][0]) < eps and abs(p[1] - dedup[-1][1]) < eps:
            continue
        dedup.append(p)
    if len(dedup) < 3:
        return dedup
    res = [dedup[0]]
    for i in range(1, len(dedup) - 1):
        a, b, c = res[-1], dedup[i], dedup[i + 1]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        dot = (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1])
        if abs(cross) < eps and dot > 0:
            continue
        res.append(b)
    res.append(dedup[-1])
    return res


def _sdir(p, q):
    dx, dy = q[0] - p[0], q[1] - p[1]
    if abs(dx) > EPS and abs(dy) < EPS:
        return "E" if dx > 0 else "W"
    if abs(dy) > EPS and abs(dx) < EPS:
        return "S" if dy > 0 else "N"
    return None


def _path_cost(pts, d_start, d_end):
    """length + BEND_PEN*bends. Start/end directions must be perpendicular to the sides; backtracking is forbidden."""
    dirs = [d for d in (_sdir(p, q) for p, q in itertools.pairwise(pts)) if d]
    if not dirs:
        return math.inf, 0
    L = sum(abs(p[0] - q[0]) + abs(p[1] - q[1]) for p, q in itertools.pairwise(pts))
    bends, prev = 0, d_start
    for d in dirs:
        if d != prev:
            bends += 1
            prev = d
    if dirs[-1] != d_end:
        bends += 1
    pen = (1e6 if dirs[0] == _OPP[d_start] else 0) + (1e6 if dirs[-1] == _OPP[d_end] else 0)
    return L + BEND_PEN * bends + pen, bends


def _direct_candidates(A, B):
    (x0, y0), (x1, y1) = A, B
    c = [[A, B]] if (abs(x0 - x1) < EPS or abs(y0 - y1) < EPS) else []
    c += [[A, (x1, y0), B], [A, (x0, y1), B]]
    for f in (0.5, 0.25, 0.75):
        mx, my = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        c += [[A, (mx, y0), (mx, y1), B], [A, (x0, my), (x1, my), B]]
    return c


def _port_xy(bb, side, t):
    x, y, w, h = bb
    return {"N": (t, y), "S": (t, y + h), "W": (x, t), "E": (x + w, t)}[side]


def _stub_xy(p, side, L):
    dx, dy = _DIRV[side]
    return (p[0] + dx * L, p[1] + dy * L)


def _side_candidates(Ca, Cb, tol=2.0):
    """Source sides based on the quadrant of the target center; target sides are their counterparts."""
    dx, dy = Cb[0] - Ca[0], Cb[1] - Ca[1]
    s = []
    if abs(dx) > tol:
        s.append("E" if dx > 0 else "W")
    if abs(dy) > tol:
        s.append("S" if dy > 0 else "N")
    if not s:
        s = ["E"]
    s.sort(key=lambda k: (k in "NS") if abs(dx) >= abs(dy) else (k in "EW"))
    return s, [_OPP[k] for k in s]


def _ideal_coord(side, bb, other, corner):
    """Ideal position along the side: center of overlapping regions if any, otherwise center of the side."""
    x, y, w, h = bb
    X, Y, W, H = other
    if side in "NS":
        lo, hi, a0, a1 = max(x, X), min(x + w, X + W), x, x + w
    else:
        lo, hi, a0, a1 = max(y, Y), min(y + h, Y + H), y, y + h
    c = (lo + hi) / 2 if hi - lo > 2 * corner else (a0 + a1) / 2
    return min(max(c, a0 + corner), a1 - corner)


def box_crossings(points, boxes, shrink=1.0):
    out = []
    for k, (p, q) in enumerate(itertools.pairwise(points)):
        for bi, r in enumerate(boxes):
            if r[2] > 2 * shrink and r[3] > 2 * shrink and _segment_intersects_rect(p, q, r, margin=-shrink):
                out.append((k, bi))
    return out


def count_wall_crossings(points, wall_rects):
    """Counts the number of segments that intersect with walls outside the doors."""
    return sum(any(_segment_intersects_rect(points[i], points[i + 1], r, margin=0.0) for r in wall_rects)
               for i in range(len(points) - 1))


# ---- numpy collision
def _arr(rects):
    return np.array(list(rects), float).reshape(-1, 4)


def _inf(a, m):
    return a + np.array([-m, -m, 2 * m, 2 * m]) if len(a) else a


def a_hit(p, q, arr, m=0.0):
    """if segment is parallel to an axis, checks if it intersects any rectangle in the array (tangent not counted)."""
    if len(arr) == 0:
        return False
    x0, x1 = (p[0], q[0]) if p[0] <= q[0] else (q[0], p[0])
    y0, y1 = (p[1], q[1]) if p[1] <= q[1] else (q[1], p[1])
    return bool(((x0 < arr[:, 0] + arr[:, 2] + m - EPS) & (x1 > arr[:, 0] - m + EPS) &
                 (y0 < arr[:, 1] + arr[:, 3] + m - EPS) & (y1 > arr[:, 1] - m + EPS)).any())


def a_path_hit(pts, arr, m=0.0):
    return any(a_hit(p, q, arr, m) for p, q in itertools.pairwise(pts))


def a_window(arr, x0, y0, x1, y1):
    if len(arr) == 0:
        return arr
    mk = (arr[:, 0] < x1) & (arr[:, 0] + arr[:, 2] > x0) & (arr[:, 1] < y1) & (arr[:, 1] + arr[:, 3] > y0)
    return arr[mk]


# ---- Hanan grid + bending-penalized Dijkstra (NumPy masked)
def r4_hanan(A, B, d_start, d_end, arr):
    rects = arr.tolist()
    xs = sorted({A[0], B[0]} | {v for r in rects for v in (r[0], r[0] + r[2])})
    ys = sorted({A[1], B[1]} | {v for r in rects for v in (r[1], r[1] + r[3])})
    nx, ny = len(xs), len(ys)
    xa, ya = np.array(xs), np.array(ys)
    vb = np.zeros((ny, nx), bool)
    hb = np.zeros((ny, max(nx - 1, 1)), bool)
    wb = np.zeros((max(ny - 1, 1), nx), bool)
    for rx, ry, rw, rh in rects:
        i0, i1 = int(np.searchsorted(xa, rx)), int(np.searchsorted(xa, rx + rw))
        j0, j1 = int(np.searchsorted(ya, ry)), int(np.searchsorted(ya, ry + rh))
        vb[j0 + 1:j1, i0 + 1:i1] = True
        hb[j0 + 1:j1, i0:i1] = True
        wb[j0:j1, i0 + 1:i1] = True
    vbl, hbl, wbl = vb.tolist(), hb.tolist(), wb.tolist()
    xi, yi = {x: i for i, x in enumerate(xs)}, {y: j for j, y in enumerate(ys)}
    si, sj, gi, gj = xi[A[0]], yi[A[1]], xi[B[0]], yi[B[1]]
    if vbl[sj][si] or vbl[gj][gi]:
        return None
    d0 = _DNAMES.index(d_start)
    de = _DNAMES.index(d_end)
    _h = lambda i, j: abs(xs[i] - xs[gi]) + abs(ys[j] - ys[gj]) 
    start = (si, sj, d0)
    best, prev, pq, cnt = {start: 0.0}, {}, [(_h(si, sj), 0, 0.0, start)], 0
    while pq:
        _, _, c, cur = heapq.heappop(pq)
        if best.get(cur, math.inf) < c - 1e-9:
            continue
        i, j, d = cur
        if (i, j) == (gi, gj):
            pts, key = [], cur
            while True:
                pts.append((xs[key[0]], ys[key[1]]))
                if key == start:
                    break
                key = prev[key]
            return _simplify_points(pts[::-1])
        for nd in range(4):
            if nd == (d ^ 1):
                continue
            if nd == 0:
                ok = i + 1 < nx and not hbl[j][i] and not vbl[j][i + 1]
                ni, nj = i + 1, j
            elif nd == 1:
                ok = i - 1 >= 0 and not hbl[j][i - 1] and not vbl[j][i - 1]
                ni, nj = i - 1, j
            elif nd == 2:
                ok = j + 1 < ny and not wbl[j][i] and not vbl[j + 1][i]
                ni, nj = i, j + 1
            else:
                ok = j - 1 >= 0 and not wbl[j - 1][i] and not vbl[j - 1][i]
                ni, nj = i, j - 1
            if not ok:
                continue
            nc = c + abs(xs[ni] - xs[i]) + abs(ys[nj] - ys[j]) + (BEND_PEN if nd != d else 0)
            if (ni, nj) == (gi, gj) and nd != de:
                nc += BEND_PEN
            key = (ni, nj, nd)
            if nc < best.get(key, math.inf) - 1e-9:
                best[key] = nc
                prev[key] = cur
                cnt += 1
                heapq.heappush(pq, (nc + _h(ni, nj), cnt, nc, key))
    return None


def r4_route_between(A, B, ds, de, arr, allow_hanan=True):
    if math.dist(A, B) < EPS:
        return [A]
    best = None
    for c in _direct_candidates(A, B):
        if not a_path_hit(c, arr):
            cost = _path_cost(c, ds, de)[0]
            if best is None or cost < best[0]:
                best = (cost, c)
    if best:
        return _simplify_points(best[1])
    if not allow_hanan:
        return None
    for pad in (200.0, 600.0, None):
        sub = arr if pad is None else a_window(arr, min(A[0], B[0]) - pad, min(A[1], B[1]) - pad,
                                               max(A[0], B[0]) + pad, max(A[1], B[1]) + pad)
        pts = r4_hanan(A, B, ds, de, sub)
        if pts and not a_path_hit(pts, arr):
            return pts
    return None


# ---- Door sprites / port distribution
def r4_lane(door, used, step):
    width = door["hi"] - door["lo"]
    offs = []
    for s in (step, step / 2.0, step / 4.0):
        n = max(1, int((width - 4.0) // s) + 1)
        for o in sorted(((i - (n - 1) / 2.0) * s for i in range(n)), key=abs):
            if all(abs(o - x) > 1e-6 for x in offs):
                offs.append(o)
    k = used.get(door["id"], 0)
    used[door["id"]] = k + 1
    return offs[k % len(offs)]


def r4_spread(ts, lo, hi, gap):
    n = len(ts)
    if n == 1:
        return [min(max(ts[0], lo), hi)]
    order = sorted(range(n), key=lambda i: ts[i])
    v = [ts[i] for i in order]
    for k in range(1, n):
        v[k] = max(v[k], v[k - 1] + gap)
    sh = (sum(ts) - sum(v)) / n
    v = [x + sh for x in v]
    if v[-1] > hi:
        d = v[-1] - hi
        v = [x - d for x in v]
    if v[0] < lo:
        d = lo - v[0]
        v = [x + d for x in v]
    if v[-1] > hi:
        v = [lo + (hi - lo) * k / (n - 1) for k in range(n)]
    out = [0.0] * n
    for pos, i in enumerate(order):
        out[i] = v[pos]
    return out


def r4_assign_lanes(entries, door, clearance):
    """entries: [(key, thickness, order_value)].
    Returns: (deviation from center {key: px}, width {key: px}, shrinkage_ratio)."""
    entries = sorted(entries, key=lambda e: e[2])
    n = len(entries)
    avail = (door["hi"] - door["lo"]) - 2 * clearance
    raw = sum(e[1] for e in entries)
    scale = 1.0 if raw + clearance * (n - 1) <= avail else max(0.05, (avail - clearance * (n - 1)) / raw)
    ws = [max(1.0, e[1] * scale) for e in entries]
    x = -(sum(ws) + clearance * (n - 1)) / 2.0
    offs, widths = {}, {}
    for e, w in zip(entries, ws):
        offs[e[0]] = x + w / 2.0
        widths[e[0]] = w
        x += w + clearance
    if n > 1 and sum(ws) + clearance * (n - 1) > avail:
        # door capacity exceeded: lines must still remain within the door opening (not extend to walls);
        # strips are distributed equally, remaining overlap is reported as 'OVERLAP' in the log.
        half = max(0.0, (avail - 1.0) / 2.0)
        for j, e in enumerate(entries):
            offs[e[0]] = -half + 2 * half * j / (n - 1)
    return offs, widths, scale


# ---- parallel strip separation
def r4_orient(p, q, eps=0.6):
    if abs(p[1] - q[1]) < eps and abs(p[0] - q[0]) > eps:
        return "H"
    if abs(p[0] - q[0]) < eps and abs(p[1] - q[1]) > eps:
        return "V"
    return None


def r4_try_move(pts, k, o, c, hard, min_end, apply=True):
    n = len(pts) - 1
    v0, v1 = pts[k], pts[k + 1]
    nv0 = (v0[0], c) if o == "H" else (c, v0[1])
    nv1 = (v1[0], c) if o == "H" else (c, v1[1])
    if a_hit(nv0, nv1, hard):
        return False
    for kk, pv, nv, far in ((k - 1, v0, nv0, pts[k - 1]), (k + 1, v1, nv1, pts[k + 2])):
        old = abs(far[0] - pv[0]) + abs(far[1] - pv[1])
        new = abs(far[0] - nv[0]) + abs(far[1] - nv[1])
        dot = (pv[0] - far[0]) * (nv[0] - far[0]) + (pv[1] - far[1]) * (nv[1] - far[1])
        if dot <= 0 or new < (min_end if kk in (0, n - 1) else 2.0):
            return False
        if new > old and a_hit(pv, nv, hard):
            return False
    if apply:
        pts[k], pts[k + 1] = nv0, nv1
    return True


def r4_measure(routes, near_px):
    H, V = defaultdict(list), defaultdict(list)
    for i, pts in enumerate(routes):
        for p, q in itertools.pairwise(pts):
            o = r4_orient(p, q)
            if o == "H":
                H[round(p[1], 1)].append((min(p[0], q[0]), max(p[0], q[0]), i))
            elif o == "V":
                V[round(p[0], 1)].append((min(p[1], q[1]), max(p[1], q[1]), i))
    exact = near = 0.0
    for D in (H, V):
        keys = sorted(D)
        for ia, a in enumerate(keys):
            for b in keys[ia:]:
                if b - a >= near_px:
                    break
                for s1 in D[a]:
                    for s2 in D[b]:
                        if s1[2] == s2[2] or (a == b and s1[2] > s2[2]):
                            continue
                        ln = min(s1[1], s2[1]) - max(s1[0], s2[0])
                        if ln > 1:
                            if b - a < 1.0:
                                exact += ln
                            else:
                                near += ln
    return exact, near


def r4_separate(routes, hard_for, fixed_arr, gap, min_end, iters=5):
    """Splits overlapping or closely spaced PARALLEL segments lying on the same line into strips.
    Port segments and segments passing through doors are fixed; they do not enter box/wall/door blockers.
    Returns: (number of moved segments, {'fixed':.., 'no_space':..})."""
    stats = {"fixed": 0, "no_space": 0}
    total = 0
    for _it in range(iters):
        fixedflag = [[a_hit(p[k], p[k + 1], fixed_arr) for k in range(len(p) - 1)] for p in routes]
        segs = []
        for ri, pts in enumerate(routes):
            n = len(pts) - 1
            ff = fixedflag[ri]
            for k in range(n):
                p, q = pts[k], pts[k + 1]
                o = r4_orient(p, q)
                if o is None:
                    continue
                c = p[1] if o == "H" else p[0]
                a_, b_ = sorted((p[0], q[0])) if o == "H" else sorted((p[1], q[1]))
                mv = 0 < k < n - 1 and not ff[k] and not ff[k - 1] and not ff[k + 1]
                segs.append({"r": ri, "k": k, "o": o, "c": c, "a": a_, "b": b_, "mv": mv})
        parent = list(range(len(segs)))

        def find(i, parent=parent):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        conflict = False
        for o in "HV":
            idx = sorted([i for i, s in enumerate(segs) if s["o"] == o], key=lambda i: segs[i]["c"])
            for ii, i in enumerate(idx):
                si = segs[i]
                for j in idx[ii + 1:]:
                    sj = segs[j]
                    if sj["c"] - si["c"] >= gap * 0.999:
                        break
                    if si["r"] != sj["r"] and min(si["b"], sj["b"]) - max(si["a"], sj["a"]) > 1.0 \
                            and (si["mv"] or sj["mv"]):
                        parent[find(i)] = find(j)
                        conflict = True
        if not conflict:
            break
        comps = defaultdict(list)
        for i in range(len(segs)):
            comps[find(i)].append(i)
        moved = 0
        for members in comps.values():
            mv = [i for i in members if segs[i]["mv"]]
            stats["fixed"] += len(members) - len(mv)
            if len(members) < 2 or not mv:
                continue
            o = segs[members[0]]["o"]
            placed = [(segs[i]["c"], segs[i]["a"], segs[i]["b"], segs[i]["r"]) for i in members if not segs[i]["mv"]]

            def sign(i, o=o):
                s = segs[i]  # noqa: B023
                pts = routes[s["r"]]
                k = s["k"]

                def sg(nb, cur):
                    d = (nb[1] - cur[1]) if o == "H" else (nb[0] - cur[0])
                    return -1 if d < -0.6 else (1 if d > 0.6 else 0)

                return sg(pts[k - 1], pts[k]) + sg(pts[k + 2], pts[k + 1])

            mv.sort(key=lambda i: (sign(i), segs[i]["c"], segs[i]["r"]))
            cm = float(np.mean([segs[i]["c"] for i in members]))
            for rank, i in enumerate(mv):
                s = segs[i]
                pts = routes[s["r"]]
                k = s["k"]
                p, q = pts[k], pts[k + 1]
                a_, b_ = sorted((p[0], q[0])) if o == "H" else sorted((p[1], q[1]))
                desired = cm + (rank - (len(mv) - 1) / 2.0) * gap
                hard = hard_for(s["r"])
                chosen = None
                for c in sorted((desired + t * gap * 0.5 for t in range(-40, 41)), key=lambda c: abs(c - desired)):
                    if any(pr != s["r"] and abs(c - pc) < gap - 1e-6 and min(b_, pb) - max(a_, pa) > 1.0
                           for pc, pa, pb, pr in placed):
                        continue
                    if r4_try_move(pts, k, o, c, hard, min_end):
                        chosen = c
                        break
                if chosen is None:
                    stats["no_space"] += 1
                    placed.append((s["c"], a_, b_, s["r"]))
                else:
                    placed.append((chosen, a_, b_, s["r"]))
                    moved += 1
        for ri in range(len(routes)):
            routes[ri] = _simplify_points(routes[ri])
        total += moved
        if moved == 0:
            break
    return total, stats


def r4_top_overlaps(routes, labels, fixed_arr, topn=12):
    """Diagnosis: longest overlaps. Returns: log lines."""
    segs = []
    for ri, pts in enumerate(routes):
        n = len(pts) - 1
        for k, (p, q) in enumerate(itertools.pairwise(pts)):
            o = r4_orient(p, q)
            if o is None:
                continue
            c = p[1] if o == "H" else p[0]
            a_, b_ = sorted((p[0], q[0])) if o == "H" else sorted((p[1], q[1]))
            role = "door" if a_hit(p, q, fixed_arr) else ("first" if k == 0 else ("last" if k == n - 1 else "middle"))
            segs.append((o, c, a_, b_, ri, role))
    res = []
    for o in "HV":
        S = sorted([s for s in segs if s[0] == o], key=lambda s: s[1])
        for i, s in enumerate(S):
            for t in S[i + 1:]:
                if t[1] - s[1] >= 1.0:
                    break
                if s[4] == t[4]:
                    continue
                ln = min(s[3], t[3]) - max(s[2], t[2])
                if ln > 1:
                    res.append((ln, o, s[1], labels[s[4]], s[5], labels[t[4]], t[5],
                                len(routes[s[4]]) - 1, len(routes[t[4]]) - 1))
    res.sort(reverse=True)
    pair = Counter()
    for r in res:
        pair[tuple(sorted((r[4], r[6])))] += r[0]
    out = []
    if pair:
        out.append("Overlap px, segment role pair: " + str({k: round(v) for k, v in pair.items()}))
    for ln, o, c, e1, r1, e2, r2, n1, n2 in res[:topn]:
        out.append(f"  {ln:6.0f}px {o} @{c:.0f} | {e1[0]}->{e1[1]} ({r1}, {n1} seg) x "
                   f"{e2[0]}->{e2[1]} ({r2}, {n2} seg)")
    return out


def r4_overlap_report(routes, fixed_arr, gap, geo, wt):
    """Diagnosis: remaining overlap classes + door capacity. Returns: log lines."""
    segs = []
    for ri, pts in enumerate(routes):
        n = len(pts) - 1
        for k, (p, q) in enumerate(itertools.pairwise(pts)):
            o = r4_orient(p, q)
            if o is None:
                continue
            c = p[1] if o == "H" else p[0]
            a_, b_ = sorted((p[0], q[0])) if o == "H" else sorted((p[1], q[1]))
            cls = "door" if a_hit(p, q, fixed_arr) else ("port" if k in (0, n - 1) else "middle")
            segs.append((o, c, a_, b_, ri, cls))
    tot = defaultdict(float)
    for o in "HV":
        S = sorted([s for s in segs if s[0] == o], key=lambda s: s[1])
        for i, s in enumerate(S):
            for t in S[i + 1:]:
                if t[1] - s[1] >= gap * 0.5:
                    break
                if s[4] == t[4] or abs(t[1] - s[1]) >= 1.0:
                    continue
                ln = min(s[3], t[3]) - max(s[2], t[2])
                if ln > 1:
                    cl = {s[5], t[5]}
                    tot["door" if "door" in cl else ("port" if "port" in cl else "middle")] += ln
    out = ["Remaining overlaps (px) class distribution: " + str({k: round(v) for k, v in tot.items()})]
    for g in geo.values():
        for d in g["doors"]:
            if d["side"] in "NS":
                r = np.array([[d["lo"], d["line"] - wt, d["hi"] - d["lo"], 2 * wt]])
            else:
                r = np.array([[d["line"] - wt, d["lo"], 2 * wt, d["hi"] - d["lo"]]])
            width = d["hi"] - d["lo"]
            cap = max(1, int((width - 4.0) // gap) + 1)
            use = sum(1 for pts in routes if a_path_hit(pts, r))
            out.append(f"  door {d['id']}: width={width:.0f}px | {gap:.0f}px spacing capacity={cap} lanes | "
                       f"lines passing={use}" + ("  <-- OVERFLOW" if use > cap else ""))
    return out


# ============================================================================
# 6) Interaction (hover / zoom / pan)
# because SVG is the first element in the iframe, so document.querySelector('svg') is used.
# ============================================================================
_JS_FLOORPLAN = """
(function() {
    var svg = document.querySelector('svg'),
        layer = svg.getElementById('edges'),
        hl = svg.getElementById('hl');
    var edges = [].slice.call(svg.querySelectorAll('.e')),
        tls = [].slice.call(svg.querySelectorAll('.tl')),
        nodes = [].slice.call(svg.querySelectorAll('.n')),
        byId = {};
    nodes.forEach(function(n) { byId[n.getAttribute('data-id')] = n; });
    function touches(el, id) {
        return el.getAttribute('data-s') === id || el.getAttribute('data-t') === id;
    }
    function enter(id) {
        edges.forEach(function(e) {
            if (touches(e, id)) {
                e.classList.add('hl');
                hl.appendChild(e);
                var o = e.getAttribute('data-s') === id ? e.getAttribute('data-t') : e.getAttribute('data-s');
                if (o !== id && byId[o]) byId[o].classList.add('p');
            } else e.classList.add('dim');
        });
        tls.forEach(function(t) { if (!touches(t, id)) t.classList.add('dim'); });
    }
    function leave() {
        edges.forEach(function(e) {
            e.classList.remove('hl');
            e.classList.remove('dim');
            layer.appendChild(e);
        });
        tls.forEach(function(t) { t.classList.remove('dim'); });
        nodes.forEach(function(n) { n.classList.remove('p'); });
    }
    nodes.forEach(function(n) {
        n.addEventListener('mouseenter', function() { enter(n.getAttribute('data-id')); });
        n.addEventListener('mouseleave', leave);
    });
    var vb = svg.viewBox.baseVal,
        X0 = vb.x, Y0 = vb.y, W0 = vb.width, H0 = vb.height;
    function pt(ev) {
        var p = svg.createSVGPoint();
        p.x = ev.clientX;
        p.y = ev.clientY;
        return p.matrixTransform(svg.getScreenCTM().inverse());
    }
    svg.addEventListener('wheel', function(ev) {
        ev.preventDefault();
        var k = ev.deltaY < 0 ? 0.85 : 1.18,
            p = pt(ev);
        vb.x = p.x - (p.x - vb.x) * k;
        vb.y = p.y - (p.y - vb.y) * k;
        vb.width *= k;
        vb.height *= k;
    }, { passive: false });
    var drag = false, lx = 0, ly = 0;
    svg.addEventListener('mousedown', function(ev) {
        drag = true;
        lx = ev.clientX;
        ly = ev.clientY;
    });
    window.addEventListener('mouseup', function() { drag = false; });
    window.addEventListener('mousemove', function(ev) {
        if (!drag) return;
        var r = svg.getBoundingClientRect(),
            s = Math.max(vb.width / r.width, vb.height / r.height);
        vb.x -= (ev.clientX - lx) * s;
        vb.y -= (ev.clientY - ly) * s;
        lx = ev.clientX;
        ly = ev.clientY;
    });
    svg.addEventListener('dblclick', function() {
        vb.x = X0; vb.y = Y0; vb.width = W0; vb.height = H0;
    });
})();
"""

_CSS_FLOORPLAN = (
    ".e .lbl{display:none}"
    ".e.hl .lbl{display:inline}"
    ".e.dim{opacity:.06}"
    ".e.hl{opacity:1;pointer-events:none}"
    ".tl.dim{opacity:.05}"
    ".n{cursor:pointer}"
    ".n:hover{stroke:#1B2631;stroke-width:2.4;fill:#F7DC6F;fill-opacity:.18}"
    ".n.p{stroke:#E67E22;stroke-width:2.2;fill:#F5CBA7;fill-opacity:.30}"
)


def _make_svg_responsive(content):
    """Root <svg>: if viewbox is not present, generate it from width/height, remove fixed width/height."""
    m = re.search(r'<svg\b[^>]*>', content)
    if not m:
        return content
    tag = m.group(0)
    new = tag
    if "viewBox" not in tag:
        w = re.search(r'\swidth="([\d.]+)', tag)
        h = re.search(r'\sheight="([\d.]+)', tag)
        if w and h:
            new = new.replace("<svg", f'<svg viewBox="0 0 {w.group(1)} {h.group(1)}"', 1)
    new = re.sub(r'\s(?:width|height)="[^"]*"', "", new)
    return content.replace(tag, new, 1)


# ============================================================================
# 7) MAIN OVERLAY: ports with perpendicular contact, 
# door/wall awareness, lane-based routing
# ============================================================================
def build_floor_plan_overlay(content, positions, pairs_df, *, title="", node_colors=None,
                             self_loop_counts=None, factory_geo=None, container_positions=None,
                             node_names=None, wall_t=8.0, min_count=1, top_n=None, label_top_n=40,
                             margin_px=None, style_overrides=None, ref_stat="max", self_loop_min_count=1,
                             mode="freq", perf_df=None, loop_perf=None):
    """content: (painted) floor plan SVG text. Returns: (svg_text, info_dict).
    Raises ValueError if no edges to draw."""
    T0 = time.time()
    log = []
    st = derive_level_style(positions, ref_stat=ref_stat, overrides=style_overrides)
    node_colors = node_colors or {}
    df_ = pairs_df[(pairs_df["count"] >= min_count)
                   & pairs_df["from"].isin(positions) & pairs_df["to"].isin(positions)].copy()
    if mode == "perf":
        if perf_df is None or perf_df.empty:
            raise ValueError("Performance mode requires perf_df.")
        df_ = df_.merge(perf_df[["from", "to", "mean_sec", "min_sec", "max_sec"]],
                        on=["from", "to"], how="left")    
    if top_n:
        df_ = df_.iloc[np.argsort(-df_["count"].to_numpy(), kind="stable")].head(top_n)
    if df_.empty:
        raise ValueError("No edges to draw.")

    def ctr(c):
        return _center_of(positions[c])

    df_["_d"] = [math.dist(ctr(a), ctr(b)) for a, b in zip(df_["from"], df_["to"])]
    df_ = df_.iloc[np.argsort(df_["_d"].to_numpy(), kind="stable")].reset_index(drop=True)

    # ---- duvar / kapı tıkacı / bölge
    geo = factory_geo or {}
    wt = wall_t
    walls = [w for g in geo.values() for w in g["walls"]]
    plugs, fixed_rects = [], []
    for g in geo.values():
        for d in g["doors"]:
            if d["side"] in "NS":
                plugs.append((d["lo"], d["line"] - wt / 2, d["hi"] - d["lo"], wt))
                fixed_rects.append((d["lo"], d["line"] - wt, d["hi"] - d["lo"], 2 * wt))
            else:
                plugs.append((d["line"] - wt / 2, d["lo"], wt, d["hi"] - d["lo"]))
                fixed_rects.append((d["line"] - wt, d["lo"], 2 * wt, d["hi"] - d["lo"]))
    zone = {c: zone_of_point(ctr(c), geo) for c in positions}

    leaf_rects = list(positions.values())
    loops = {c: n for c, n in (self_loop_counts or {}).items() if c in positions and n >= self_loop_min_count}
    max_count = max([df_["count"].max()] + list(loops.values()))
    loop_perf = loop_perf or {}
    if mode == "perf":
        _vals = list(df_["mean_sec"].dropna()) + [loop_perf[c] for c in loops if c in loop_perf]
        vmin, vmax = (np.percentile(_vals, [5, 95]) if _vals else (0.0, 1.0))
    else:
        vmin = vmax = 0.0

    def loop_label(c, n):
        return fmt_dur(loop_perf[c]) if mode == "perf" and c in loop_perf else f"{n:,}"

    def width_of(n):
        return round(st["min_width"] + (n / max_count) * (st["max_width"] - st["min_width"]), 2)

    arrow_len = max(st["arrow_min"], st["arrow_k"] * st["max_width"])
    margin = margin_px if margin_px is not None else max(4.0, st["corridor_margin"] * 0.6)
    margins = (margin, margin * 0.5, 1.5)
    wall_m = max(1.5, st["max_width"] / 2.0 + 1.5)
    corner = max(2.0, st["max_width"])
    gap = max(1.5 * st["max_width"] + 3.0, 5.0)
    door_off = wt / 2 + margins[0] + 2.0
    min_end = arrow_len * 1.1

    # ---- Top boxes (floor plan position based) and self-loop rectangles
    cont = [((j, c), tuple(bb)) for j, d in (container_positions or {}).items() for c, bb in d.items()]

    def anc(code):
        cx, cy = ctr(code)
        return frozenset(k for k, r in cont if r[0] < cx < r[0] + r[2] and r[1] < cy < r[1] + r[3])

    ancs = {c: anc(c) for c in positions}
    cont_raw, cont_keys = _arr([r for _, r in cont]), [k for k, _ in cont]

    # ---- self-loop edges: find a side (N/S) that does not intersect any other box or wall, 
    # and compute the rectangle for the loop label
    loop_side, loop_rects = {}, {}
    for c, n in loops.items():
        others = [r for k, r in positions.items() if k != c] + walls
        side = "N"
        for side in ("N", "S"):
            rect = _loop_rect_for_side(positions[c], n, st, width_of(n), side, loop_label(c, n))
            if not any(not (b[0] >= rect[0] + rect[2] or b[0] + b[2] <= rect[0] or
                            b[1] >= rect[1] + rect[3] or b[1] + b[3] <= rect[1]) for b in others):
                break
        loop_side[c] = side
        loop_rects[c] = _loop_rect_for_side(positions[c], n, st, width_of(n), side, loop_label(c, n))

    box_raw = _arr(leaf_rects + list(loop_rects.values()))
    wall_raw = _arr(walls + plugs)
    wall_block = _inf(wall_raw, wall_m)

    def stub_ok(P, A, own_zone):
        """Port->stub piece must not intersect with walls/doors and the stub end must remain within its own zone."""
        if len(wall_block) and a_hit(P, A, wall_block):
            return False
        return zone_of_point(A, geo) == own_zone

    cache = {}

    def obs(mg, allowed):
        ck = (mg, allowed)
        if ck not in cache:
            parts = [_inf(box_raw, mg), _inf(wall_raw, wall_m)]
            if allowed is not None and len(cont_raw):
                mk = np.array([k not in allowed for k in cont_keys])
                parts.append(_inf(cont_raw[mk], mg))
            cache[ck] = np.vstack([p for p in parts if len(p)])
        return cache[ck]

    lane_off, lane_w = {}, {}
    cur: dict[str, int | None] = {"i": None}

    def lane_for(d, used):
        o = lane_off.get((cur["i"], d["id"]))
        return o if o is not None else r4_lane(d, used, st["max_width"] + 2.5)

    # ---- zoned leg routing (door-to-door transitions)
    def zoned(A, B, ds, de, za, zb, arr, used, allow_hanan):
        def leg(P, Q, d0, d1):
            return r4_route_between(P, Q, d0, d1, arr, allow_hanan)

        if za == zb or not geo:
            return leg(A, B, ds, de)
        dA, dB = _pick_doors(A, B, za, zb, geo, used, door_off)
        wps, pairs = [[A, None, ds]], set()
        if dA:
            vA = _NORMAL[dA["side"]]
            nd = _VEC2DIR[(vA[0], vA[1])]
            i, o = door_passage(dA, lane_for(dA, used), door_off)
            wps.append([i, nd, None])
            pairs.add(len(wps) - 1)
            wps.append([o, None, nd])
        if dB:
            vB = _NORMAL[dB["side"]]
            nd = _VEC2DIR[(vB[0], vB[1])]
            i, o = door_passage(dB, lane_for(dB, used), door_off)
            wps.append([o, _OPP[nd], None])
            pairs.add(len(wps) - 1)
            wps.append([i, None, _OPP[nd]])
        wps.append([B, de, None])
        pts = [A]
        for k in range(len(wps) - 1):
            P, Q = wps[k], wps[k + 1]
            if k in pairs:
                pts.append(Q[0])
                continue
            seg = leg(P[0], Q[0], P[2], Q[1])
            if seg is None:
                return None
            pts.extend(seg[1:])
        return pts

    def route_edge(a, sa, ta, b, sb, tb, allow_hanan, used, use_forb=True):
        pa, pb = _port_xy(positions[a], sa, ta), _port_xy(positions[b], sb, tb)
        allowed = frozenset(ancs[a] | ancs[b]) if use_forb else None
        for mg in margins:
            arr = obs(mg, allowed)
            L = max(arrow_len * 1.2, mg + 3.0)
            A, B = _stub_xy(pa, sa, L), _stub_xy(pb, sb, L)
            if not (stub_ok(pa, A, zone[a]) and stub_ok(pb, B, zone[b])):
                continue          # This margin is breaking through the stub wall: try a margin with a shorter stub.
            u = dict(used)
            mid = zoned(A, B, sa, _OPP[sb], zone[a], zone[b], arr, u, allow_hanan)
            if mid:
                used.update(u)
                full = _simplify_points([pa] + mid + [pb])
                return full, _path_cost(full, sa, _OPP[sb])[0]
        return None

    # ================= 1. TRANSITION: selection of the front using only cheap (direct) candidates
    t1 = time.time()
    plan, tmp_used = [], {}
    for _, r in df_.iterrows():
        a, b, cnt = r["from"], r["to"], int(r["count"])
        srcs, dsts = _side_candidates(ctr(a), ctr(b))
        pref = [(s, t) for s in srcs for t in dsts]
        rest = [(s, t) for s in "NSWE" for t in "NSWE" if (s, t) not in pref]
        best = None
        for group in (pref, rest):
            for use_forb in (True, False):
                for s, t in group:
                    ta = _ideal_coord(s, positions[a], positions[b], corner)
                    tb = _ideal_coord(t, positions[b], positions[a], corner)
                    res = route_edge(a, s, ta, b, t, tb, False, tmp_used, use_forb)
                    if res and (best is None or res[1] < best["cost"]):
                        best = {"cost": res[1], "s": s, "t": t, "ta": ta, "tb": tb}
                if best:
                    break
            if best:
                break
        if best is None:                      # No direct route: approach along the dominant axis, via the second Hanan pass.
            s, t = srcs[0], dsts[0]
            best = {"s": s, "t": t, "ta": _ideal_coord(s, positions[a], positions[b], corner),
                    "tb": _ideal_coord(t, positions[b], positions[a], corner)}
        plan.append({"a": a, "b": b, "cnt": cnt, "mean": r.get("mean_sec", np.nan), **best})
    t2 = time.time()

    # ================= port spreading: same sides shared by multiple edges
    groups = defaultdict(list)
    for i, p in enumerate(plan):
        groups[(p["a"], p["s"])].append((i, "ta", p["b"]))
        groups[(p["b"], p["t"])].append((i, "tb", p["a"]))
    for (node, side), lst in groups.items():
        x, y, w, h = positions[node]
        lo, hi = ((x + corner, x + w - corner) if side in "NS" else (y + corner, y + h - corner))
        if node in loop_rects and side == loop_side[node]:
            hi = min(hi, loop_rects[node][0] - gap * 0.5)
        ax = 0 if side in "NS" else 1
        vals = [plan[i][k] + 1e-4 * ctr(partner)[ax] for i, k, partner in lst]
        for (i, k, _), t in zip(lst, r4_spread(vals, lo, max(lo, hi), gap)):
            plan[i][k] = t

    # ---- door pre-assignment: strip allocation based on thickness
    door_users = defaultdict(list)
    for i, p in enumerate(plan):
        a, b = p["a"], p["b"]
        if zone[a] == zone[b] or not geo:
            continue
        pa = _port_xy(positions[a], p["s"], p["ta"])
        pb = _port_xy(positions[b], p["t"], p["tb"])
        L = max(arrow_len * 1.2, margins[0] + 3.0)
        A, B = _stub_xy(pa, p["s"], L), _stub_xy(pb, p["t"], L)
        dA, dB = _pick_doors(A, B, zone[a], zone[b], geo, {}, door_off)
        for d in (dA, dB):
            if d:
                ax = 0 if d["side"] in "NS" else 1
                door_users[d["id"]].append((i, (A[ax] + B[ax]) / 2.0, d))
    for did, lst in door_users.items():
        d = lst[0][2]
        offs, ws, sc = r4_assign_lanes([(i, width_of(plan[i]["cnt"]), o) for i, o, _ in lst], d, 2.5)
        for i, _o, _d in lst:
            lane_off[(i, did)] = offs[i]
            lane_w[i] = min(lane_w.get(i, 1e9), ws[i])
        log.append(f"door {did}: {len(lst)} lines" + (f" | thickness x{sc:.2f} reduced" if sc < 1 else " | fits"))

    # ================= 2. TRANSITION: final port routing (Hanan only if necessary)
    used_doors, failed = {}, []
    for i, p in enumerate(plan):
        cur["i"] = i
        a, b = p["a"], p["b"]
        res = None
        for use_forb in (True, False):
            res = route_edge(a, p["s"], p["ta"], b, p["t"], p["tb"], True, used_doors, use_forb)
            if res:
                break
        if res is None:                       # last resort: other sides (with ideal ports) and Hanan pass
            for s in "NSWE":
                for t in "NSWE":
                    ta = _ideal_coord(s, positions[a], positions[b], corner)
                    tb = _ideal_coord(t, positions[b], positions[a], corner)
                    r2 = route_edge(a, s, ta, b, t, tb, True, used_doors, False)
                    if r2 and (res is None or r2[1] < res[1]):
                        res = r2
                        p.update({"s": s, "t": t, "ta": ta, "tb": tb})
        p["pts"] = res[0] if res else None
        if res is None:
            failed.append((a, b))
    t3 = time.time()

    # ================= strip separation (parallel, non-overlapping)
    ok = [p for p in plan if p["pts"] is not None]
    routes = [p["pts"] for p in ok]
    hcache = {}

    def hard_for(ri):
        key = frozenset(ancs[ok[ri]["a"]] | ancs[ok[ri]["b"]])
        if key not in hcache:
            parts = [_inf(box_raw, 1.0), _inf(wall_raw, wall_m)]
            if len(cont_raw):
                mk = np.array([k not in key for k in cont_keys])
                parts.append(_inf(cont_raw[mk], 1.0))
            hcache[key] = np.vstack([q for q in parts if len(q)])
        return hcache[key]

    fixed_arr = _arr(fixed_rects)
    pre_exact, pre_near = r4_measure(routes, gap * 0.5) if routes else (0.0, 0.0)
    moved, sep_stats = r4_separate(routes, hard_for, fixed_arr, gap, min_end) if routes else (0, {})
    for p, r in zip(ok, routes):
        p["pts"] = r
    exact, near = r4_measure(routes, gap * 0.5) if routes else (0.0, 0.0)
    if routes:
        log.extend(r4_overlap_report(routes, fixed_arr, gap, geo, wt))
        log.extend(r4_top_overlaps(routes, [(p["a"], p["b"]) for p in ok], fixed_arr))
        log.append(f"Separation obstacles: {sep_stats}")
    t4 = time.time()

    # ================= drawing
    def esc(v):
        return html.escape(str(v), quote=True)

    edges_svg, labels, out_tot, in_tot = [], [], defaultdict(int), defaultdict(int)
    for pi, p in enumerate(plan):
        a, b, cnt = p["a"], p["b"], p["cnt"]
        out_tot[a] += cnt
        in_tot[b] += cnt
        w = min(width_of(cnt), lane_w.get(pi, 1e9))
        has_perf = (mode == "perf") and pd.notna(p.get("mean"))
        color = duration_to_color(p["mean"], vmin, vmax) if has_perf else node_colors.get(a, "#5D6D7E")
        pts, extra = p["pts"], ""
        if pts is None:
            pa = _port_xy(positions[a], p["s"], p["ta"])
            pb = _port_xy(positions[b], p["t"], p["tb"])
            pts = _simplify_points(_orthogonalize(manhattan_fallback(pa, pb)))
            extra = ' stroke-dasharray="6 4"'
        p["final"] = pts
        trimmed, poly = _arrow_for(pts, w, st)
        d = f"M {trimmed[0][0]:.1f},{trimmed[0][1]:.1f} " + " ".join(f"L {x:.1f},{y:.1f}" for x, y in trimmed[1:])
        lx, ly, anchor = _label_anchor(pts, w, st)
        lbl = _label_svg(lx, ly, anchor, fmt_dur(p["mean"]) if has_perf else f"{cnt:,}", st)
        tip_extra = f" | avg. delay {fmt_dur(p['mean'])}" if has_perf else ""
        lbl_in_edge = lbl.replace("<text ", '<text class="lbl" ', 1)
        body = (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{w}"{extra} stroke-linejoin="miter" '
                f'stroke-linecap="butt" opacity="0.9"/><polygon points="{poly}" fill="{color}" opacity="0.9"/>')
        edges_svg.append(f'<g class="e" data-s="{esc(a)}" data-t="{esc(b)}">'
                         f'<title>{esc(a)} -> {esc(b)}\n{cnt:,} transitions{tip_extra}</title>{body}{lbl_in_edge}</g>')
        labels.append((cnt, a, b, lbl))
    for c, n in loops.items():
        lcol = (duration_to_color(loop_perf[c], vmin, vmax)
                if mode == "perf" and c in loop_perf else node_colors.get(c, "#5D6D7E"))
        lp, la, ll = draw_self_loop_svg_side(positions[c], n, lcol, st, width_of(n),
                                             loop_side[c], loop_label(c, n))
        edges_svg.append(f'<g class="e" data-s="{esc(c)}" data-t="{esc(c)}"><title>{esc(c)} -> {esc(c)}\n{n:,}</title>'
                         f'{lp}{la}</g>')
        labels.append((n, c, c, ll))
    shown = sorted([t for t in labels if t[1] != t[2]], key=lambda t: -t[0])
    shown = shown if label_top_n is None else shown[:int(label_top_n)]
    tl = [t[3].replace("<text ", f'<text class="tl" data-s="{esc(t[1])}" data-t="{esc(t[2])}" ', 1)
          for t in shown + [t for t in labels if t[1] == t[2]]]
    nodes_svg = []
    for c, (x, y, w, h) in positions.items():
        nm = (node_names or {}).get(c, "")
        tip = f"{c}" + (f" - {nm}" if nm else "") + f"\nOut: {out_tot.get(c, 0):,} | In: {in_tot.get(c, 0):,}"
        nodes_svg.append(f'<rect class="n" data-id="{esc(c)}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
                         f'height="{h:.1f}" rx="4" fill="transparent"><title>{esc(tip)}</title></rect>')

    # ================= VALIDATION
    good = [p for p in plan if p["pts"] is not None]
    nonortho = sum(1 for p in good for q, r_ in zip(p["final"], p["final"][1:])
                   if _sdir(q, r_) is None and math.dist(q, r_) > 1e-3)
    leaf_viol = [(p["a"], p["b"]) for p in good if box_crossings(p["final"], leaf_rects)]
    wall_bad = [(p["a"], p["b"]) for p in good if count_wall_crossings(p["final"], walls) > 0]
    door_cross = sum(1 for p in good if zone[p["a"]] != zone[p["b"]])
    nb = [_path_cost(p["final"], p["s"], _OPP[p["t"]])[1] for p in good]
    bend_hist = [sum(1 for b_ in nb if (b_ == k if k < 3 else b_ >= 3)) for k in range(4)]
    if wall_bad:
        log.append(f"Wall violations: {wall_bad[:10]}")
    if leaf_viol:
        log.append(f"Routes crossing boxes: {leaf_viol[:10]}")
    if failed:
        log.append(f"{len(failed)} edges have no box/wall-free path (dashed L): {failed[:5]}")
    log.append(f"Bend distribution (0/1/2/3+): {bend_hist}")
    log.append(f"Strip separation: {moved} segments | overlapping {pre_exact:.0f}px -> {exact:.0f}px | "
               f"near parallel {pre_near:.0f}px -> {near:.0f}px")
    log.append(f"TIME: 1st pass {t2 - t1:.1f}s | 2nd pass {t3 - t2:.1f}s | strip separation {t4 - t3:.1f}s | "
               f"total {time.time() - T0:.1f}s")

    tf = st["title_font"]
    title_svg = (f'<text x="{tf:.0f}" y="{tf * 1.3:.0f}" fill="black" font-family="Arial" font-size="{tf:.0f}" '
                 f'font-weight="bold">{html.escape(title)}</text>' if title else "")
    legend_svg = ""
    if mode == "perf":
        lf = st["font_size"] * 1.2
        lx0, ly0 = tf, tf * 1.3
        legend_svg = (
            '<defs><linearGradient id="gperf" x1="0" x2="1" y1="0" y2="0">'
            '<stop offset="0" stop-color="#2ecc71"/><stop offset=".5" stop-color="#f1c40f"/>'
            '<stop offset="1" stop-color="#e74c3c"/></linearGradient></defs>'
            f'<text x="{lx0:.0f}" y="{ly0 + lf:.0f}" fill="black" font-family="Arial" font-size="{lf:.0f}">'
            f'Line color = avg. delay · line width = frequency</text>'
            f'<rect x="{lx0:.0f}" y="{ly0 + lf * 2:.0f}" width="200" height="{lf:.0f}" fill="url(#gperf)"/>'
            f'<text x="{lx0:.0f}" y="{ly0 + lf * 4:.0f}" fill="black" font-family="Arial" font-size="{lf:.0f}">{fmt_dur(vmin)}</text>'
            f'<text x="{lx0 + 200:.0f}" y="{ly0 + lf * 4:.0f}" fill="black" font-family="Arial" font-size="{lf:.0f}" '
            f'text-anchor="end">{fmt_dur(vmax)}</text>'
        )    
    overlay = "\n".join([f"<style>{_CSS_FLOORPLAN}</style>", title_svg, legend_svg, '<g id="edges">'] + edges_svg +
                        ['</g><g id="nodes">'] + nodes_svg + ['</g><g id="tlabels" pointer-events="none">'] + tl +
                        ['</g><g id="hl"></g>', "<script><![CDATA[" + _JS_FLOORPLAN + "]]></script>"])
    content = _make_svg_responsive(content)
    end = content.rfind("</svg>")
    svg = content[:end] + overlay + "\n" + content[end:] if end >= 0 else content
    info = {
        "n_edges": len(df_), "n_loops": len(loops), "failed": len(failed),
        "door_cross": door_cross, "leaf_viol": len(leaf_viol), "wall_viol": len(wall_bad),
        "nonortho": nonortho, "lane_moves": moved, "overlap_exact_px": round(exact),
        "bends": bend_hist, "seconds": round(time.time() - T0, 1), "log": log,
    }
    return svg, info
