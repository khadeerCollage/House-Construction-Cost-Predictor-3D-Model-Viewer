"""
Smart Re-Plan Studio (intent-based editing)
===========================================
The editing section of the Design tab. Owners change the generated house by
saying what they want — no CAD skills, no dragging:

  ① tap a room on the house map (a 3×3 picture laid out like the drawing,
     road at the top)  ② tap what they want (move · other side · swap · next to
     · bigger · smaller · don't move) — most wishes are one tap; "move" asks for
     a square on a Vastu-coloured grid, "swap"/"next to" ask for a second room.
  … or type it in their own words (Indian English, spelling mistakes, Hindi /
     Malayalam / Tamil words — floorplan_generator.wish_parser); the studio shows
     what it understood and asks "did you mean …" when unsure.

Wishes stack up; "Re-plan my house" re-solves the whole layout around them
(floorplan_generator.intent_engine) and shows three complete options with why
each was chosen, wishes met, rooms that moved, Vastu and cost impact (with the
work items behind the price change). Choosing one rebuilds every drawing and
export, and the blueprint above updates; every applied version is kept in a
design & price history with one-click undo.
"""

import os
import re
import tempfile
from html import escape
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from floorplan_generator import intent_engine as ie
from floorplan_generator.vastu_engine import plan_sides_for_facing
from floorplan_generator.wish_parser import understand, suggest_wishes
from floorplan_renderer.svg_renderer import SVGRenderer
from floorplan_renderer.png_renderer import PNGRenderer
from floorplan_renderer.dxf_renderer import DXFRenderer
from floorplan_renderer.pdf_renderer import PDFRenderer
from model_folder.cost_engine import CostEngine

STACK = "intent_stack"
OPTIONS = "intent_options"
FEEDBACK = "intent_feedback"
HISTORY = "plan_history"                 # undo snapshots of every gen_* key
VERSIONS = "design_history"              # applied designs, for the price history
BASE_ID = "intent_base_plan_id"
SEL = "intent_sel"
PENDING = "intent_pending"
GEN_KEYS = ("gen_plan", "gen_cost", "gen_svg_clean", "gen_svg_blueprint", "gen_dxf_bytes",
            "gen_pdf_bytes", "gen_png_bytes", "gen_svg_bytes", "gen_mask")

ICONS = {"BEDROOM": "🛏️", "KITCHEN": "🍳", "BATHROOM": "🚿", "POOJA": "🪔", "LIVING": "🛋️",
         "DINING": "🍽️", "STUDY": "💻", "STORE": "📦", "UTILITY": "🧺", "STAIRS": "🪜",
         "LIFT": "🛗", "SERVANT": "🧹", "CORRIDOR": "🚪"}

ACTIONS = [("move", "📍 Move it"), ("opposite", "🔁 Other side"), ("swap", "⇄ Swap with…"),
           ("adjacent", "🤝 Next to…"), ("bigger", "➕ Bigger"), ("smaller", "➖ Smaller"),
           ("lock", "📌 Don't move")]

WHY = {
    "Balanced — recommended": "Meets your wishes with the fewest side-effects.",
    "Best Vastu": "The highest Vastu score that still meets your wishes.",
    "Least change": "Moves the fewest other rooms.",
    "Alternative": "Another valid arrangement to compare.",
    "Re-planned layout": "The best layout found for your wishes.",
}


# =============================================================================
# Pure helpers (no Streamlit) — also used by tests
# =============================================================================

def plan_supports_intents(plan) -> bool:
    return bool(getattr(plan, "generation_params", None)) and bool((getattr(plan, "layout_state", None) or {}).get("rooms"))


def _short(name: str) -> str:
    return name.split("/")[0].strip()


def _icon(info: Dict) -> str:
    return ICONS.get(info.get("category"), "▫️")


def _estimate(plan, cp: Dict[str, Any]) -> Optional[Dict]:
    try:
        return CostEngine.estimate_for_plan(plan, city_tier=cp.get("city_tier", "Tier-2 (Urban)"),
                                           quality_level=cp.get("quality", "Standard"),
                                           num_floors=cp.get("floors", getattr(plan, "num_floors", 1)),
                                           country=cp.get("country"), location=cp.get("location"))
    except Exception:
        return None


def _money(amount: float, country: Optional[str], signed: bool = False) -> str:
    txt = CostEngine.format_money(abs(amount), country or "India", compact=True)
    return (("+" if amount >= 0 else "-") + txt) if signed else txt     # ASCII '-': st.metric reads the sign


def cost_reasons(option_est: Optional[Dict], current_est: Optional[Dict], country: Optional[str]) -> List[str]:
    """Work items behind the price difference between two designs (biggest first)."""
    if not option_est or not current_est:
        return []
    o = {i["key"]: i for i in (option_est.get("layout_delta") or {}).get("items", [])}
    c = {i["key"]: i for i in (current_est.get("layout_delta") or {}).get("items", [])}
    out = []
    for key in set(o) | set(c):
        a, b = o.get(key, {}), c.get(key, {})
        amt = a.get("amount", 0.0) - b.get("amount", 0.0)
        dq = a.get("delta_qty", 0.0) - b.get("delta_qty", 0.0)
        if abs(amt) < 1:
            continue
        info = a or b
        unit = info.get("unit", "")
        qty = f"{dq:+.0f}" if unit == "nos" else f"{dq:+.1f} {unit}"
        out.append((abs(amt), f"{info.get('label', key)} {qty} ({_money(amt, country, True)})"))
    return [t for _, t in sorted(out, reverse=True)]


def thumbnail_svg(svg: str, plan, prefix: str = "opt") -> str:
    """
    Crop a sheet to the drawing itself (drops the title-block column), let CSS size it,
    and namespace drawsvg's auto ids so several inline SVGs on one page never share
    hatch patterns.
    """
    from floorplan_renderer.svg_renderer import PPM
    pad = 1.6 * PPM
    w, h = plan.plot_width * PPM + 2 * pad, plan.plot_height * PPM + 2 * pad
    head_end = svg.find(">", svg.find("<svg"))
    head = svg[:head_end]
    head = re.sub(r'\swidth="[^"]*"', "", head, count=1)
    head = re.sub(r'\sheight="[^"]*"', "", head, count=1)
    head = re.sub(r'viewBox="[^"]*"', f'viewBox="{-pad:.1f} {-pad:.1f} {w:.1f} {h:.1f}"', head, count=1)
    body = re.sub(r'(id="|url\(#|href="#)(d\d+)', lambda m: f"{m.group(1)}{prefix}{m.group(2)}", svg[head_end:])
    return head + body


def build_options(plan, intents: List[Dict], cost_params: Dict[str, Any], k: int = 3) -> List[Dict]:
    """Re-plan `plan` around `intents`; return k option bundles (plan, report, drawing, cost, deltas)."""
    from floorplan_generator.layout_engine import LayoutEngine
    engine = LayoutEngine(**dict(plan.generation_params), design_intents=list(intents),
                          base_layout=plan.layout_state)
    options = engine.generate_alternatives(k)
    current = _estimate(plan, cost_params)
    svg = SVGRenderer()
    out = []
    for i, p in enumerate(options):
        est = _estimate(p, cost_params)
        d_cost = (est or {}).get("total_cost", 0) - (current or {}).get("total_cost", 0)
        out.append({
            "plan": p, "report": p.edit_report,
            "svg": thumbnail_svg(svg.render_to_string(p, mode="blueprint"), p, prefix=f"opt{i}"),
            "cost": est, "d_cost": d_cost,
            "cost_reasons": cost_reasons(est, current, cost_params.get("country")),
            "d_vastu": round(p.vastu_score - plan.vastu_score, 1),
            "d_carpet": round(p.total_carpet_area - plan.total_carpet_area, 1),
        })
    return out


def render_exports(plan) -> Dict[str, Any]:
    """Every drawing / export for a plan, written to a private temp folder (no shared paths)."""
    out: Dict[str, Any] = {}
    svg = SVGRenderer()
    out["gen_svg_clean"] = svg.render_to_string(plan, mode="clean")
    out["gen_svg_blueprint"] = svg.render_to_string(plan, mode="blueprint")
    out["gen_svg_bytes"] = out["gen_svg_clean"].encode("utf-8")
    d = tempfile.mkdtemp(prefix="replan_")
    paths = {"gen_dxf_bytes": os.path.join(d, "blueprint.dxf"), "gen_pdf_bytes": os.path.join(d, "blueprint.pdf"),
             "gen_png_bytes": os.path.join(d, "blueprint.png")}
    DXFRenderer().render(plan, paths["gen_dxf_bytes"])
    PDFRenderer().render(plan, paths["gen_pdf_bytes"])
    png = PNGRenderer()
    png.render(plan, paths["gen_png_bytes"])
    for key, p in paths.items():
        with open(p, "rb") as f:
            out[key] = f.read()
    try:
        out["gen_mask"] = png.render_wall_mask(plan, ppm=50)
    except Exception:
        pass
    return out


# =============================================================================
# Session-state actions (button callbacks)
# =============================================================================

def _stack() -> List[Dict]:
    return st.session_state.setdefault(STACK, [])


def _feedback() -> List[Dict]:
    return st.session_state.setdefault(FEEDBACK, [])


def _push(intent: Dict) -> None:
    """Add a checked intent; a newer wish for the same room/kind replaces the older one."""
    stack = _stack()

    def same(a, b):
        if a["type"] == "mirror" and b["type"] == "mirror":
            return True
        return a.get("type") == b.get("type") and a.get("room") == b.get("room") and \
            {a.get("a"), a.get("b")} == {b.get("a"), b.get("b")}
    stack[:] = [s for s in stack if not same(s, intent)]
    if intent.get("room") and intent["type"] in ("zone", "flip", "lock"):
        stack[:] = [s for s in stack if not (s.get("room") == intent["room"] and s["type"] in ("zone", "flip", "lock"))]
    if intent.get("room") and intent["type"] == "size":
        stack[:] = [s for s in stack if not (s.get("room") == intent["room"] and s["type"] == "size")]
    stack.append(intent)
    st.session_state.pop(OPTIONS, None)


def _add(intent: Dict, ls: Dict) -> bool:
    n, why = ie.check_intent(intent, ls)
    if not n:
        _feedback().insert(0, {"kind": "warn", "text": why})
        return False
    _push(n)
    _feedback().insert(0, {"kind": "ok", "text": ie.describe_intent(n, ls)})
    del _feedback()[6:]
    return True


def _tap_room(key: str, ls: Dict) -> None:
    pend, sel = st.session_state.get(PENDING), st.session_state.get(SEL)
    if pend in ("swap", "adjacent") and sel and key != sel:
        _add({"type": pend, "a": sel, "b": key}, ls)
        st.session_state[PENDING] = None
        return
    st.session_state[SEL] = key
    st.session_state[PENDING] = None


def _act(kind: str, ls: Dict) -> None:
    sel = st.session_state.get(SEL)
    if not sel:
        return
    if kind in ("move", "swap", "adjacent"):
        st.session_state[PENDING] = None if st.session_state.get(PENDING) == kind else kind
        return
    st.session_state[PENDING] = None
    if kind == "opposite":
        it = ie.opposite_intent(ls, sel)
        if not it:
            _feedback().insert(0, {"kind": "warn", "text": f"{ls['rooms'][sel]['name']} has no reachable "
                                                          "'other side' in this layout."})
            return
        _add(it, ls)
    elif kind == "bigger":
        _add({"type": "size", "room": sel, "factor": 1.2}, ls)
    elif kind == "smaller":
        _add({"type": "size", "room": sel, "factor": 0.85}, ls)
    elif kind == "lock":
        _add({"type": "lock", "room": sel}, ls)


def _pick_zone(zone: str, ls: Dict) -> None:
    sel = st.session_state.get(SEL)
    if sel:
        _add({"type": "zone", "room": sel, "zones": [zone]}, ls)
    st.session_state[PENDING] = None


def _understand_text(ls: Dict) -> None:
    text = st.session_state.get("intent_text", "")
    if not text.strip():
        return
    fb: List[Dict] = []
    for r in understand(text, ls):
        fixes = ", ".join(f"“{a}” → {b}" for a, b in r["corrections"])
        for it in r["intents"]:
            n, why = ie.check_intent(it, ls)
            if n:
                _push(n)
                fb.append({"kind": "ok", "text": ie.describe_intent(n, ls), "fixes": fixes, "clause": r["clause"]})
            else:
                fb.append({"kind": "warn", "text": why})
        if not r["intents"]:
            sugg = [s for s in r["suggestions"] if ie.check_intent(s, ls)[0]]
            if sugg:
                fb.append({"kind": "ask", "clause": r["clause"], "note": r["note"], "suggestions": sugg,
                           "fixes": fixes})
            elif r["confidence"] == "high" and r["note"]:
                fb.append({"kind": "info", "text": r["note"]})
            else:
                fb.append({"kind": "warn", "text": r["note"] or f"Couldn't understand “{r['clause']}”."})
    st.session_state[FEEDBACK] = fb
    st.session_state["intent_text"] = ""


def _accept(i: int, j: int, ls: Dict) -> None:
    fb = _feedback()
    if 0 <= i < len(fb) and fb[i].get("kind") == "ask":
        sugg = fb[i]["suggestions"]
        if 0 <= j < len(sugg):
            fb.pop(i)
            _add(sugg[j], ls)


def _remove(i: int) -> None:
    stack = _stack()
    if 0 <= i < len(stack):
        stack.pop(i)
    st.session_state.pop(OPTIONS, None)


def _clear() -> None:
    st.session_state[STACK] = []
    st.session_state.pop(OPTIONS, None)
    st.session_state[FEEDBACK] = []
    st.session_state[PENDING] = None


def commit(option: Dict, cost_params: Dict[str, Any]) -> None:
    """Make an option the current design: rebuild every export, keep the old one for undo."""
    snap = {k: st.session_state.get(k) for k in GEN_KEYS if k in st.session_state}
    hist = st.session_state.setdefault(HISTORY, [])
    hist.append(snap)
    del hist[:-10]
    plan = option["plan"]
    ls = getattr(st.session_state.get("gen_plan"), "layout_state", {}) or {}
    wishes = [ie.describe_intent(i, ls) for i in _stack()]
    st.session_state.update(render_exports(plan))
    st.session_state["gen_plan"] = plan
    if option.get("cost"):
        st.session_state["gen_cost"] = option["cost"]
    versions = st.session_state.setdefault(VERSIONS, [])
    versions.append({"plan": plan, "plan_id": id(plan), "wishes": wishes,
                     "label": option["report"].get("label", "Option")})
    del versions[:-12]
    _reset_editor()
    st.session_state["intent_flash"] = f"Applied “{option['report'].get('label', 'option')}”. The blueprint above " \
                                       f"and all downloads now show your re-planned house."


def undo() -> None:
    hist = st.session_state.get(HISTORY) or []
    if not hist:
        return
    snap = hist.pop()
    for k, v in snap.items():
        st.session_state[k] = v
    versions = st.session_state.get(VERSIONS) or []
    if len(versions) > 1:
        versions.pop()
    _reset_editor()
    st.session_state["intent_flash"] = "Restored the previous design."


def _reset_editor() -> None:
    for k in ("editor_working_plan", "editor_initial_json", OPTIONS):
        st.session_state.pop(k, None)
    st.session_state[STACK] = []
    st.session_state[FEEDBACK] = []
    st.session_state[PENDING] = None
    st.session_state[SEL] = None


# =============================================================================
# UI
# =============================================================================

_CSS = """
<style>
.ir-card{background:rgba(0,229,255,.06);border:1px solid rgba(0,229,255,.25);border-left:4px solid #00E5FF;
  border-radius:8px;padding:12px 16px;margin:4px 0 14px 0;line-height:1.6}
.ir-card b{color:#00E5FF}
.ir-road{text-align:center;font-weight:700;letter-spacing:.08em;font-size:.82rem;color:#0A2342;
  background:#00E5FF;border-radius:6px;padding:4px 0;margin-bottom:6px}
.ir-zone{font-size:.72rem;color:#94A3B8;letter-spacing:.06em;margin-bottom:2px}
.ir-empty{color:#475569;font-size:.8rem;text-align:center;padding:6px 0}
.ir-chip{display:block;background:rgba(0,229,255,.08);border:1px solid rgba(0,229,255,.35);border-radius:6px;
  padding:7px 10px;font-size:.92rem}
.ir-opt h4{margin:0 0 4px 0;color:#00E5FF;font-size:1.02rem}
.ir-svg{background:#0A2342;border-radius:6px;overflow:hidden;margin:6px 0}
.ir-svg svg{width:100%;height:auto;display:block}
.ir-ok{color:#4ADE80}.ir-no{color:#F87171}.ir-warn{color:#FBBF24;font-size:.86rem}
.ir-muted{color:#94A3B8;font-size:.85rem}
.ir-hero{font-size:1.25rem;font-weight:700}
.ir-th{color:#94A3B8;font-size:.8rem;letter-spacing:.04em;text-transform:uppercase}
.ir-now{color:#0A2342;background:#00E5FF;border-radius:5px;padding:1px 6px;font-size:.72rem;font-weight:700;
  margin-left:6px;white-space:nowrap}
.ir-cell{font-size:.92rem}
.ir-chg{padding:4px 0;border-bottom:1px solid rgba(148,163,184,.15);font-size:.92rem}
.ir-view svg{width:100%;height:66vh}
.ir-view.half svg{height:52vh}
</style>
"""


def _fit_emoji(score: int, taboo: bool) -> str:
    if taboo:
        return "🔴"
    if score >= 5:
        return "🟢"
    if score >= 0:
        return "🟡"
    return "🟠"


def render_intent_studio(current_plan, cost_params: Dict[str, Any]) -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    flash = st.session_state.pop("intent_flash", None)
    if flash:
        st.success(flash)

    if not plan_supports_intents(current_plan):
        st.info("🧠 Smart Re-Plan needs an AI layout. Click **Generate** once more to enable it for this plan "
                "(very compact plots that use the simple fallback layout cannot be re-planned).")
        _undo_bar()
        return

    ls = current_plan.layout_state
    if st.session_state.get(BASE_ID) != id(current_plan):        # wishes belong to the plan they were made on
        st.session_state[BASE_ID] = id(current_plan)
        _reset_editor()
    versions = st.session_state.get(VERSIONS) or []
    if not versions or versions[-1]["plan_id"] != id(current_plan):   # a freshly generated design
        st.session_state[VERSIONS] = [{"plan": current_plan, "plan_id": id(current_plan), "wishes": [],
                                       "label": "Original design"}]
        st.session_state[HISTORY] = []                                # undo never jumps to an older project

    st.markdown(
        "<div class='ir-card'><b>Change your house the easy way.</b><br>"
        "① <b>Tap a room</b> on the map &nbsp;② <b>tap what you want</b> — or just <b>type it</b> in your own words. "
        "Then press <b>Re-plan my house</b>: you get 3 complete designs to compare. "
        "Nothing changes until you pick one, and you can always undo.</div>", unsafe_allow_html=True)

    col_map, col_talk = st.columns([1.25, 1], gap="large")
    with col_map:
        _house_map(ls)
        _actions(ls)
    with col_talk:
        _talk_box(ls)
        go = _wish_list(ls)

    if go and _stack():
        with st.spinner("Re-designing your house around your wishes (Vastu, NBC rules, doors, light, plumbing)…"):
            try:
                st.session_state[OPTIONS] = build_options(current_plan, _stack(), cost_params)
            except Exception as e:  # keep the editor alive; show what failed
                st.session_state.pop(OPTIONS, None)
                st.error(f"Re-plan failed: {e}")

    options = st.session_state.get(OPTIONS)
    if options:
        _render_options(options, current_plan, cost_params)
    _history_panel(cost_params)
    _undo_bar()


def _house_map(ls: Dict) -> None:
    facing = ls.get("facing", "N")
    road = ie.ZONE_NAMES.get(plan_sides_for_facing(facing)["top"], "")
    sel, pend = st.session_state.get(SEL), st.session_state.get(PENDING)
    head = "① Tap a room" if pend not in ("swap", "adjacent") else \
        f"👆 Now tap the room to {'swap with' if pend == 'swap' else 'put next to'} the " \
        f"{ls['rooms'].get(sel, {}).get('name', '')}"
    st.markdown(f"**{head}**")
    st.markdown(f"<div class='ir-road'>🛣️ ROAD · MAIN GATE ({road.upper()})</div>", unsafe_allow_html=True)
    by_zone: Dict[str, List[Tuple[str, Dict]]] = {}
    for k, v in ie.editable_rooms(ls):
        by_zone.setdefault(ie.current_zone(ls, k), []).append((k, v))
    for r, row in enumerate(ie.plan_grid(facing)):
        cols = st.columns(3, gap="small")
        for c, z in enumerate(row):
            with cols[c]:
                with st.container(border=True):
                    st.markdown(f"<div class='ir-zone'>{ie.ZONE_NAMES[z].split(' ')[0].upper()}</div>",
                                unsafe_allow_html=True)
                    rooms = by_zone.get(z, [])
                    if not rooms:
                        st.markdown("<div class='ir-empty'>—</div>", unsafe_allow_html=True)
                    for k, v in rooms:
                        st.button(f"{_icon(v)} {_short(v['name'])}", key=f"map_{k}", width="stretch",
                                  type="primary" if k == sel else "secondary",
                                  on_click=_tap_room, args=(k, ls))


def _actions(ls: Dict) -> None:
    sel = st.session_state.get(SEL)
    info = ls["rooms"].get(sel) if sel else None
    pend = st.session_state.get(PENDING)
    if info:
        st.markdown(f"**② What do you want for the {_icon(info)} {info['name']}?**")
    else:
        st.markdown("**② Then tap what you want**")
        st.caption("Tap a room on the map first.")
    rows = [ACTIONS[:4], ACTIONS[4:]]
    for row in rows:
        cols = st.columns(len(row), gap="small")
        for col, (kind, label) in zip(cols, row):
            col.button(label, key=f"act_{kind}", width="stretch", disabled=info is None,
                       type="primary" if pend == kind else "secondary", on_click=_act, args=(kind, ls))
    if info and pend == "move":
        _zone_grid(sel, ls)


def _zone_grid(key: str, ls: Dict) -> None:
    facing = ls.get("facing", "N")
    ok = set(ie.feasible_zones(ls, key))
    here = ie.current_zone(ls, key)
    st.caption("Tap where it should go (same layout as the map). "
               "🟢 best · 🟡 fine · 🟠 weak · 🔴 Vastu taboo · ⛔ not possible here")
    for r, row in enumerate(ie.plan_grid(facing)):
        cols = st.columns(3, gap="small")
        for c, z in enumerate(row):
            score, taboo = ie.zone_fit(ls, key, z)
            reach = z in ok
            mark = "⛔" if not reach else _fit_emoji(score, taboo)
            label = f"{mark} {ie.ZONE_NAMES[z]}" + ("  📍" if z == here else "")
            tip = f"Vastu fit {score:+d}" + (" (taboo)" if taboo else "") + ("" if reach else " — not possible")
            cols[c].button(label, key=f"iz_{r}_{c}", help=tip, disabled=not reach or z == here, width="stretch",
                           on_click=_pick_zone, args=(z, ls))


def _talk_box(ls: Dict) -> None:
    st.markdown("**💬 Or tell us in your own words**")
    st.text_area("Your wish", key="intent_text", height=90, label_visibility="collapsed",
                 placeholder="e.g.  make hall big · kitchen to north east · attached bathroom other side · "
                             "swap kids room and study · pooja near hall")
    st.caption("Simple English is fine — spelling mistakes are OK. Hindi / Malayalam / Tamil words work too "
               "(rasoi, adukkala, uttar, vadakku, bada, valiya…).")
    st.button("✨ Understand & add", key="ia_text", on_click=_understand_text, args=(ls,), width="stretch",
              type="primary")
    for i, f in enumerate(_feedback()):
        if f["kind"] == "ok":
            fx = f" <span class='ir-muted'>(read as: {f['fixes']})</span>" if f.get("fixes") else ""
            st.markdown(f"<div class='ir-ok'>✅ Added: {f['text']}{fx}</div>", unsafe_allow_html=True)
        elif f["kind"] == "warn":
            st.markdown(f"<div class='ir-warn'>⚠️ {f['text']}</div>", unsafe_allow_html=True)
        elif f["kind"] == "info":
            st.markdown(f"<div class='ir-muted'>ℹ️ {f['text']}</div>", unsafe_allow_html=True)
        else:
            fx = f" (read as: {f['fixes']})" if f.get("fixes") else ""
            st.markdown(f"❓ **“{f['clause']}”** — {f['note'] or 'did you mean'}{fx}")
            for j, s in enumerate(f["suggestions"][:4]):
                st.button(f"👉 {ie.describe_intent(s, ls)}", key=f"fb_{i}_{j}", on_click=_accept, args=(i, j, ls))
    have = {ie.describe_intent(i, ls) for i in _stack()}
    ideas = [s for s in suggest_wishes(ls) if ie.check_intent(s, ls)[0] and ie.describe_intent(s, ls) not in have]
    if ideas:
        st.markdown("**💡 Ideas for your house**")
        for j, s in enumerate(ideas):
            why = " (best Vastu)" if s.get("why") else ""
            st.button(f"{ie.describe_intent(s, ls)}{why}", key=f"idea_{j}", on_click=_add, args=(s, ls))


def _wish_list(ls: Dict) -> bool:
    st.markdown("**📝 Your wishes**")
    stack = _stack()
    if not stack:
        st.caption("No wishes yet.")
    for i, it in enumerate(stack):
        c1, c2 = st.columns([6, 1])
        c1.markdown(f"<span class='ir-chip'>{i + 1}. {ie.describe_intent(it, ls)}</span>", unsafe_allow_html=True)
        c2.button("✕", key=f"ia_rm_{i}", on_click=_remove, args=(i,), help="Remove this wish")
    q1, q2 = st.columns(2)
    q1.button("↔️ Mirror whole plan", key="ia_mirror", on_click=_add, args=({"type": "mirror"}, ls),
              width="stretch", help="Mirror the layout left ↔ right (the road side stays at the top)")
    q2.button("🧹 Clear all", key="ia_clear", on_click=_clear, width="stretch", disabled=not stack)
    return st.button("🔮 Re-plan my house", key="ia_go", type="primary", width="stretch", disabled=not stack)


def _render_options(options: List[Dict], current_plan, cost_params: Dict[str, Any]) -> None:
    country = cost_params.get("country")
    st.markdown("#### Choose a design")
    st.caption(f"Your current design: Vastu {current_plan.vastu_score:.0f}/100 · carpet "
               f"{current_plan.total_carpet_area:.1f} m². Nothing changes until you pick one.")
    cols = st.columns(len(options), gap="medium")
    for i, (col, op) in enumerate(zip(cols, options)):
        rep, p = op["report"], op["plan"]
        with col:
            met, tot = rep.get("satisfied", 0), rep.get("total", 0)
            label = rep.get("label", "Option")
            st.markdown(f"<div class='ir-opt'><h4>{chr(65 + i)} · {label}</h4>"
                        f"<span class='ir-muted'>{WHY.get(label, '')} · {met}/{tot} wishes met</span>"
                        f"<div class='ir-svg'>{op['svg']}</div></div>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Vastu", f"{p.vastu_score:.0f}/100", f"{op['d_vastu']:+.1f}")
            m2.metric("Carpet", f"{p.total_carpet_area:.1f} m²", f"{op['d_carpet']:+.1f} m²")
            lines = []
            if op.get("cost"):
                d = op["d_cost"]
                css = "ir-muted" if abs(d) < 1 else ("ir-no" if d > 0 else "ir-ok")
                change = "same as now" if abs(d) < 1 else f"{_money(d, country, True)} vs now"
                lines.append(f"<div>💰 <b>{op['cost']['display']['total_compact']}</b> "
                             f"<span class='{css}'>({change})</span></div>")
            for d in rep.get("intents", []):
                css, sym = ("ir-ok", "✅") if d["satisfied"] else ("ir-no", "❌")
                extra = f" — <i>not possible: {d['blocked_by']}</i>" if d.get("blocked_by") else ""
                lines.append(f"<div class='{css}'>{sym} {d['text']} <span class='ir-muted'>({d['detail']})</span>{extra}</div>")
            if op.get("cost_reasons"):
                lines.append("<div class='ir-muted' style='margin-top:6px'>Why the price changes:</div>")
                for t in op["cost_reasons"][:4]:
                    lines.append(f"<div class='ir-muted'>• {t}</div>")
            elif op.get("cost") and abs(op.get("d_cost", 0)) < 1:
                lines.append("<div class='ir-muted' style='margin-top:6px'>Price: same as your current design "
                             "(same walls, doors, windows and pipes).</div>")
            moved = [c for c in rep.get("changes", []) if not c["requested"]]
            if moved:
                lines.append("<div class='ir-muted' style='margin-top:6px'>Also moved to make room:</div>")
                for c in moved[:6]:
                    lines.append(f"<div class='ir-muted'>• {c['room']}: {ie.ZONE_NAMES.get(c['from'], c['from'])} → "
                                 f"{ie.ZONE_NAMES.get(c['to'], c['to'])}</div>")
            elif rep.get("intents"):
                lines.append("<div class='ir-muted' style='margin-top:6px'>No other room changed place.</div>")
            for w in rep.get("warnings", [])[:4]:
                lines.append(f"<div class='ir-warn'>⚠️ {w}</div>")
            st.markdown("".join(lines), unsafe_allow_html=True)
            st.button("✅ Use this design", key=f"ia_use_{i}", type="primary", width="stretch",
                      on_click=commit, args=(op, cost_params))


def _version_name(i: int) -> str:
    return "Original" if i == 0 else f"Edit {i}"


def _version_svg(v: Dict, prefix: str) -> str:
    """Blueprint drawing of a history version (rendered once, cropped/namespaced per use)."""
    if "svg_bp" not in v:
        v["svg_bp"] = SVGRenderer().render_to_string(v["plan"], mode="blueprint")
    return thumbnail_svg(v["svg_bp"], v["plan"], prefix=prefix)


def version_changes(old, new, limit: int = 10) -> List[str]:
    """What changed between two versions of the house, room by room, in plain words."""
    from floorplan_generator.vastu_engine import VastuEngine
    from floorplan_generator.house_report import ICON, ZONE_WORDS, is_site
    eo, en = VastuEngine.for_plan(old, scope="building"), VastuEngine.for_plan(new, scope="building")
    before = {r.name: r for r in old.rooms if not is_site(r)}
    after = {r.name: r for r in new.rooms if not is_site(r)}
    moved, resized = [], []
    for name, r in after.items():
        icon = ICON.get(r.category, "▫️")
        p = before.get(name)
        if p is None:
            moved.append((99.0, f"{icon} **{name}** — new"))
            continue
        za, zb = eo.get_zone_for_position(p.cx, p.cy).value, en.get_zone_for_position(r.cx, r.cy).value
        parts = []
        if za != zb:
            parts.append(f"moved {ZONE_WORDS.get(za, za)} → **{ZONE_WORDS.get(zb, zb)}**")
        if abs(r.area - p.area) >= 0.2:
            parts.append(f"{p.area:.1f} → **{r.area:.1f} m²** ({r.area - p.area:+.1f})")
        if not parts and abs(r.cx - p.cx) + abs(r.cy - p.cy) > 0.3:
            parts.append(f"shifted within the {ZONE_WORDS.get(zb, zb)}")
        if parts:
            (moved if za != zb else resized).append((abs(r.area - p.area), f"{icon} **{name}** — " + ", ".join(parts)))
    gone = [f"▫️ **{n}** — removed" for n in before if n not in after]
    # rooms that changed zone first, then the biggest size changes (the wish itself usually leads)
    out = [t for _, t in sorted(moved, key=lambda x: -x[0])] + [t for _, t in sorted(resized, key=lambda x: -x[0])] + gone
    return out[:limit] + ([f"…and {len(out) - limit} more"] if len(out) > limit else [])


def _restore(i: int, cost_params: Dict[str, Any]) -> None:
    versions = st.session_state.get(VERSIONS) or []
    if i >= len(versions):
        return
    v = versions[i]
    commit({"plan": v["plan"], "report": {"label": f"Back to the {_version_name(i)} design"},
            "cost": _estimate(v["plan"], cost_params)}, cost_params)
    versions = st.session_state.get(VERSIONS) or []
    if versions:
        versions[-1]["wishes"] = []
    st.session_state["intent_flash"] = (f"Back to the {_version_name(i)} design — the blueprint above and all "
                                        "downloads show it again (Undo is below).")


@st.dialog("👁️ Design version", width="large")
def _version_dialog(i: int, cost_params: Dict[str, Any]) -> None:
    versions = st.session_state.get(VERSIONS) or []
    if i >= len(versions):
        st.info("This version is no longer in the history.")
        return
    v = versions[i]
    plan = v["plan"]
    country = cost_params.get("country")
    current = v["plan"] is st.session_state.get("gen_plan")
    est = _estimate(plan, cost_params) or {}
    prev = versions[i - 1] if i > 0 else None
    prev_est = (_estimate(prev["plan"], cost_params) or {}) if prev else {}
    st.markdown(f"### {_version_name(i)}" + (" <span class='ir-now'>● YOUR DESIGN NOW</span>" if current else ""),
                unsafe_allow_html=True)
    st.caption("AI design" if i == 0 else ("; ".join(v["wishes"]) or v.get("label", "")))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cost", est.get("display", {}).get("total_compact", "—"))
    if prev:
        d = est.get("total_cost", 0.0) - prev_est.get("total_cost", 0.0)
        m2.metric(f"vs {_version_name(i - 1)}", _money(d, country, True))
    m3.metric("Vastu", f"{plan.vastu_score:.0f}/100",
              f"{plan.vastu_score - prev['plan'].vastu_score:+.0f}" if prev else None)
    m4.metric("Carpet", f"{plan.total_carpet_area:.1f} m²",
              f"{plan.total_carpet_area - prev['plan'].total_carpet_area:+.1f}" if prev else None)
    if prev is None:
        st.markdown(f"<div class='ir-svg ir-view'>{_version_svg(v, f'hv{i}')}</div>", unsafe_allow_html=True)
    else:
        t_this, t_cmp = st.tabs(["🖼️ This design", f"🔀 {_version_name(i - 1)} → {_version_name(i)}"])
        with t_this:
            st.markdown(f"<div class='ir-svg ir-view'>{_version_svg(v, f'hv{i}')}</div>", unsafe_allow_html=True)
        with t_cmp:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Before · {_version_name(i - 1)}**")
                st.markdown(f"<div class='ir-svg ir-view half'>{_version_svg(prev, f'hb{i}')}</div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"**After · {_version_name(i)}**")
                st.markdown(f"<div class='ir-svg ir-view half'>{_version_svg(v, f'ha{i}')}</div>", unsafe_allow_html=True)
        changes = version_changes(prev["plan"], plan)
        st.markdown("**What changed in the drawing**")
        st.markdown("\n".join(f"- {c}" for c in changes) if changes else "No room moved or changed size.")
    if not current:
        # a button inside a dialog reruns only the dialog — restore, then rerun the whole page
        # (the blueprint, history and tabs update and the dialog closes)
        if st.button(f"↩️ Make the {_version_name(i)} design my design again", key=f"ih_restore_{i}",
                     type="primary"):
            _restore(i, cost_params)
            st.rerun()


def _history_panel(cost_params: Dict[str, Any]) -> None:
    versions = st.session_state.get(VERSIONS) or []
    if len(versions) < 2:
        return
    country = cost_params.get("country")
    now = st.session_state.get("gen_plan")
    now_idx = max((i for i, v in enumerate(versions) if v["plan"] is now), default=-1)   # latest copy only
    rows = []
    for i, v in enumerate(versions):
        est = _estimate(v["plan"], cost_params) or {}
        rows.append({"Version": _version_name(i),
                     "What changed": "AI design" if i == 0 else ("; ".join(v["wishes"]) or v["label"]),
                     "total": est.get("total_cost", 0.0),
                     "Cost": est.get("display", {}).get("total_compact", "—"),
                     "Vastu": round(v["plan"].vastu_score),
                     "Carpet m²": round(v["plan"].total_carpet_area, 1),
                     "current": i == now_idx})
    for i, r in enumerate(rows):
        r["Change"] = "—" if i == 0 else _money(r["total"] - rows[i - 1]["total"], country, True)
    st.markdown("#### 📈 Your design & price history")
    first, last = rows[0], rows[-1]
    diff = last["total"] - first["total"]
    st.markdown(f"<div class='ir-hero'>{first['Cost']} → {last['Cost']} "
                f"<span class='ir-muted'>({_money(diff, country, True)} vs the original AI design)</span></div>",
                unsafe_allow_html=True)
    # one row per version; 👁️ opens that version's drawing (and before → after) without changing anything
    widths = [1.15, 3.0, 1.1, 1.0, 0.6, 0.8, 0.9]
    with st.container(border=True):
        head = st.columns(widths, vertical_alignment="center")
        for col, h in zip(head, ["Version", "What changed", "Cost", "Change", "Vastu", "Carpet m²", "Drawing"]):
            col.markdown(f"<div class='ir-th'>{h}</div>", unsafe_allow_html=True)
        for i, r in enumerate(rows):
            c = st.columns(widths, vertical_alignment="center")
            c[0].markdown(f"<div class='ir-cell'><b>{r['Version']}</b>"
                          + ("<span class='ir-now'>● now</span>" if r["current"] else "") + "</div>",
                          unsafe_allow_html=True)
            c[1].markdown(f"<div class='ir-cell'>{escape(r['What changed'])}</div>", unsafe_allow_html=True)
            c[2].markdown(f"<div class='ir-cell'>{r['Cost']}</div>", unsafe_allow_html=True)
            c[3].markdown(f"<div class='ir-cell'>{r['Change']}</div>", unsafe_allow_html=True)
            c[4].markdown(f"<div class='ir-cell'>{r['Vastu']}</div>", unsafe_allow_html=True)
            c[5].markdown(f"<div class='ir-cell'>{r['Carpet m²']}</div>", unsafe_allow_html=True)
            if c[6].button("👁️ View", key=f"ih_view_{i}", width="stretch",
                           help=f"See the {r['Version']} drawing" + ("" if i == 0 else " and what changed")):
                _version_dialog(i, cost_params)
    if len(rows) >= 3:
        import pandas as pd
        import altair as alt
        india = CostEngine.normalize_country(country) == "India"
        scale, unit = (1e5, "₹ Lakhs") if india else (1e3, "thousands")
        df = pd.DataFrame([{"Version": r["Version"], "Cost": round(r["total"] / scale, 2), "Total": r["Cost"],
                            "Vastu": r["Vastu"], "What changed": r["What changed"]} for r in rows])
        order = list(df["Version"])
        base = alt.Chart(df).encode(
            x=alt.X("Version:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Cost:Q", title=f"Estimated cost ({unit})", scale=alt.Scale(zero=False),
                    axis=alt.Axis(grid=True, gridOpacity=0.15)),
            tooltip=["Version", "Total", "Vastu", "What changed"])
        chart = base.mark_line(color="#00E5FF", strokeWidth=2) + base.mark_point(color="#00E5FF", filled=True, size=80)
        st.altair_chart(chart.properties(height=220), width="stretch")


def _undo_bar() -> None:
    hist = st.session_state.get(HISTORY) or []
    if hist:
        st.button(f"↩️ Undo last change ({len(hist)} saved)", key="ia_undo", on_click=undo)
