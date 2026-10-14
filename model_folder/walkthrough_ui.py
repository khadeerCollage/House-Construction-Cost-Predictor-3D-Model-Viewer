"""
🚶 Walk through your house — the 3D section at the bottom of the Design tab
===========================================================================
The plan becomes a scene (floorplan_renderer/walkthrough.py: walls, doors, windows, stairs,
furniture, site, walking grid, guided tour), cached per design; the page itself is a Streamlit
component served from model_folder/walkthrough3d with Three.js vendored there — it works
offline and calls no outside service.

A wish made while standing in a room ("make it bigger", "move it to the other side", "keep it")
goes straight into the 🧠 Smart Re-Plan wish list above, so the owner can see 3 new designs.
Moving the lift (inside the home / beside the main door / at the side) re-makes the design with the
lift there at once — a new version in the design history, with undo.
"""

import hashlib
import importlib
import json
import os

import streamlit as st
import streamlit.components.v1 as components

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "walkthrough3d")
_component = components.declare_component("house_walkthrough", path=_DIR)

CACHE = "walk3d_scene"
SEEN = "walk3d_seen_events"


def _scene(plan, state):
    """Scene JSON for this design (built once per design and location)."""
    key = (id(plan), state)
    hit = st.session_state.get(CACHE)
    if hit and hit[0] == key:
        return hit[1], hit[2]
    wt = importlib.import_module("floorplan_renderer.walkthrough")
    data = wt.build_scene(plan, state=state)
    js = json.dumps(data, separators=(",", ":"))
    sid = hashlib.sha1(js.encode("utf-8")).hexdigest()[:16]
    st.session_state[CACHE] = (key, js, sid)
    return js, sid


def _wish_from_event(plan, ev):
    """3D button → a Smart Re-Plan intent (or a plain-language reason it can't be one)."""
    ie = importlib.import_module("floorplan_generator.intent_engine")
    ls = getattr(plan, "layout_state", None) or {}
    key = ev.get("room")
    if not key or key not in (ls.get("rooms") or {}):
        return None, f"{ev.get('name', 'This room')} cannot be changed by Smart Re-Plan."
    kind = ev.get("kind")
    if kind == "bigger":
        return {"type": "size", "room": key, "factor": 1.2}, None
    if kind == "smaller":
        return {"type": "size", "room": key, "factor": 0.85}, None
    if kind == "lock":
        return {"type": "lock", "room": key}, None
    if kind == "opposite":
        it = ie.opposite_intent(ls, key)
        return (it, None) if it else (None, f"{ev.get('name', 'This room')} has no 'other side' it can move to "
                                            "in this layout.")
    return None, None


LIFT_WORDS = {"inside": "Lift inside the home", "entrance": "Lift outside, beside the main door",
              "side": "Lift outside, at the side of the house"}


def _move_lift(plan, pos, cost_params) -> None:
    """Re-make the design with the lift `pos` (same plot, rooms brief and amenities) and make it current."""
    if pos not in LIFT_WORDS:
        return
    params = dict(getattr(plan, "generation_params", None) or {})
    if not params:
        st.session_state["walk3d_note"] = ("warn", "This design cannot be re-made — click Generate once more.")
        return
    params.pop("lift_position", None)
    if pos != "inside":
        params["lift_position"] = pos
    le = importlib.import_module("floorplan_generator.layout_engine")
    studio = importlib.import_module("model_folder.intent_studio")
    with st.spinner("Re-making your design with the lift moved…"):
        new = le.LayoutEngine(**params).generate()
    got = ((getattr(new, "engineering", None) or {}).get("lift") or {}).get("position") or "inside"
    est = studio._estimate(new, cost_params) if cost_params else None
    studio.commit({"plan": new, "report": {"label": LIFT_WORDS.get(got, "Lift moved")}, "cost": est}, cost_params or {})
    if got in ("inside", "entrance", "side"):
        st.session_state["lift_position_pending"] = got  # the Design form shows the same choice (next run)
    if got != pos:
        why = next((n for n in new.design_notes if "lift could not" in n.lower()), "")
        st.session_state["walk3d_note"] = ("warn", f"🛗 The lift could not go there on this plot. {why}")
    else:
        st.session_state["walk3d_note"] = ("ok", f"🛗 {LIFT_WORDS[pos]} — the drawing, the cost and this 3D house now "
                                                 "show it. To go back: 🧠 Smart Re-Plan → Undo.")


def _handle(plan, ev, cost_params=None) -> None:
    if not isinstance(ev, dict) or ev.get("action") not in ("wish", "lift"):
        return
    seen = st.session_state.setdefault(SEEN, [])
    if ev.get("id") in seen:                       # a component keeps returning its last value
        return
    seen.append(ev.get("id"))
    del seen[:-50]
    if ev.get("action") == "lift":
        _move_lift(plan, ev.get("position"), cost_params)
        st.rerun()
        return
    studio = importlib.import_module("model_folder.intent_studio")
    if not studio.plan_supports_intents(plan):
        st.session_state["walk3d_note"] = ("warn", "Smart Re-Plan is not available for this layout — click Generate once more.")
        return
    if st.session_state.get(studio.BASE_ID) != id(plan):     # wishes belong to the design they were made on
        st.session_state[studio.BASE_ID] = id(plan)
        studio._reset_editor()
    intent, why = _wish_from_event(plan, ev)
    if intent is None:
        st.session_state["walk3d_note"] = ("warn", why or "That change is not possible here.")
        return
    if studio._add(intent, plan.layout_state):
        ie = importlib.import_module("floorplan_generator.intent_engine")
        text = ie.describe_intent(intent, plan.layout_state)
        st.session_state["intent_flash"] = (f"🚶 From your 3D walk: **{text}**. Add more wishes or press "
                                            "**Re-plan my house** to see 3 new designs.")
        st.session_state["walk3d_note"] = ("ok", f"Added to your wishes: {text}. Open **🧠 Smart Re-Plan** above "
                                                 "and press **Re-plan my house**.")
    else:
        fb = (st.session_state.get(studio.FEEDBACK) or [{}])[0]
        st.session_state["walk3d_note"] = ("warn", fb.get("text") or "That change is not possible here.")
    st.rerun()


def render_walkthrough(plan, land=None, height: int = 720, cost_params=None) -> None:
    st.markdown("### 🚶 Walk through your house — in 3D, before you build it")
    st.caption("Built from your drawing: the same walls, doors, windows, stairs and furniture at real size. "
               "Take the **guided walk** (it tells you about every room), walk on your own (drag to look, tap the "
               "floor to walk), see the **sunlight** at any hour and season, switch between the **blueprint** look "
               "and **real colours**, and change a room right where you stand. Works offline.")
    note = st.session_state.pop("walk3d_note", None)
    if note:
        (st.success if note[0] == "ok" else st.warning)(note[1])
    try:
        js, sid = _scene(plan, getattr(land, "state", None))
    except Exception as e:  # keep the page alive; the 3D view is an extra
        st.warning(f"The 3D house could not be built for this design ({e}).")
        return
    studio = importlib.import_module("model_folder.intent_studio")
    ev = _component(scene=js, sid=sid, height=height, can_wish=bool(studio.plan_supports_intents(plan)),
                    key="walk3d", default=None)
    if ev:
        _handle(plan, ev, cost_params)
