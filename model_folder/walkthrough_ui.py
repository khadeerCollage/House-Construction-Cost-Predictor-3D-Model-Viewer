"""
🚶 Walk through your house — the 3D section at the bottom of the Design tab
===========================================================================
The plan becomes a scene (floorplan_renderer/walkthrough.py: walls, doors, windows, stairs,
furniture, site, walking grid, guided tour), cached per design; the page itself is a Streamlit
component served from model_folder/walkthrough3d with Three.js vendored there — it works
offline and calls no outside service.

A wish made while standing in a room ("make it bigger", "move it to the other side", "keep it")
goes straight into the 🧠 Smart Re-Plan wish list above, so the owner can see 3 new designs.

The guided walk talks like a good broker (floorplan_renderer/narration.py) in English, Telugu or Hindi.
"🗣️ The guide's voice" picks the language and who speaks: a named neural voice (Indic Parler-TTS), the
owner's own recorded voice (IndicF5), or the browser's voice — model_folder/voice_engine.py has the
sentences spoken offline in the background; the viewer plays each one as soon as it is ready.
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
    """Scene JSON for this design (built once per design and location) and the guide's sentences per language."""
    key = (id(plan), state)
    hit = st.session_state.get(CACHE)
    if hit and hit[0] == key and len(hit) == 4:
        return hit[1], hit[2], hit[3]
    wt = importlib.import_module("floorplan_renderer.walkthrough")
    nar = importlib.import_module("floorplan_renderer.narration")
    data = wt.build_scene(plan, state=state)
    js = json.dumps(data, separators=(",", ":"))
    sid = hashlib.sha1(js.encode("utf-8")).hexdigest()[:16]
    lines = {lang: nar.speech_lines(data.get("tour") or [], lang) for lang in nar.LANGS}
    st.session_state[CACHE] = (key, js, sid, lines)
    return js, sid, lines


LANG = "walk_lang"
LANG_WORDS = {"en": "English (Indian)", "te": "తెలుగు — Telugu", "hi": "हिन्दी — Hindi"}
DEFAULT_VOICE = {"en": "piper-ryan", "hi": "piper-rohan", "te": "piper-venkatesh"}


def _voice_key(lang):
    return f"walk_voice_{lang}"


def _chosen_voice(ve, lang):
    """The voice picked for `lang` (the owner's own first, if they recorded one)."""
    vs = ve.voices(lang)
    ids = [v["id"] for v in vs]
    vid = st.session_state.get(_voice_key(lang))
    if vid not in ids:
        own = [v["id"] for v in vs if v["engine"] == "f5"]
        vid = own[0] if own else (DEFAULT_VOICE[lang] if DEFAULT_VOICE[lang] in ids else ids[0])
    return next(v for v in vs if v["id"] == vid)


def _voice_panel(lines) -> dict:
    """🗣️ Language and voice of the guide; returns what the viewer needs ({lang, voices})."""
    ve = importlib.import_module("model_folder.voice_engine")
    pend = st.session_state.pop("walk_lang_pending", None)
    if pend in LANG_WORDS:
        st.session_state[LANG] = pend
    pv = st.session_state.pop("walk_voice_pending", None)
    if pv:
        st.session_state[_voice_key(pv[0])] = pv[1]
    setup = ve.setup_state()
    lang = st.session_state.get(LANG, "en")
    with st.expander("🗣️ The guide's voice — English · తెలుగు · हिन्दी", expanded=False, key="walk_voice_expander"):
        st.caption("The guided walk explains your home the way a good broker would — Vastu, how it is built, and how it "
                   "fits your family — with every word measured from your drawing. Choose the language and who speaks. "
                   "Voices are made on this computer; nothing is sent anywhere.")
        lang = st.radio("Language", list(LANG_WORDS), format_func=LANG_WORDS.get, horizontal=True, key=LANG)
        vs = ve.voices(lang)
        cur = _chosen_voice(ve, lang)
        if st.session_state.get(_voice_key(lang)) != cur["id"]:
            st.session_state[_voice_key(lang)] = cur["id"]
        st.selectbox("Who speaks", [v["id"] for v in vs], key=_voice_key(lang),
                     format_func=lambda i: next((("🎙️ " if v["engine"] == "f5" else "🗣️ ") + v["label"]
                                                 for v in vs if v["id"] == i), i))
        cur = _chosen_voice(ve, lang)
        missing = ve.ready(cur, setup)
        if missing:
            st.info("🗣️ " + missing + " Until then the walk shows the words on screen and uses the browser's own voice "
                    "where it has one for this language.")
            with st.popover("How to set up the voices (once)"):
                st.markdown("In PowerShell, in this project folder:")
                st.code(r".\setup_voice.bat", language="bash")
                st.caption("The voices (Telugu Venkatesh, Hindi Rohan) are about 130 MB, no sign-in needed. After that "
                           "everything works offline.")
        elif cur["engine"] != "browser":
            ve.ensure(cur, lines.get(lang) or [])
            _progress_line(ve, cur, lines.get(lang) or [])
        if cur["engine"] == "f5":
            if st.button("🗑️ Forget this recorded voice", key=f"walk_forget_{lang}"):
                ve.delete_recording(cur["id"])
                st.session_state.pop(_voice_key(lang), None)
                st.rerun()
        if lang in ve.OWN_LANGS:
            _record_box(ve, lang)
    out = {}
    for l in LANG_WORDS:
        v = _chosen_voice(ve, l)
        neural = v["engine"] != "browser" and ve.ready(v, setup) is None
        pr = ve.progress(v, lines.get(l) or []) if neural else {}
        out[l] = {"url": f"voice/{v['id']}", "label": v["label"].split(" (")[0], "neural": neural,
                  "have": pr.get("have", 0), "total": pr.get("total", 0), "running": pr.get("running", False)}
    return {"lang": lang, "voices": out}


def _progress_line(ve, voice, items):
    """How far the voice has got — refreshed every few seconds while it is being made."""
    def body():
        pr = ve.progress(voice, items)
        if pr.get("error"):
            st.warning(f"🎙️ The voice could not be made: {pr['error']}")
        elif pr["total"] and not pr["left"]:
            who = voice['label'].split(' (')[0]
            st.success(f"🎙️ Ready — the whole walk is in {who}'s voice." if not pr["failed"] else
                       f"🎙️ Ready — {pr['have']} of {pr['total']} sentences in {who}'s voice; the model could not say "
                       f"{pr['failed']}, the browser's voice reads those.")
        else:
            eta = ve.eta_words(pr)
            st.progress(pr["have"] / max(1, pr["total"]),
                        text=f"🎙️ Making {voice['label'].split(' (')[0]}'s voice: {pr['have']} of {pr['total']} sentences"
                             + (f" · {eta} left" if eta else "") + ". The walk already plays the ready ones.")
    running = ve.progress(voice, items).get("running")
    st.fragment(body, run_every=4 if running else None)()


def _record_box(ve, lang):
    """The owner reads one sentence; IndicF5 then speaks the whole walk in that voice."""
    with st.container(border=True):
        st.markdown("**🎙️ Your own voice** — read this sentence aloud, clearly, about 10 seconds:")
        st.markdown(f"> {ve.REF_SENTENCE[lang]}")
        st.caption("Record only your own voice, or someone's who agreed. The recording stays on this computer.")
        rec = st.audio_input("Record", key=f"walk_rec_{lang}")
        text = st.text_input("The words you said (fix them if you said something different)", ve.REF_SENTENCE[lang],
                             key=f"walk_rec_text_{lang}")
        name = st.text_input("Name this voice", "My voice", key=f"walk_rec_name_{lang}")
        if rec is not None and st.button("✅ Use my voice for the walk", key=f"walk_rec_use_{lang}", type="primary"):
            vid = ve.save_recording(lang, rec.getvalue(), text or ve.REF_SENTENCE[lang], name)
            st.session_state["walk_voice_pending"] = (lang, vid)
            st.rerun()


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
    if not isinstance(ev, dict) or ev.get("action") not in ("wish", "lift", "lang"):
        return
    seen = st.session_state.setdefault(SEEN, [])
    if ev.get("id") in seen:                       # a component keeps returning its last value
        return
    seen.append(ev.get("id"))
    del seen[:-50]
    if ev.get("action") == "lang":                 # language picked in the 3D view: prepare that voice
        if ev.get("lang") in LANG_WORDS and ev.get("lang") != st.session_state.get(LANG):
            st.session_state["walk_lang_pending"] = ev["lang"]
            st.rerun()
        return
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
        js, sid, lines = _scene(plan, getattr(land, "state", None))
    except Exception as e:  # keep the page alive; the 3D view is an extra
        st.warning(f"The 3D house could not be built for this design ({e}).")
        return
    try:
        talk = _voice_panel(lines)
    except Exception as e:  # noqa: BLE001 — the walk still works with captions
        st.caption(f"🗣️ Voices unavailable ({e}).")
        talk = {"lang": st.session_state.get(LANG, "en"), "voices": {}}
    studio = importlib.import_module("model_folder.intent_studio")
    ev = _component(scene=js, sid=sid, height=height, can_wish=bool(studio.plan_supports_intents(plan)),
                    lang=talk["lang"], voices=talk["voices"], key="walk3d", default=None)
    if ev:
        _handle(plan, ev, cost_params)
