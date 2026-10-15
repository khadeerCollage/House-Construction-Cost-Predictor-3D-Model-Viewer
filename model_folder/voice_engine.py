"""
🗣️ The guide's voice — the app side of the neural voices (no ML imports here)
=============================================================================
The guided walk speaks English (Indian), Telugu or Hindi (floorplan_renderer/narration.py writes the
words). Who speaks them:

  • Piper with IIT Madras IndicTTS voices — Telugu: Venkatesh, Hindi: Rohan, English: Rohan reading
    English (an Indian accent). Fast on a laptop CPU (~5× real time): a whole walk in about a minute.
  • the browser's built-in voice (instant, robotic; also used for any sentence that is not ready yet)
  • the owner's own voice (IndicF5, from a ~10 s recording) — only when VOICE_OWN=1, i.e. on a computer
    with a GPU: on a laptop CPU it did not finish one sentence in 20 minutes (measured 2026-10-15).
    (Indic Parler-TTS was tried and dropped: ≈ 230 s for 7 s of speech here.)

The voices run in a separate environment (~/.vit_voice/venv, ≈ 190 MB, set up by setup_voice.bat) through
model_folder/voice_worker.py, started here in the background at low priority: sentences are spoken in
walking order into model_folder/walkthrough3d/voice/<voice>/<k>.wav, where the 3D viewer (a Streamlit
component served from walkthrough3d/) plays them. A clip's name is the sentence's key, so a re-planned
house only speaks the sentences that changed. Offline: nothing leaves the computer.
"""

import hashlib
import json
import os
import subprocess
import time
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
# the voice environment lives OUTSIDE the project (≈ 190 MB, Piper only; setup_voice.bat makes it)
VOICE_HOME = os.environ.get("VOICE_HOME") or os.path.join(os.path.expanduser("~"), ".vit_voice")
VENV_PY = os.path.join(VOICE_HOME, "venv", "Scripts" if os.name == "nt" else "bin",
                       "python.exe" if os.name == "nt" else "python")
WORKER = os.path.join(_HERE, "voice_worker.py")
CLIPS = os.path.join(_HERE, "walkthrough3d", "voice")      # served to the viewer as ./voice/…
DATA = os.path.join(_HERE, "voice_data")                   # the owner's recordings, jobs, logs (gitignored)
REFS = os.path.join(DATA, "refs")

PIPER_REPO = "rhasspy/piper-voices"
F5_REPO = "ai4bharat/IndicF5"
OWN_VOICE = False      # the owner's recorded voice (IndicF5) needs a GPU machine; off here, see the docstring

# Piper voices (the owner's choice, 2026-10-15): per language, (id, label, model path in rhasspy/piper-voices)
PIPER = {
    "te": [("piper-venkatesh", "Venkatesh (man)", "te/te_IN/venkatesh/medium/te_IN-venkatesh-medium")],
    "hi": [("piper-rohan", "Rohan (man)", "hi/hi_IN/rohan/medium/hi_IN-rohan-medium")],
    "en": [("piper-ryan", "Ryan (man, English)", "en/en_US/ryan/high/en_US-ryan-high")],   # Rohan reading English was poor
}

# what the owner reads for their own voice (≈ 10 s; IndicF5 needs the exact words of the clip)
REF_SENTENCE = {
    "te": "నమస్కారం అండీ. రండి, మన కొత్త ఇల్లు చూద్దాం. ఈశాన్యంలో పూజ గది, ఆగ్నేయంలో కిచెన్, అన్నీ వాస్తు ప్రకారమే ఉన్నాయి.",
    "hi": "नमस्ते जी। आइए, हमारा नया घर देखते हैं। ईशान कोण में पूजा घर, आग्नेय कोण में किचन, सब वास्तु के हिसाब से है।",
}
OWN_LANGS = tuple(REF_SENTENCE) if OWN_VOICE else ()

_PROC: Dict[str, object] = {}              # the one running worker: {"p": Popen, "dir": out_dir, "voice": id}


# ─────────────────────────────────────────────────────────────── what is installed
def _cached(repo: str, fn: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        p = try_to_load_from_cache(repo, fn)
        return bool(isinstance(p, str) and os.path.exists(p))
    except Exception:  # noqa: BLE001 — no hub library: nothing downloaded
        return False


def setup_state() -> Dict:
    """Environment and downloaded voices — read from disk, no model is loaded."""
    models = {m for vs in PIPER.values() for _, _, m in vs}
    return {"venv": os.path.exists(VENV_PY), "f5": _cached(F5_REPO, "model.safetensors"),
            "piper_models": {m: _cached(PIPER_REPO, m + ".onnx") for m in models}}


# ─────────────────────────────────────────────────────────────── voices
def own_voices(lang: str) -> List[Dict]:
    out = []
    if not os.path.isdir(REFS):
        return out
    for fn in sorted(os.listdir(REFS)):
        if not (fn.startswith(lang + "-") and fn.endswith(".json")):
            continue
        try:
            with open(os.path.join(REFS, fn), encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            continue
        vid = "own-" + fn[:-5]
        out.append({"id": vid, "lang": lang, "engine": "f5", "label": meta.get("name") or "My voice",
                    "ref_audio": os.path.join(REFS, fn[:-5] + ".wav"), "ref_text": meta.get("text", ""),
                    "created": meta.get("created", 0)})
    return sorted(out, key=lambda v: -v["created"])


def voices(lang: str) -> List[Dict]:
    """Every voice that can speak `lang`, best first: the owner's own, the named neural ones, the browser."""
    out = own_voices(lang) if lang in OWN_LANGS else []
    for vid, label, model in PIPER.get(lang, []):
        out.append({"id": vid, "lang": lang, "engine": "piper", "label": label, "model": model})
    out.append({"id": "browser", "lang": lang, "engine": "browser", "label": "Browser voice (instant, robotic)"})
    return out


def find(voice_id: str, lang: str) -> Optional[Dict]:
    return next((v for v in voices(lang) if v["id"] == voice_id), None)


def save_recording(lang: str, wav: bytes, text: str, name: str = "") -> str:
    """Keep the owner's clip (local only) and return its voice id."""
    os.makedirs(REFS, exist_ok=True)
    sha = hashlib.sha1(wav + text.encode("utf-8")).hexdigest()[:8]
    base = os.path.join(REFS, f"{lang}-{sha}")
    with open(base + ".wav", "wb") as f:
        f.write(wav)
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({"lang": lang, "text": text.strip(), "name": name or "My voice", "created": time.time()}, f,
                  ensure_ascii=False)
    return f"own-{lang}-{sha}"


def delete_recording(voice_id: str) -> None:
    """Forget an own voice: the recording and every clip spoken in it."""
    if not voice_id.startswith("own-"):
        return
    base = os.path.join(REFS, voice_id[4:])
    for ext in (".wav", ".json", ".clean.wav"):
        try:
            os.remove(base + ext)
        except OSError:
            pass
    d = clip_dir(voice_id)
    if os.path.isdir(d):
        for fn in os.listdir(d):
            try:
                os.remove(os.path.join(d, fn))
            except OSError:
                pass
        try:
            os.rmdir(d)
        except OSError:
            pass


def clip_dir(voice_id: str) -> str:
    return os.path.join(CLIPS, voice_id)


def ready(voice: Dict, setup: Optional[Dict] = None) -> Optional[str]:
    """None when this voice can be spoken here, else what is missing (plain words)."""
    if voice["engine"] == "browser":
        return None
    s = setup or setup_state()
    if not s["venv"]:
        return r"The voices are not installed yet — run .\setup_voice.bat once (about 320 MB, then offline)."
    if voice["engine"] == "piper" and not s["piper_models"].get(voice["model"]):
        return r"This voice is not downloaded yet — run .\setup_voice.bat once (about 130 MB)."
    if voice["engine"] == "f5" and not s["f5"]:
        return "The own-voice model (IndicF5) is not downloaded."
    return None


# ─────────────────────────────────────────────────────────────── speaking
def progress(voice: Dict, items: List[Dict]) -> Dict:
    """How many of these sentences have a clip, and what the worker says it is doing."""
    d = clip_dir(voice["id"])
    have = sum(1 for it in items if os.path.exists(os.path.join(d, it["k"] + ".wav")))
    st = {}
    try:
        with open(os.path.join(d, "_status.json"), encoding="utf-8") as f:
            st = json.load(f)
    except (OSError, ValueError):
        pass
    failed = set(st.get("failed") or [])
    gave_up = sum(1 for it in items if it["k"] in failed and not os.path.exists(os.path.join(d, it["k"] + ".wav")))
    p = _PROC.get("p")
    running = bool(p is not None and p.poll() is None and _PROC.get("voice") == voice["id"])
    left = [it for it in items if it["k"] not in failed and not os.path.exists(os.path.join(d, it["k"] + ".wav"))]
    return {"have": have, "failed": gave_up, "total": len(items), "running": running, "state": st.get("state"),
            "error": st.get("error"), "spc": st.get("sec_per_char"), "left": len(left),
            "left_chars": sum(len(it["t"]) for it in left)}


def stop() -> None:
    p = _PROC.get("p")
    if p is not None and p.poll() is None:
        try:
            open(os.path.join(_PROC["dir"], "_cancel"), "w").close()
            p.terminate()
        except OSError:
            pass
    _PROC.clear()


def ensure(voice: Dict, items: List[Dict]) -> Dict:
    """Start (or keep) the worker speaking every missing sentence of `items` in this voice."""
    if voice["engine"] == "browser" or ready(voice) is not None or not items:
        return progress(voice, items)
    d = clip_dir(voice["id"])
    failed = set()
    try:
        with open(os.path.join(d, "_status.json"), encoding="utf-8") as f:
            failed = set(json.load(f).get("failed") or [])          # the model could not say these: the browser will
    except (OSError, ValueError):
        pass
    missing = [it for it in items if it["k"] not in failed and not os.path.exists(os.path.join(d, it["k"] + ".wav"))]
    p = _PROC.get("p")
    if not missing:
        return progress(voice, items)
    if p is not None and p.poll() is None:
        if _PROC.get("voice") == voice["id"] and {it["k"] for it in missing} <= _PROC.get("all", set()):
            return progress(voice, items)                  # already on it
        stop()                                             # one worker at a time: the CPU is the limit
    os.makedirs(d, exist_ok=True)
    try:
        os.remove(os.path.join(d, "_cancel"))
    except OSError:
        pass
    os.makedirs(os.path.join(DATA, "jobs"), exist_ok=True)
    job = {"voice": {k: voice[k] for k in ("id", "engine", "model", "ref_audio", "ref_text") if k in voice},
           "out_dir": d, "items": items, "threads": max(1, min(4, (os.cpu_count() or 4) // 2))}
    job_path = os.path.join(DATA, "jobs", f"{voice['id']}.json")
    with open(job_path, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False)
    log = open(os.path.join(DATA, "jobs", f"{voice['id']}.log"), "w", encoding="utf-8")
    flags = 0
    if os.name == "nt":
        flags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    p = subprocess.Popen([VENV_PY, WORKER, "run", job_path], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                         creationflags=flags)
    _PROC.update(p=p, dir=d, voice=voice["id"], all={it["k"] for it in items})
    return progress(voice, items)


def eta_words(pr: Dict) -> str:
    """'about 6 minutes' left at the speed measured so far."""
    if not pr.get("spc") or not pr.get("left_chars"):
        return ""
    sec = pr["spc"] * pr["left_chars"]
    if sec < 90:
        return "about a minute"
    return f"about {round(sec / 60)} minutes"
