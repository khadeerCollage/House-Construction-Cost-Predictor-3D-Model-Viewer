"""
🎙️ Voice worker — speaks the guide's sentences with local neural voices (runs in the voice environment)
=======================================================================================================
The guided walk's words (floorplan_renderer/narration.py) become WAV clips here, offline, on this computer's
CPU, with rhasspy Piper (MIT; VITS exported to ONNX) and the IIT Madras IndicTTS voices the owner picked:
Telugu Venkatesh, Hindi Rohan, English Ryan.

Why Piper: measured 2026-10-15 on this i7-8665U for the same 7 s sentence — Piper 1.2-1.8 s (a whole walk
≈ 1 min per language) · ai4bharat/indic-parler-tts ≈ 230 s · ai4bharat/IndicF5 > 20 min (both dropped).

The environment lives OUTSIDE the project (~/.vit_voice/venv, ≈ 190 MB: piper-tts + onnxruntime + numpy +
huggingface_hub), made by setup_voice.bat, so the app's own environment and the repo stay untouched. The app
(model_folder/voice_engine.py) starts it:

    python voice_worker.py run JOB.json        speak every missing sentence of a job, in order
    python voice_worker.py check               what is installed / downloaded (JSON on stdout)
    python voice_worker.py download            fetch the voices once (public, ≈ 130 MB, no sign-in)

A job: {"voice": {"id", "engine": "piper", "model", "style"?}, "out_dir": ".../walkthrough3d/voice/<id>",
        "items": [{"k", "t"}, ...]}
Each sentence becomes <out_dir>/<k>.wav (16-bit mono, written to a temp name then renamed, so the viewer never
plays half a file); <out_dir>/_status.json tells the app how far it got. Nothing is sent anywhere: the voices
load from the local Hugging Face cache (HF_HUB_OFFLINE=1 while speaking).
"""

import json
import os
import sys
import time
import traceback

PIPER_REPO = "rhasspy/piper-voices"
PIPER_VOICES = ("te/te_IN/venkatesh/medium/te_IN-venkatesh-medium", "hi/hi_IN/rohan/medium/hi_IN-rohan-medium",
                "en/en_US/ryan/high/en_US-ryan-high")

# How the guide speaks (Piper's VITS knobs): a little slower than the recordings (clear for every owner),
# steadier intonation than the default (fewer wobbles on long numbers), a short breath between sentences.
STYLE = {"length_scale": 1.12, "noise_scale": 0.55, "noise_w_scale": 0.7}
TAIL_S = 0.22                                   # silence after each sentence


def _status(out_dir, **kw):
    path = os.path.join(out_dir, "_status.json")
    try:
        with open(path, encoding="utf-8") as f:
            st = json.load(f)
    except (OSError, ValueError):
        st = {}
    st.update(kw, updated=time.time(), pid=os.getpid())
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False)
    os.replace(tmp, path)


def _write_wav(path, audio, sr):
    """Edges trimmed, gentle fade in/out (no clicks), peak-normalised, a short breath after; 16-bit mono; atomic."""
    import wave

    import numpy as np

    a = np.asarray(audio)
    a = (a.astype(np.float32) / 32768.0 if a.dtype.kind == "i" else a.astype(np.float32)).reshape(-1)
    loud = np.flatnonzero(np.abs(a) > 0.01)
    if loud.size:
        a = a[max(0, loud[0] - int(0.04 * sr)): loud[-1] + int(0.10 * sr)]
    fade = min(len(a) // 4, int(0.012 * sr))
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        a[:fade] *= ramp
        a[-fade:] *= ramp[::-1]
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    if peak > 1e-4:
        a = a * (0.89 / peak)
    a = np.concatenate([a, np.zeros(int(TAIL_S * sr), dtype=np.float32)])
    pcm = (np.clip(a, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    tmp = path[:-4] + ".part.wav"
    with wave.open(tmp, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm)
    os.replace(tmp, path)


class Piper:
    """rhasspy Piper — ~5× faster than real time on a laptop CPU. Voice = a path in rhasspy/piper-voices."""

    def __init__(self, voice):
        from huggingface_hub import hf_hub_download
        from piper import PiperVoice

        hf_hub_download(PIPER_REPO, voice["model"] + ".onnx.json")
        self.voice = PiperVoice.load(hf_hub_download(PIPER_REPO, voice["model"] + ".onnx"))
        self.cfg = None
        try:
            from piper import SynthesisConfig

            self.cfg = SynthesisConfig(**dict(STYLE, **(voice.get("style") or {})))
        except (ImportError, TypeError):                 # an older piper: its own defaults
            pass

    def say(self, text):
        import io
        import wave

        import numpy as np

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            if self.cfg is not None:
                self.voice.synthesize_wav(text, w, syn_config=self.cfg)
            else:
                self.voice.synthesize_wav(text, w)
        buf.seek(0)
        with wave.open(buf) as w:
            sr = w.getframerate()
            a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return a, sr


ENGINES = {"piper": Piper}


def run(job_path):
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    out_dir = job["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    items = [it for it in job["items"] if not os.path.exists(os.path.join(out_dir, it["k"] + ".wav"))]
    total = len(job["items"])
    try:
        with open(os.path.join(out_dir, "_status.json"), encoding="utf-8") as f:
            failed = set(json.load(f).get("failed") or [])
    except (OSError, ValueError):
        failed = set()
    _status(out_dir, state="loading", done=total - len(items), total=total, error=None, started=time.time())
    if not items:
        _status(out_dir, state="done")
        return 0
    os.environ["HF_HUB_OFFLINE"] = "1"                 # speak from the local cache only
    try:
        eng = ENGINES[job["voice"]["engine"]](job["voice"])
    except Exception as e:  # noqa: BLE001 — reported to the app
        _status(out_dir, state="error", error=f"Could not load the voice: {type(e).__name__}: {e}",
                trace=traceback.format_exc()[-2000:])
        return 2
    done = total - len(items)
    secs, chars = 0.0, 0
    for it in items:
        if os.path.exists(os.path.join(out_dir, "_cancel")):
            _status(out_dir, state="cancelled")
            return 3
        _status(out_dir, state="running", current=it["t"][:80])
        t0 = time.time()
        try:
            audio, sr = eng.say(it["t"])
            _write_wav(os.path.join(out_dir, it["k"] + ".wav"), audio, sr)
        except Exception as e:  # noqa: BLE001 — one bad sentence must not stop the rest (the browser says it)
            failed.add(it["k"])
            _status(out_dir, last_error=f"{type(e).__name__}: {e}"[:300], failed=sorted(failed))
            continue
        if it["k"] in failed:
            failed.discard(it["k"])
            _status(out_dir, failed=sorted(failed))
        secs += time.time() - t0
        chars += len(it["t"])
        done += 1
        _status(out_dir, done=done, sec_per_char=round(secs / max(1, chars), 3))
    _status(out_dir, state="done", current=None)
    return 0


def check():
    out = {"python": sys.version.split()[0], "engines": {}, "models": {}}
    try:
        __import__("piper")
        out["engines"]["piper"] = True
    except Exception as e:  # noqa: BLE001
        out["engines"]["piper"] = f"{type(e).__name__}: {e}"[:200]
    try:
        from huggingface_hub import try_to_load_from_cache

        for v in PIPER_VOICES:
            p = try_to_load_from_cache(PIPER_REPO, v + ".onnx")
            out["models"][v.split("/")[2]] = bool(isinstance(p, str) and os.path.exists(p))
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)
    print(json.dumps(out))
    return 0


def download():
    from huggingface_hub import hf_hub_download

    print("Piper voices: Telugu Venkatesh, Hindi Rohan, English Ryan (≈250 MB)…", flush=True)
    for v in PIPER_VOICES:
        for ext in (".onnx", ".onnx.json"):
            hf_hub_download(PIPER_REPO, v + ext)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "run":
        sys.exit(run(sys.argv[2]))
    if cmd == "download":
        sys.exit(download())
    sys.exit(check())
