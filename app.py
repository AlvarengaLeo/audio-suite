"""
app.py — Majes de Sivar Audio Suite

Multi-service audio platform:
  1. Audio Enhance  — AI denoising + super-resolution
  2. Audio Extract  — Video → audio extraction at max quality
  3. Text-to-Audio  — Neural TTS with premium voices

Usage:
    python app.py
"""

import os
import re
import sys
import shutil
import logging
import tempfile
from datetime import datetime
import gradio as gr

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from core.pipeline import run_pipeline
from core.enhancer import is_model_available, is_resemble_available
from core.extractor import extract_audio, get_video_info, VIDEO_FORMATS
from core.tts import (
    synthesize_speech, available_engines, get_voices,
    engine_char_limit, openai_budget_remaining,
    SPEED_PRESETS, PITCH_PRESETS, OPENAI_STYLE_PRESETS,
    ENGINE_EDGE, ENGINE_OPENAI,
)
from core.voice_clone import clone_voice, is_clone_available

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# Cap outputs/ so a public Space's disk never fills up: by file count AND by
# total bytes (one big WAV extraction can dwarf hundreds of TTS mp3s).
MAX_OUTPUT_FILES = 60
MAX_OUTPUT_BYTES = 500 * 1024 * 1024  # ~500 MB


def prune_outputs(max_files: int = MAX_OUTPUT_FILES,
                  max_bytes: int = MAX_OUTPUT_BYTES):
    """Delete oldest files until outputs/ is under both the count and byte cap."""
    try:
        if not os.path.isdir(OUTPUTS_DIR):
            return
        files = [
            os.path.join(OUTPUTS_DIR, f) for f in os.listdir(OUTPUTS_DIR)
            if os.path.isfile(os.path.join(OUTPUTS_DIR, f))
        ]
        files.sort(key=lambda p: os.path.getmtime(p))  # oldest first
        total = sum(os.path.getsize(p) for p in files)
        while files and (len(files) > max_files or total > max_bytes):
            old = files.pop(0)
            try:
                total -= os.path.getsize(old)
                os.remove(old)
            except OSError:
                pass
    except OSError:
        pass


def _slug(name: str, fallback: str = "audio", maxlen: int = 40) -> str:
    """Clean a filename base: strip accents, keep [a-z0-9-_], collapse repeats."""
    import unicodedata
    name = unicodedata.normalize("NFKD", name or "")
    name = name.encode("ascii", "ignore").decode("ascii").lower()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    name = re.sub(r"_{2,}", "_", name)[:maxlen].strip("_")
    return name or fallback


def _friendly_error(e: Exception) -> str:
    """User-facing message: pass through our (Spanish) ValueErrors, otherwise
    a generic Spanish message so raw tracebacks/ffmpeg stderr never leak."""
    if isinstance(e, ValueError):
        return str(e)
    return ("Ocurrió un error al procesar el archivo. Verifica que el formato "
            "sea válido e inténtalo de nuevo.")


# ===================================================================
# HANDLERS
# ===================================================================

def handle_enhance(audio_input, ai_engine, enhancement_strength,
                   postprocess_preset, progress=gr.Progress()):
    """Audio Enhance handler."""
    if audio_input is None:
        gr.Warning("Sube un archivo de audio primero.")
        return None, None

    try:
        prune_outputs()

        def cb(v, m):
            progress(v, desc=m)

        output_path, _ = run_pipeline(
            audio_input,
            enhancement_strength=enhancement_strength,
            postprocess_preset=postprocess_preset,
            ai_engine=ai_engine,
            progress_callback=cb,
        )
        return output_path, output_path

    except Exception as e:
        logger.exception("Enhancement failed")
        raise gr.Error(_friendly_error(e))


def _format_video_info(info: dict) -> str:
    """Build a short Spanish summary line from get_video_info()."""
    try:
        secs = int(info.get("duration", 0) or 0)
        dur = f"{secs // 60}:{secs % 60:02d}" if secs else "—"
        size = info.get("size_mb", 0) or 0
        fmt = info.get("format", "Desconocido")
        return f"**Duración:** {dur}  ·  **Tamaño:** {size:.1f} MB  ·  **Formato:** {fmt}"
    except Exception:
        return ""


def handle_extract(video_input, out_format, progress=gr.Progress()):
    """Audio Extract handler."""
    if video_input is None:
        gr.Warning("Sube un archivo de video primero.")
        return None, None, ""

    try:
        progress(0.2, desc="Analizando video...")
        info = get_video_info(video_input)
        logger.info(f"Video info: {info}")

        # --- Guards (protect the free CPU tier before invoking ffmpeg) ---
        # Sin límite de tamaño: solo se acota la duración (hasta 4 horas).
        if info.get("duration", 0) > 14400:  # 4 horas
            raise ValueError(
                "El video supera el límite de 4 horas. "
                "Recorta o sube un clip más corto."
            )
        if not info.get("has_audio", True):
            raise ValueError("El video no contiene una pista de audio.")

        progress(0.4, desc=f"Extrayendo audio en {out_format.upper()} "
                           f"(puede tardar en videos largos)...")
        extracted = extract_audio(video_input, output_format=out_format)

        try:
            # Copy to outputs/ with a sanitized, clean filename.
            os.makedirs(OUTPUTS_DIR, exist_ok=True)
            prune_outputs()
            base = _slug(os.path.splitext(os.path.basename(video_input))[0])
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            final = os.path.join(OUTPUTS_DIR, f"{base}_extracted_{ts}.{out_format}")
            shutil.copy2(extracted, final)
        finally:
            # Don't leak the intermediate temp file into /tmp on every run.
            try:
                if os.path.exists(extracted):
                    os.remove(extracted)
            except OSError:
                pass

        progress(1.0, desc="Extraccion completada!")
        return final, final, _format_video_info(info)

    except Exception as e:
        logger.exception("Extraction failed")
        raise gr.Error(_friendly_error(e))


def handle_tts(text_input, engine, voice_name, speed,
               style_preset, custom_instructions, pitch_preset, master,
               progress=gr.Progress()):
    """Text-to-Audio handler (multi-engine)."""
    if not text_input or not text_input.strip():
        gr.Warning("Escribe algun texto para sintetizar.")
        return None, None

    try:
        progress(0.3, desc=f"Generando voz con {engine}...")
        output, used_engine = synthesize_speech(
            text_input,
            engine=engine,
            voice_name=voice_name,
            speed=speed,
            style_preset=style_preset,
            custom_instructions=custom_instructions,
            pitch_preset=pitch_preset,
            master=master,
        )

        # Tell the user if their chosen engine wasn't available and we fell back.
        if used_engine != engine:
            gr.Warning(
                f"El motor '{engine}' no estaba disponible; se usó "
                f"'{used_engine}'."
            )

        # Copy to outputs/, preserving the real extension (.wav for Kokoro).
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        prune_outputs()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = os.path.splitext(output)[1] or ".mp3"
        final = os.path.join(OUTPUTS_DIR, f"tts_output_{ts}{ext}")
        shutil.copy2(output, final)

        progress(1.0, desc="Voz generada!")
        return final, final

    except Exception as e:
        logger.exception("TTS failed")
        raise gr.Error(_friendly_error(e))


def handle_clone(reference, text, consent, master, progress=gr.Progress()):
    """Voice-cloning handler (OpenVoice V2, runs on CPU)."""
    if not consent:
        gr.Warning("Debes confirmar que tienes permiso para clonar esta voz.")
        return None, None
    if reference is None:
        gr.Warning("Sube un audio de referencia (30-60s de voz limpia).")
        return None, None
    if not text or not text.strip():
        gr.Warning("Escribe el texto que dirá la voz clonada.")
        return None, None
    if not is_clone_available():
        raise gr.Error(
            "El motor de clonación todavía no está disponible en este entorno "
            "(OpenVoice V2 se instala en el Space)."
        )
    try:
        progress(0.3, desc="Analizando y clonando la voz (CPU, ~15-40s)...")
        output = clone_voice(reference, text)

        if master:
            try:
                from core.postprocessor import finalize_voice
                output = finalize_voice(output)
            except Exception as e:
                logger.warning(f"finalize skipped: {e}")

        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        prune_outputs()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = os.path.splitext(output)[1] or ".wav"
        final = os.path.join(OUTPUTS_DIR, f"voz_clonada_{ts}{ext}")
        shutil.copy2(output, final)

        progress(1.0, desc="¡Voz clonada!")
        return final, final

    except Exception as e:
        logger.exception("Voice clone failed")
        raise gr.Error(_friendly_error(e))


def _char_count_md(text, engine):
    """Live 'N / limit caracteres' hint (turns red when over the limit)."""
    n = len(text or "")
    limit = engine_char_limit(engine)
    color = "#E0533A" if n > limit else "#8C8C9A"
    return f"<span style='color:{color};font-size:0.8rem'>{n:,} / {limit:,} caracteres</span>"


def _engine_info(engine, limit):
    """Dropdown helper text; for OpenAI, surface the remaining daily budget."""
    if engine == ENGINE_OPENAI:
        return (f"Premium (de pago, financiado por el dueño del Space). "
                f"Máximo {limit:,} caracteres. Hoy quedan ~{openai_budget_remaining():,}.")
    return f"Máximo {limit:,} caracteres para este motor."


def on_tts_engine_change(engine, text):
    """Update voice list + control visibility + counter when engine changes."""
    voices = get_voices(engine)
    limit = engine_char_limit(engine)
    is_openai = engine == ENGINE_OPENAI
    # Clear any stale custom instruction when leaving OpenAI so it can't quietly
    # steer a later run; leave it untouched while on OpenAI.
    instr_update = (gr.update(visible=True) if is_openai
                    else gr.update(visible=False, value=""))
    return (
        gr.update(choices=voices, value=(voices[0] if voices else None)),
        gr.update(visible=is_openai),          # style preset (OpenAI only)
        instr_update,                          # custom instructions (OpenAI only)
        gr.update(visible=(engine == ENGINE_EDGE)),  # pitch (Edge only)
        gr.update(visible=(not is_openai)),    # mastering (not for OpenAI)
        gr.update(info=_engine_info(engine, limit)),  # engine dropdown info
        _char_count_md(text, engine),          # char counter
    )


# ===================================================================
# UI CONSTRUCTION
# ===================================================================

def build_ui():
    """Build the multi-service Gradio UI with Midnight Luxe design."""

    ai_available = is_model_available()
    status_text = (
        "Modelo DeepFilterNet activo" if ai_available
        else "DeepFilterNet no disponible — respaldo FFmpeg"
    )
    status_color = "#00FF66" if ai_available else "#E0A33A"  # green vs amber

    # ------------------------------------------------------------------
    # CSS — Midnight Luxe + Tab Navigation
    # ------------------------------------------------------------------
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@1,400;1,600&display=swap');

    body {
        margin: 0;
        background-color: #0D0D12 !important;
        color: #FAF8F5 !important;
    }
    .gradio-container {
        max-width: 1200px !important;
        margin: auto;
        border-radius: 2rem !important;
        overflow: hidden;
        padding: 0 !important;
        background-color: transparent !important;
        box-shadow: 0 30px 80px rgba(0,0,0,0.6);
    }

    /* ── Noise overlay ── */
    .noise-bg {
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        pointer-events: none;
        z-index: 1;            /* decorative: stay below dropdowns/audio controls */
        opacity: 0.2;
        mix-blend-mode: overlay;
        background: url('data:image/svg+xml;utf8,%3Csvg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"%3E%3Cfilter id="n"%3E%3CfeTurbulence type="fractalNoise" baseFrequency="0.8" numOctaves="4" stitchTiles="stitch"/%3E%3C/filter%3E%3Crect width="100%25" height="100%25" filter="url(%23n)"/%3E%3C/svg%3E');
    }

    /* ── Card frame ── */
    .card-frame {
        background: #0D0D12;
        border: 1px solid rgba(201, 168, 76, 0.12);
        border-radius: 2rem;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5),
                    inset 0 1px 0 rgba(250,248,245,0.04);
    }

    /* ── Header / Navigation ── */
    .suite-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.5rem 2.5rem;
        background: linear-gradient(180deg, rgba(201,168,76,0.06) 0%, transparent 100%);
        border-bottom: 1px solid rgba(250,248,245,0.05);
    }
    .suite-brand {
        font-family: "Playfair Display", serif;
        font-style: italic;
        color: #C9A84C;
        font-size: 1.3rem;
        letter-spacing: 0.05em;
        white-space: nowrap;
    }
    .suite-status {
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: "JetBrains Mono", monospace;
        font-size: 0.72rem;
        color: #8C8C9A;
        letter-spacing: 0.05em;
    }
    .pulse-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #00FF66;
        box-shadow: 0 0 8px #00FF66;
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 0.6; transform: scale(0.9); }
        50%      { opacity: 1;   transform: scale(1.1); }
    }

    /* ── Tab navigation styling ── */
    .tabs > .tab-nav {
        background: transparent !important;
        border: none !important;
        display: flex !important;
        justify-content: center !important;
        gap: 0.5rem !important;
        padding: 1.5rem 2rem 0 !important;
    }
    .tabs > .tab-nav > button {
        background: transparent !important;
        border: 1px solid rgba(250,248,245,0.08) !important;
        border-radius: 2rem !important;
        color: #8C8C9A !important;
        font-family: "Inter", sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        padding: 0.75rem 1.8rem !important;
        transition: all 0.35s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
        letter-spacing: 0.02em !important;
    }
    .tabs > .tab-nav > button:hover {
        color: #FAF8F5 !important;
        border-color: rgba(201,168,76,0.3) !important;
        transform: translateY(-1px) !important;
    }
    .tabs > .tab-nav > button.selected {
        background: rgba(201,168,76,0.12) !important;
        border-color: #C9A84C !important;
        color: #C9A84C !important;
        box-shadow: 0 4px 20px rgba(201,168,76,0.15) !important;
    }

    /* ── Tab content area ── */
    .tabitem {
        background: transparent !important;
        border: none !important;
        padding: 2rem 2.5rem 3rem !important;
    }

    /* ── Section headers ── */
    .section-hero {
        text-align: center;
        margin-bottom: 2.5rem;
        padding-bottom: 2rem;
        border-bottom: 1px solid rgba(250,248,245,0.04);
    }
    .section-title {
        font-family: "Inter", sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #FAF8F5;
        letter-spacing: -0.02em;
        margin-bottom: 0.8rem;
    }
    .section-italic {
        font-family: "Playfair Display", serif;
        font-style: italic;
        color: #C9A84C;
    }
    .section-desc {
        font-family: "Inter", sans-serif;
        font-size: 1.05rem;
        color: #8C8C9A;
        max-width: 700px;
        margin: 0 auto;
        line-height: 1.7;
        font-weight: 300;
    }
    .section-desc strong {
        color: #8C8C9A;
        font-weight: 500;
    }

    /* ── Content area ── */
    .app-content {
        padding: 0;
        background-color: #0D0D12;
    }

    /* ── Buttons ── */
    button.primary {
        background: #C9A84C !important;
        color: #0D0D12 !important;
        font-weight: 600 !important;
        font-family: "Inter", sans-serif !important;
        border-radius: 2rem !important;
        border: none !important;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
        box-shadow: 0 4px 14px rgba(201,168,76,0.2) !important;
    }
    button.primary:hover {
        transform: scale(1.03) translateY(-2px) !important;
        box-shadow: 0 10px 25px rgba(201,168,76,0.4) !important;
        background: #E0C068 !important;
    }

    /* ── Centered action bar ── */
    .centered-action {
        max-width: 500px !important;
        margin: 2rem auto 0 auto !important;
    }

    /* ── Pipeline Diagram ── */
    .pipeline-container {
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid rgba(250,248,245,0.04);
    }
    .pipeline-title {
        color: #C9A84C;
        font-family: "Playfair Display", serif;
        font-style: italic;
        font-size: 1.4rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .pipeline-flow {
        display: flex;
        justify-content: space-between;
        align-items: stretch;
        gap: 1rem;
    }
    .pipeline-step {
        flex: 1;
        background: #111118;
        border: 1px solid rgba(201,168,76,0.08);
        border-radius: 1rem;
        padding: 1.2rem;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .pipeline-step:hover {
        transform: translateY(-4px);
        border-color: rgba(201,168,76,0.3);
    }
    .step-num {
        font-family: "JetBrains Mono", monospace;
        font-size: 0.75rem;
        color: #C9A84C;
        margin-bottom: 0.4rem;
        letter-spacing: 0.1em;
    }
    .step-name {
        font-size: 1rem;
        font-weight: 600;
        color: #FAF8F5;
        margin-bottom: 0.3rem;
    }
    .step-desc {
        font-size: 0.82rem;
        color: #8C8C9A;
        line-height: 1.4;
    }
    .pipeline-arrow {
        color: #C9A84C;
        font-size: 1.2rem;
        opacity: 0.4;
        display: flex;
        align-items: center;
    }
    .final-step {
        background: radial-gradient(circle at top left, rgba(201,168,76,0.08) 0%, #111118 100%);
        border-color: rgba(201,168,76,0.2);
    }
    @media (max-width: 768px) {
        .pipeline-flow { flex-direction: column; }
        .pipeline-arrow { justify-content: center; transform: rotate(90deg); }
        .suite-header { flex-direction: column; gap: 1rem; padding: 1rem 1.25rem; }
        /* Tighter padding + wrap so the 3 tabs and the form fit small screens */
        .tabitem { padding: 1.25rem 1rem 2rem !important; }
        .tabs > .tab-nav { flex-wrap: wrap !important; padding: 1rem 1rem 0 !important; }
        .tabs > .tab-nav > button { padding: 0.6rem 1.1rem !important; font-size: 0.85rem !important; }
        .section-title { font-size: 1.6rem !important; }
        .centered-action { max-width: 100% !important; }
    }

    /* ── Footer ── */
    .suite-footer {
        text-align: center;
        padding: 1.5rem 2rem;
        border-top: 1px solid rgba(250,248,245,0.04);
        font-family: "JetBrains Mono", monospace;
        font-size: 0.7rem;
        color: #6B6B7A;
        letter-spacing: 0.1em;
    }

    /* ════════════════════════════════════════════════════════════════
       CONTRAST SAFETY NET — guarantees no black-on-black text (WCAG AA)
       Palette: bg #0D0D12 · surface #16161F · surface-2 #1E1E2A
                text #F5F3EF · text-2 #A9A9B8 · accent #C9A84C / #E0C068
       ════════════════════════════════════════════════════════════════ */
    .gradio-container input,
    .gradio-container textarea,
    .gradio-container select,
    .gradio-container [contenteditable="true"] {
        color: #F5F3EF !important;
        background-color: #16161F !important;
    }
    .gradio-container input::placeholder,
    .gradio-container textarea::placeholder { color: #7E7E8C !important; }

    /* Dropdown popup list (a very common black-on-black offender) */
    .gradio-container ul.options,
    .gradio-container .options {
        background: #16161F !important;
        border: 1px solid rgba(201,168,76,0.25) !important;
    }
    .gradio-container ul.options li,
    .gradio-container .options .item { color: #F5F3EF !important; }
    .gradio-container .options .item:hover,
    .gradio-container .options .item.active,
    .gradio-container .options li.selected {
        background: #2A2A3A !important; color: #FFFFFF !important;
    }

    /* Labels / info / markdown */
    .gradio-container label,
    .gradio-container .label-wrap span { color: #CFCFDA !important; }
    .gradio-container .prose,
    .gradio-container .md,
    .gradio-container .markdown { color: #E8E6E1 !important; }
    .gradio-container .prose code { color: #E0C068 !important; background: rgba(201,168,76,0.10); }

    /* Icons (record / upload / trash / play…): they inherit `currentColor`,
       which was dark on dark — force them light so they're visible. */
    .gradio-container button svg,
    .gradio-container .icon-button svg,
    .gradio-container [class*="icon"] svg,
    .gradio-container .source-selection svg,
    .gradio-container label svg { color: #F5F3EF !important; opacity: 0.92; }
    .gradio-container button:hover svg { color: #FFFFFF !important; opacity: 1; }
    /* Audio/file widget helper text & timestamps */
    .gradio-container .info,
    .gradio-container .block-info,
    .gradio-container small,
    .gradio-container .wrap .or,
    .gradio-container .time,
    .gradio-container [class*="time"] { color: #A9A9B8 !important; }
    .gradio-container .upload-text,
    .gradio-container .wrap-text { color: #CFCFDA !important; }

    /* gr.Examples table */
    .gradio-container .examples table,
    .gradio-container .examples td,
    .gradio-container [class*="example"] td { color: #F5F3EF !important; }

    /* Visible focus ring for keyboard a11y */
    .gradio-container input:focus,
    .gradio-container textarea:focus,
    .gradio-container button:focus-visible,
    .gradio-container [tabindex]:focus-visible {
        outline: 2px solid rgba(201,168,76,0.6) !important;
        outline-offset: 2px;
    }

    /* ── "How to use" step strip (per-tab, easy to digest) ── */
    .how-to {
        display: flex; gap: 0.75rem; flex-wrap: wrap; justify-content: center;
        max-width: 860px; margin: 0 auto 2rem;
    }
    .how-step {
        display: flex; align-items: center; gap: 0.7rem; flex: 1; min-width: 210px;
        background: linear-gradient(180deg, rgba(201,168,76,0.05), rgba(255,255,255,0.01));
        border: 1px solid rgba(201,168,76,0.14);
        border-radius: 0.9rem; padding: 0.8rem 1rem;
        transition: border-color 0.25s ease, transform 0.25s ease;
    }
    .how-step:hover { border-color: rgba(201,168,76,0.35); transform: translateY(-2px); }
    .how-step .n {
        width: 28px; height: 28px; flex-shrink: 0; border-radius: 50%;
        background: rgba(201,168,76,0.16); color: #E0C068;
        display: flex; align-items: center; justify-content: center;
        font-family: "JetBrains Mono", monospace; font-size: 0.8rem; font-weight: 600;
    }
    .how-step .t { font-size: 0.86rem; color: #C9C9D6; line-height: 1.4; }
    .how-step .t b { color: #F5F3EF; font-weight: 600; }
    @media (max-width: 768px) { .how-to { flex-direction: column; } }

    /* ════════════ Ambient polish (lightweight, GPU transform/opacity) ════════════ */
    .gradio-container { position: relative; z-index: 1; }

    /* Aurora glow — dark luxe ambient light behind the content */
    .aurora {
        position: fixed; top: -15%; left: -10%; right: -10%; height: 65vh;
        z-index: 0; pointer-events: none; filter: blur(100px); opacity: 0.32;
        background:
          radial-gradient(40% 60% at 18% 22%, rgba(201,168,76,0.16), transparent 72%),
          radial-gradient(34% 52% at 84% 16%, rgba(224,192,104,0.10), transparent 72%);
        animation: aurora-drift 20s ease-in-out infinite alternate;
        will-change: transform;
    }
    @keyframes aurora-drift {
        0%   { transform: translate3d(-3%, -2%, 0) scale(1); }
        100% { transform: translate3d(3%, 3%, 0) scale(1.08); }
    }

    /* Glassmorphism feel on cards */
    .how-step, .pipeline-step {
        backdrop-filter: blur(8px) saturate(120%);
        -webkit-backdrop-filter: blur(8px) saturate(120%);
    }
    .pipeline-step:hover { box-shadow: 0 12px 30px rgba(0,0,0,0.45); }

    /* CSS audio-waveform accent (no JS) */
    .wave { display:inline-flex; align-items:flex-end; gap:3px; height:18px; vertical-align:middle; }
    .wave span {
        width:3px; height:6px; border-radius:2px;
        background:linear-gradient(180deg,#E0C068,#C9A84C);
        animation: wavebar 1.1s ease-in-out infinite;
    }
    .wave span:nth-child(2){animation-delay:.15s} .wave span:nth-child(3){animation-delay:.30s}
    .wave span:nth-child(4){animation-delay:.45s} .wave span:nth-child(5){animation-delay:.60s}
    @keyframes wavebar { 0%,100%{height:5px;opacity:.55} 50%{height:18px;opacity:1} }

    @media (prefers-reduced-motion: reduce) {
        .aurora { animation: none; }
        .wave span { animation: none; height: 11px; }
        .pulse-dot { animation: none; }
    }

    footer { display: none !important; }
    """

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------
    gradio_theme = gr.themes.Base(
        primary_hue=gr.themes.colors.Color(
            c50="#FDF8E8", c100="#FAF0C9", c200="#F5E4A0", c300="#F0D571",
            c400="#EBC54D", c500="#C9A84C", c600="#A3883C", c700="#7A662C",
            c800="#52441D", c900="#29220E", c950="#141107"
        ),
        neutral_hue=gr.themes.colors.Color(
            c50="#2A2A35", c100="#2A2A35", c200="#2A2A35", c300="#2A2A35",
            c400="#2A2A35", c500="#1A1A22", c600="#14141B", c700="#101015",
            c800="#0D0D12", c900="#0A0A0E", c950="#050507"
        ),
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace"],
    ).set(
        body_background_fill="#0D0D12",
        body_background_fill_dark="#0D0D12",
        body_text_color="#FAF8F5",
        body_text_color_dark="#FAF8F5",
        background_fill_primary="#0D0D12",
        background_fill_primary_dark="#0D0D12",
        background_fill_secondary="#101015",
        background_fill_secondary_dark="#101015",
        border_color_accent="#C9A84C",
        border_color_primary="rgba(250,248,245,0.1)",
        border_color_primary_dark="rgba(250,248,245,0.1)",
        button_primary_background_fill="#C9A84C",
        button_primary_background_fill_dark="#C9A84C",
        button_primary_text_color="#0D0D12",
        button_primary_text_color_dark="#0D0D12",
        block_background_fill="#14141C",
        block_background_fill_dark="#14141C",
        block_border_color="rgba(250,248,245,0.1)",
        block_border_color_dark="rgba(250,248,245,0.1)",
        panel_background_fill="#0D0D12",
        panel_background_fill_dark="#0D0D12",
        # --- Inputs & labels: fix black-on-black, ensure WCAG AA contrast ---
        input_background_fill="#16161F",
        input_background_fill_dark="#16161F",
        input_background_fill_focus="#1E1E2A",
        input_background_fill_focus_dark="#1E1E2A",
        input_border_color="rgba(250,248,245,0.14)",
        input_border_color_dark="rgba(250,248,245,0.14)",
        input_border_color_focus="#C9A84C",
        input_border_color_focus_dark="#C9A84C",
        block_label_text_color="#CFCFDA",
        block_label_text_color_dark="#CFCFDA",
        block_title_text_color="#F5F3EF",
        block_title_text_color_dark="#F5F3EF",
        body_text_color_subdued="#A9A9B8",
        body_text_color_subdued_dark="#A9A9B8",
        block_info_text_color="#A9A9B8",
        block_info_text_color_dark="#A9A9B8",
    )

    # ------------------------------------------------------------------
    # Build Blocks
    # ------------------------------------------------------------------
    with gr.Blocks(title="Majes de Sivar — Audio Suite") as app:

        # Noise overlay
        gr.HTML('<div class="aurora"></div>')

        # ── Card frame start ──
        # Header with brand + status (dot color reflects real availability)
        gr.HTML(f"""
            <div class="suite-header">
                <span class="suite-brand">
                    <span class="wave" aria-hidden="true"><span></span><span></span><span></span><span></span><span></span></span>
                    &nbsp;Majes de Sivar
                </span>
                <span class="suite-status" role="status" aria-label="Estado: {status_text}">
                    <span class="pulse-dot" style="background:{status_color};box-shadow:0 0 8px {status_color};"></span>
                    {status_text}
                </span>
            </div>
        """)

        # ── Tab navigation (the carousel) ──
        with gr.Tabs():

            # ════════════════════════════════════════════
            #  TAB 1 — AUDIO ENHANCE
            # ════════════════════════════════════════════
            with gr.Tab("Audio Enhance"):
                gr.HTML("""
                    <div class="section-hero">
                        <h2 class="section-title">
                            Audio impecable e
                            <span class="section-italic">Inteligencia Quirúrgica.</span>
                        </h2>
                        <p class="section-desc">
                            Transforma grabaciones ordinarias en audio de estudio
                            con IA (<strong>DeepFilterNet</strong>) y una cadena de
                            <strong>masterización profesional</strong>.
                            Procesamiento <strong>100% local</strong> — cero latencia,
                            máxima privacidad, control milimétrico sin destruir tu voz.
                        </p>
                    </div>
                """)

                gr.HTML("""
                    <div class="how-to">
                        <div class="how-step"><span class="n">1</span><span class="t"><b>Sube tu audio</b> de voz (WAV, MP3, M4A…).</span></div>
                        <div class="how-step"><span class="n">2</span><span class="t">Opcional: ajusta <b>intensidad</b> y <b>preset</b> en Configuración avanzada.</span></div>
                        <div class="how-step"><span class="n">3</span><span class="t">Pulsa <b>Mejorar Audio</b>, compara antes/después y descarga.</span></div>
                    </div>
                """)

                # Only offer engines that are actually installed (mirrors TTS).
                _enh_engines = ["DeepFilterNet"] + (
                    ["Resemble Enhance"] if is_resemble_available() else []
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        enh_input = gr.Audio(
                            label="Subir Audio",
                            sources=["upload"],
                            type="filepath",
                            elem_id="enh-input",
                        )
                        gr.Markdown(
                            "*Formatos:* `WAV` `MP3` `M4A` `FLAC` `OGG` `AAC` `WMA` · máx. 500 MB\n\n"
                            "*La salida se optimiza como voz mono a 48 kHz.*"
                        )
                        enh_engine = gr.Radio(
                            label="Motor de IA",
                            choices=_enh_engines,
                            value="DeepFilterNet",
                            visible=(len(_enh_engines) > 1),
                            elem_id="enh-engine",
                        )

                    with gr.Column(scale=1):
                        enh_original = gr.Audio(
                            label="Original (antes)",
                            type="filepath",
                            interactive=False,
                            elem_id="enh-original",
                        )
                        enh_output = gr.Audio(
                            label="Audio Mejorado (después)",
                            type="filepath",
                            interactive=False,
                            elem_id="enh-output",
                        )
                        enh_download = gr.File(
                            label="Descargar audio mejorado",
                            elem_id="enh-download",
                        )

                # Load the uploaded file into the "before" player for A/B compare.
                enh_input.change(lambda p: p, inputs=enh_input, outputs=enh_original)

                with gr.Column(elem_classes="centered-action"):
                    with gr.Accordion("Configuración avanzada", open=False):
                        enh_strength = gr.Dropdown(
                            label="Intensidad",
                            choices=["gentle", "balanced", "aggressive",
                                     "maximum"],
                            value="balanced",
                        )
                        enh_preset = gr.Dropdown(
                            label="Preset de masterización",
                            choices=["natural", "podcast", "broadcast"],
                            value="natural",
                        )
                    enh_btn = gr.Button(
                        "Mejorar Audio",
                        variant="primary",
                        size="lg",
                    )

                enh_btn.click(
                    fn=handle_enhance,
                    inputs=[enh_input, enh_engine, enh_strength, enh_preset],
                    outputs=[enh_output, enh_download],
                )

                # Pipeline diagram
                gr.HTML("""
                    <div class="pipeline-container">
                        <h3 class="pipeline-title">Arquitectura de Procesamiento</h3>
                        <div class="pipeline-flow">
                            <div class="pipeline-step">
                                <div class="step-num">01</div>
                                <div class="step-name">Conversión</div>
                                <div class="step-desc">48kHz mono WAV. Preparación matemática de la señal.</div>
                            </div>
                            <div class="pipeline-arrow">➔</div>
                            <div class="pipeline-step">
                                <div class="step-num">02</div>
                                <div class="step-name">Motor de IA</div>
                                <div class="step-desc"><strong>DeepFilterNet</strong>. Denoising neuronal que limpia el ruido sin destruir la voz.</div>
                            </div>
                            <div class="pipeline-arrow">➔</div>
                            <div class="pipeline-step">
                                <div class="step-num">03</div>
                                <div class="step-name">Masterización</div>
                                <div class="step-desc">EQ paramétrica + Compresión multi-banda + Limitador.</div>
                            </div>
                            <div class="pipeline-arrow">➔</div>
                            <div class="pipeline-step final-step">
                                <div class="step-num">04</div>
                                <div class="step-name">Audio Final</div>
                                <div class="step-desc">Lista de emisión. <strong>0 Latencia en Nube</strong>.</div>
                            </div>
                        </div>
                    </div>
                """)

            # ════════════════════════════════════════════
            #  TAB 2 — AUDIO EXTRACT
            # ════════════════════════════════════════════
            with gr.Tab("Audio Extract"):
                gr.HTML("""
                    <div class="section-hero">
                        <h2 class="section-title">
                            Extracción de audio con
                            <span class="section-italic">Fidelidad Absoluta.</span>
                        </h2>
                        <p class="section-desc">
                            Sube cualquier archivo de video y obtén la pista de audio
                            en alta fidelidad: <strong>WAV/FLAC 24-bit / 48kHz</strong>
                            (sin pérdida) o <strong>MP3</strong> (ligero).
                            Potenciado por <strong>FFmpeg</strong>.
                        </p>
                    </div>
                """)

                gr.HTML("""
                    <div class="how-to">
                        <div class="how-step"><span class="n">1</span><span class="t"><b>Sube un video</b> (MP4, MKV, MOV… máx. 500 MB / 30 min).</span></div>
                        <div class="how-step"><span class="n">2</span><span class="t">Elige el <b>formato</b>: WAV/FLAC (sin pérdida) o MP3 (ligero).</span></div>
                        <div class="how-step"><span class="n">3</span><span class="t">Pulsa <b>Extraer Audio</b> y descarga la pista.</span></div>
                    </div>
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        ext_input = gr.File(
                            label="Subir Video",
                            file_types=[".mp4", ".mkv", ".avi", ".mov",
                                        ".webm", ".flv", ".wmv", ".m4v"],
                            elem_id="ext-input",
                        )
                        gr.Markdown(
                            "*Formatos:* "
                            + " ".join(f"`{e[1:].upper()}`" for e in sorted(VIDEO_FORMATS))
                            + " · máx. 500 MB / 30 min"
                        )
                        ext_format = gr.Radio(
                            label="Formato de salida",
                            choices=["wav", "flac", "mp3"],
                            value="wav",
                            info="WAV = sin pérdida (el más pesado). FLAC = sin pérdida comprimido. "
                                 "MP3 = ligero. Para videos largos usa FLAC o MP3.",
                            elem_id="ext-format",
                        )

                    with gr.Column(scale=1):
                        ext_info = gr.Markdown(visible=True, elem_id="ext-info")
                        ext_output = gr.Audio(
                            label="Audio Extraído",
                            type="filepath",
                            interactive=False,
                            elem_id="ext-output",
                        )
                        ext_download = gr.File(
                            label="Descargar audio extraído",
                            elem_id="ext-download",
                        )

                with gr.Column(elem_classes="centered-action"):
                    ext_btn = gr.Button(
                        "Extraer Audio",
                        variant="primary",
                        size="lg",
                    )

                ext_btn.click(
                    fn=handle_extract,
                    inputs=[ext_input, ext_format],
                    outputs=[ext_output, ext_download, ext_info],
                )

                # Extract pipeline
                gr.HTML("""
                    <div class="pipeline-container">
                        <h3 class="pipeline-title">Proceso de Extracción</h3>
                        <div class="pipeline-flow">
                            <div class="pipeline-step">
                                <div class="step-num">01</div>
                                <div class="step-name">Análisis</div>
                                <div class="step-desc">FFprobe detecta codec, bitrate y canales del video.</div>
                            </div>
                            <div class="pipeline-arrow">➔</div>
                            <div class="pipeline-step">
                                <div class="step-num">02</div>
                                <div class="step-name">Demuxing</div>
                                <div class="step-desc">Separación de tracks de video y audio sin re-encoding.</div>
                            </div>
                            <div class="pipeline-arrow">➔</div>
                            <div class="pipeline-step final-step">
                                <div class="step-num">03</div>
                                <div class="step-name">Export</div>
                                <div class="step-desc">PCM 24-bit / 48kHz. Máxima fidelidad posible.</div>
                            </div>
                        </div>
                    </div>
                """)

            # ════════════════════════════════════════════
            #  TAB 3 — TEXT-TO-AUDIO
            # ════════════════════════════════════════════
            with gr.Tab("Text-to-Audio"):
                gr.HTML("""
                    <div class="section-hero">
                        <h2 class="section-title">
                            Voces neurales de
                            <span class="section-italic">Calibre Profesional.</span>
                        </h2>
                        <p class="section-desc">
                            Genera narraciones con voces <strong>indistinguibles de un humano</strong>.
                            Elige entre <strong>tres motores</strong>: Edge Neural (gratis),
                            <strong>Kokoro HD</strong> (local, sin coste por uso) y
                            <strong>OpenAI Studio</strong> (premium, voz dirigible por emoción).
                            <br><span style="font-size:0.85rem;color:#8C8C9A;">Nota: Kokoro
                            es 100% local; Edge y OpenAI envían el texto a sus servidores.</span>
                        </p>
                    </div>
                """)

                gr.HTML("""
                    <div class="how-to">
                        <div class="how-step"><span class="n">1</span><span class="t"><b>Escribe el texto</b> (o usa un ejemplo de abajo).</span></div>
                        <div class="how-step"><span class="n">2</span><span class="t">Elige <b>motor</b> y <b>voz</b>; ajusta velocidad/tono o emoción.</span></div>
                        <div class="how-step"><span class="n">3</span><span class="t">Pulsa <b>Generar Voz</b> y descarga el audio.</span></div>
                    </div>
                """)

                _tts_engines = available_engines()
                _tts_engine0 = _tts_engines[0]
                _tts_voices0 = get_voices(_tts_engine0)
                _openai_on = _tts_engine0 == ENGINE_OPENAI

                with gr.Row():
                    with gr.Column(scale=1):
                        tts_text = gr.Textbox(
                            label="Texto",
                            placeholder="Escribe o pega el texto que deseas narrar...",
                            lines=8,
                            max_lines=20,
                            elem_id="tts-text",
                        )
                        tts_counter = gr.Markdown(
                            _char_count_md("", _tts_engine0),
                            elem_id="tts-counter",
                        )
                        tts_engine = gr.Dropdown(
                            label="Motor de voz",
                            choices=_tts_engines,
                            value=_tts_engine0,
                            info=_engine_info(_tts_engine0, engine_char_limit(_tts_engine0)),
                            elem_id="tts-engine",
                        )
                        tts_voice = gr.Dropdown(
                            label="Voz",
                            choices=_tts_voices0,
                            value=(_tts_voices0[0] if _tts_voices0 else None),
                            elem_id="tts-voice",
                        )
                        tts_speed = gr.Radio(
                            label="Velocidad",
                            choices=SPEED_PRESETS,
                            value="Normal",
                            elem_id="tts-speed",
                        )
                        # ── Emotion controls (OpenAI premium only) ──
                        tts_style = gr.Dropdown(
                            label="Estilo / Emoción",
                            choices=list(OPENAI_STYLE_PRESETS.keys()),
                            value="Narrador profesional",
                            visible=_openai_on,
                            info="Dirige el tono de la voz (solo OpenAI Studio).",
                            elem_id="tts-style",
                        )
                        tts_instructions = gr.Textbox(
                            label="Instrucciones personalizadas (opcional)",
                            placeholder="Ej: Habla como un locutor de radio nocturna, voz grave y relajada...",
                            lines=2,
                            visible=_openai_on,
                            info="Si lo rellenas, sobrescribe el estilo de arriba.",
                            elem_id="tts-instructions",
                        )
                        # ── Pitch control (Edge only) ──
                        tts_pitch = gr.Radio(
                            label="Tono",
                            choices=list(PITCH_PRESETS.keys()),
                            value="Normal",
                            visible=(_tts_engine0 == ENGINE_EDGE),
                            elem_id="tts-pitch",
                        )
                        tts_master = gr.Checkbox(
                            label="Masterizar audio (voz más nítida y volumen parejo)",
                            value=True,
                            visible=(not _openai_on),
                            info="Recorta silencios y normaliza el volumen. Recomendado. "
                                 "(No aplica a OpenAI, que ya entrega audio masterizado).",
                            elem_id="tts-master",
                        )

                    with gr.Column(scale=1):
                        tts_output = gr.Audio(
                            label="Audio Generado",
                            type="filepath",
                            interactive=False,
                            elem_id="tts-output",
                        )
                        tts_download = gr.File(
                            label="Descargar audio generado",
                            elem_id="tts-download",
                        )

                with gr.Column(elem_classes="centered-action"):
                    tts_btn = gr.Button(
                        "Generar Voz",
                        variant="primary",
                        size="lg",
                    )

                # Sample texts so a first-time visitor can hear quality instantly.
                gr.Examples(
                    examples=[
                        ["El pedido número 4521 cuesta $1.299,50 y llega el 15/03/2026."],
                        ["Bienvenidos a Majes de Sivar. Hoy te contamos cómo la "
                         "inteligencia artificial está transformando el audio."],
                        ["Respira hondo. Relaja los hombros. Todo está en calma, "
                         "y tienes el tiempo que necesitas."],
                        ["Última hora: el comité anunció los resultados a las 14:30 "
                         "de hoy, con una participación del 87 %."],
                    ],
                    inputs=[tts_text],
                    label="Ejemplos (haz clic para probar)",
                )

                tts_text.input(
                    fn=_char_count_md,
                    inputs=[tts_text, tts_engine],
                    outputs=[tts_counter],
                )

                tts_engine.change(
                    fn=on_tts_engine_change,
                    inputs=[tts_engine, tts_text],
                    outputs=[tts_voice, tts_style, tts_instructions,
                             tts_pitch, tts_master, tts_engine, tts_counter],
                )

                tts_btn.click(
                    fn=handle_tts,
                    inputs=[tts_text, tts_engine, tts_voice, tts_speed,
                            tts_style, tts_instructions, tts_pitch, tts_master],
                    outputs=[tts_output, tts_download],
                )

                # TTS pipeline
                gr.HTML("""
                    <div class="pipeline-container">
                        <h3 class="pipeline-title">Motor de Síntesis</h3>
                        <div class="pipeline-flow">
                            <div class="pipeline-step">
                                <div class="step-num">01</div>
                                <div class="step-name">Análisis Lingüístico</div>
                                <div class="step-desc">Tokenización, prosodia y acentuación contextual.</div>
                            </div>
                            <div class="pipeline-arrow">➔</div>
                            <div class="pipeline-step">
                                <div class="step-num">02</div>
                                <div class="step-name">Motor Neural</div>
                                <div class="step-desc"><strong>Edge</strong>, <strong>Kokoro HD</strong> u <strong>OpenAI</strong>. Síntesis con expresividad y emoción dirigible.</div>
                            </div>
                            <div class="pipeline-arrow">➔</div>
                            <div class="pipeline-step final-step">
                                <div class="step-num">03</div>
                                <div class="step-name">Voz Final</div>
                                <div class="step-desc">Calidad de narrador profesional. Exportación directa.</div>
                            </div>
                        </div>
                    </div>
                """)

            # ════════════════════════════════════════════
            #  TAB 4 — CLONAR VOZ (ZeroGPU)
            # ════════════════════════════════════════════
            with gr.Tab("Clonar Voz"):
                gr.HTML("""
                    <div class="section-hero">
                        <h2 class="section-title">
                            Clona una voz con
                            <span class="section-italic">Fidelidad de Alto Nivel.</span>
                        </h2>
                        <p class="section-desc">
                            Sube una grabación y genera nueva voz que suena
                            <strong>claramente como esa persona</strong>, con
                            <strong>OpenVoice V2</strong> — 100% local en CPU, gratis.
                            <br><span style="font-size:0.85rem;color:#8C8C9A;">
                            Cuanta más referencia limpia (30–60s), mayor similitud.
                            Logra <strong>alta similitud</strong>, no una copia idéntica.
                            Corre en CPU: cada clonación puede tardar ~15–40s.</span>
                        </p>
                    </div>
                """)

                gr.HTML("""
                    <div class="how-to">
                        <div class="how-step"><span class="n">1</span><span class="t"><b>Sube 30–60s</b> de la voz a clonar (limpia, sin música).</span></div>
                        <div class="how-step"><span class="n">2</span><span class="t">Escribe el <b>texto</b> y marca el <b>consentimiento</b>.</span></div>
                        <div class="how-step"><span class="n">3</span><span class="t">Pulsa <b>Clonar y Generar</b> (alta similitud, no idéntica).</span></div>
                    </div>
                """)

                if not is_clone_available():
                    gr.Markdown(
                        "> ⚠️ El motor de clonación (OpenVoice V2) todavía no está "
                        "disponible en este entorno; se instala en el Space."
                    )

                with gr.Row():
                    with gr.Column(scale=1):
                        clone_ref = gr.Audio(
                            label="Voz de referencia (30–60s limpios)",
                            sources=["upload", "microphone"],
                            type="filepath",
                            elem_id="clone-ref",
                        )
                        gr.Markdown(
                            "*Sube voz clara, mono, sin música ni ruido. "
                            "Más referencia limpia = mayor parecido.*"
                        )
                        clone_text = gr.Textbox(
                            label="Texto a decir con esa voz",
                            placeholder="Escribe lo que dirá la voz clonada...",
                            lines=5,
                            max_lines=12,
                            elem_id="clone-text",
                        )
                        clone_master = gr.Checkbox(
                            label="Masterizar audio (más nítido y parejo)",
                            value=True,
                            elem_id="clone-master",
                        )
                        clone_consent = gr.Checkbox(
                            label="Confirmo que es mi propia voz o que tengo permiso "
                                  "explícito del titular para clonarla.",
                            value=False,
                            info="Clonar la voz de un tercero sin permiso es ilegal en "
                                 "muchos países. Obligatorio.",
                            elem_id="clone-consent",
                        )

                    with gr.Column(scale=1):
                        clone_output = gr.Audio(
                            label="Voz Clonada",
                            type="filepath",
                            interactive=False,
                            elem_id="clone-output",
                        )
                        clone_download = gr.File(
                            label="Descargar voz clonada",
                            elem_id="clone-download",
                        )

                with gr.Column(elem_classes="centered-action"):
                    clone_btn = gr.Button(
                        "Clonar y Generar",
                        variant="primary",
                        size="lg",
                    )

                clone_btn.click(
                    fn=handle_clone,
                    inputs=[clone_ref, clone_text, clone_consent, clone_master],
                    outputs=[clone_output, clone_download],
                )

        # ── Footer ──
        gr.HTML("""
            <div class="suite-footer">
                MAJES DE SIVAR · AUDIO SUITE · ENHANCE, EXTRACT &amp; KOKORO RUN LOCALLY
            </div>
        """)

    return app, gradio_theme, custom_css


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  Majes de Sivar — Audio Suite — Starting up...")
    logger.info("=" * 60)

    os.makedirs("outputs", exist_ok=True)

    # Detect if running on Hugging Face Spaces
    is_hf_space = os.environ.get("SPACE_ID") is not None

    app, theme, css = build_ui()
    # In Gradio 6, theme/css belong on launch() (not on the Blocks constructor).
    app.queue(default_concurrency_limit=1, max_size=20)  # protect the CPU tier
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=not is_hf_space,  # No share needed on HF Spaces
        show_error=True,
        theme=theme,
        css=css,
        allowed_paths=[os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))],
    )
