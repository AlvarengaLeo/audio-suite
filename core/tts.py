"""
tts.py - Multi-engine Text-to-Speech

Three selectable engines, ordered by naturalness. The goal is to escape the
"robotic" feel of plain Edge TTS by offering genuinely expressive options:

1. "Edge Neural (Gratis)"  — Microsoft Edge neural voices via edge-tts.
        Free, online, no API key. Improved here with pitch control and light
        text normalization. Solid baseline, but edge-tts is restricted by
        Microsoft to plain prosody (no emotion/styles), so its ceiling is low.

2. "Kokoro HD (Local)"     — Kokoro-82M, #1 on the HuggingFace TTS Arena.
        Apache-2.0, runs locally on CPU (~82M params). Free, no key, and NO
        per-use cost — ideal default for a public Space. Limited Spanish
        voices (1F/2M) but very natural.

3. "OpenAI Studio (Premium)" — gpt-4o-mini-tts. Human-level, *steerable* via
        natural-language emotion instructions ("habla cálido y pausado, como
        un narrador de documentales"). Requires the Space owner to set the
        OPENAI_API_KEY secret. Costs money per call, so it is character-limited
        to protect the owner from abuse on a public deployment.

The synthesizer degrades gracefully: if the chosen engine is unavailable or
fails, it falls back to Edge so the feature never goes dark.
"""

import os
import re
import tempfile
import logging
import asyncio
import threading

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine identifiers (display labels shown in the UI)
# ---------------------------------------------------------------------------
ENGINE_EDGE = "Edge Neural (Gratis)"
ENGINE_KOKORO = "Kokoro HD (Local)"
ENGINE_OPENAI = "OpenAI Studio (Premium)"

# Per-engine character limits.
#   Edge   : generous, it's free and fast.
#   Kokoro : runs on the shared CPU of the Space -> guard against huge jobs.
#   OpenAI : costs money per character -> tight cap to protect the owner.
MAX_CHARS = {
    ENGINE_EDGE: 10000,
    ENGINE_KOKORO: 5000,
    ENGINE_OPENAI: 2000,
}

# Free-text emotion instructions (OpenAI) are also capped, so a user can't
# inflate the paid token cost through the instructions field.
MAX_INSTRUCTION_CHARS = 300

# Global daily character budget for the paid OpenAI engine. Protects the Space
# owner from runaway cost on a public deployment. In-memory (resets on restart),
# so keep it conservative. Override with the OPENAI_DAILY_CHAR_BUDGET env var.
OPENAI_DAILY_CHAR_BUDGET = int(os.environ.get("OPENAI_DAILY_CHAR_BUDGET", "50000"))
_openai_usage = {"date": None, "chars": 0}
_budget_lock = threading.Lock()


def _openai_budget_remaining() -> int:
    """Characters left in today's OpenAI budget (resets at midnight UTC)."""
    import datetime
    today = datetime.date.today().isoformat()
    with _budget_lock:
        if _openai_usage["date"] != today:
            _openai_usage["date"] = today
            _openai_usage["chars"] = 0
        return max(0, OPENAI_DAILY_CHAR_BUDGET - _openai_usage["chars"])


def _record_openai_usage(n_chars: int):
    with _budget_lock:
        _openai_usage["chars"] += n_chars


def openai_budget_remaining() -> int:
    """Public accessor for the UI (characters left in today's OpenAI budget)."""
    return _openai_budget_remaining()

# ---------------------------------------------------------------------------
# Speed presets (shared display names, mapped per engine)
# ---------------------------------------------------------------------------
SPEED_PRESETS = ["Muy lento", "Lento", "Normal", "Rápido", "Muy rápido"]

# Edge uses percentage strings; Kokoro/OpenAI use a float factor.
_SPEED_EDGE = {
    "Muy lento": "-30%",
    "Lento": "-15%",
    "Normal": "+0%",
    "Rápido": "+15%",
    "Muy rápido": "+30%",
}
_SPEED_FACTOR = {
    "Muy lento": 0.7,
    "Lento": 0.85,
    "Normal": 1.0,
    "Rápido": 1.15,
    "Muy rápido": 1.3,
}
# Natural-language speed hint for OpenAI (it ignores the numeric `speed` param).
_SPEED_HINT_ES = {
    "Muy lento": "Habla muy despacio, con calma.",
    "Lento": "Habla a un ritmo pausado.",
    "Normal": "",
    "Rápido": "Habla a un ritmo ágil.",
    "Muy rápido": "Habla rápido y con energía.",
}

# Edge pitch presets
PITCH_PRESETS = {
    "Grave": "-20Hz",
    "Ligeramente grave": "-10Hz",
    "Normal": "+0Hz",
    "Ligeramente agudo": "+10Hz",
    "Agudo": "+20Hz",
}

# ---------------------------------------------------------------------------
# Voice catalogs (one per engine)
# ---------------------------------------------------------------------------

# Edge / Azure neural voices (display -> voice id)
EDGE_VOICES = {
    # Spanish
    "🇪🇸 Elvira (ES - Femenina)": "es-ES-ElviraNeural",
    "🇪🇸 Álvaro (ES - Masculina)": "es-ES-AlvaroNeural",
    "🇪🇸 Ximena (ES - Femenina)": "es-ES-XimenaNeural",
    "🇲🇽 Dalia (MX - Femenina)": "es-MX-DaliaNeural",
    "🇲🇽 Jorge (MX - Masculina)": "es-MX-JorgeNeural",
    "🇦🇷 Elena (AR - Femenina)": "es-AR-ElenaNeural",
    "🇨🇴 Salomé (CO - Femenina)": "es-CO-SalomeNeural",
    # English
    "🇺🇸 Ava multilingüe (US - F)": "en-US-AvaMultilingualNeural",
    "🇺🇸 Andrew multilingüe (US - M)": "en-US-AndrewMultilingualNeural",
    "🇺🇸 Aria (US - Female)": "en-US-AriaNeural",
    "🇺🇸 Guy (US - Male)": "en-US-GuyNeural",
    "🇬🇧 Sonia (UK - Female)": "en-GB-SoniaNeural",
    "🇬🇧 Ryan (UK - Male)": "en-GB-RyanNeural",
    # Portuguese
    "🇧🇷 Francisca (BR - Feminina)": "pt-BR-FranciscaNeural",
    "🇧🇷 Antonio (BR - Masculino)": "pt-BR-AntonioNeural",
    # French / German / Italian / Japanese
    "🇫🇷 Denise (FR - Féminine)": "fr-FR-DeniseNeural",
    "🇫🇷 Henri (FR - Masculin)": "fr-FR-HenriNeural",
    "🇩🇪 Katja (DE - Weiblich)": "de-DE-KatjaNeural",
    "🇮🇹 Elsa (IT - Femminile)": "it-IT-ElsaNeural",
    "🇯🇵 Nanami (JP - 女性)": "ja-JP-NanamiNeural",
}

# Kokoro-82M voices (display -> (lang_code, voice_id)).
# IDs verified against hexgrad/Kokoro-82M VOICES.md.
KOKORO_VOICES = {
    # Spanish (lang_code 'e')
    "🇪🇸 Dora (ES - Femenina)": ("e", "ef_dora"),
    "🇪🇸 Alex (ES - Masculina)": ("e", "em_alex"),
    "🇪🇸 Santa (ES - Masculina)": ("e", "em_santa"),
    # American English (lang_code 'a')
    "🇺🇸 Heart (US - Female)": ("a", "af_heart"),
    "🇺🇸 Bella (US - Female)": ("a", "af_bella"),
    "🇺🇸 Nicole (US - Female)": ("a", "af_nicole"),
    "🇺🇸 Michael (US - Male)": ("a", "am_michael"),
    "🇺🇸 Fenrir (US - Male)": ("a", "am_fenrir"),
    # British English (lang_code 'b')
    "🇬🇧 Emma (UK - Female)": ("b", "bf_emma"),
    "🇬🇧 George (UK - Male)": ("b", "bm_george"),
    # Brazilian Portuguese (lang_code 'p')
    "🇧🇷 Dora (BR - Feminina)": ("p", "pf_dora"),
    "🇧🇷 Alex (BR - Masculino)": ("p", "pm_alex"),
    # French / Italian
    "🇫🇷 Siwis (FR - Féminine)": ("f", "ff_siwis"),
    "🇮🇹 Sara (IT - Femminile)": ("i", "if_sara"),
}

# OpenAI gpt-4o-mini-tts voices (display -> voice id). Voices are
# language-agnostic; the input text determines the spoken language.
OPENAI_VOICES = {
    "Coral (cálida, femenina)": "coral",
    "Sage (serena, femenina)": "sage",
    "Nova (enérgica, femenina)": "nova",
    "Shimmer (suave, femenina)": "shimmer",
    "Alloy (neutral)": "alloy",
    "Echo (clara, masculina)": "echo",
    "Onyx (profunda, masculina)": "onyx",
    "Ash (cálida, masculina)": "ash",
    "Ballad (expresiva, masculina)": "ballad",
    "Verse (versátil)": "verse",
    "Fable (narrador)": "fable",
}

# Emotion / delivery presets for OpenAI's `instructions` parameter.
OPENAI_STYLE_PRESETS = {
    "Narrador profesional": (
        "Habla con la voz cálida, clara y segura de un narrador de "
        "documentales profesional. Dicción impecable, ritmo pausado y natural, "
        "con entonación expresiva pero sobria."
    ),
    "Conversacional cercano": (
        "Habla de forma natural y cercana, como si conversaras con un amigo. "
        "Tono relajado, cálido y espontáneo, con entonación natural y pausas "
        "realistas."
    ),
    "Noticiero / Serio": (
        "Habla con el tono claro, autoritario y neutral de un presentador de "
        "noticias profesional. Articulación nítida y cadencia firme."
    ),
    "Alegre y enérgico": (
        "Habla con energía, entusiasmo y una sonrisa en la voz. Tono positivo, "
        "dinámico y motivador."
    ),
    "Calmado y relajante": (
        "Habla de forma muy calmada, suave y relajante, como en una meditación "
        "guiada. Ritmo lento, voz serena y respiración tranquila."
    ),
    "Dramático / Cinematográfico": (
        "Habla con intensidad dramática y gran emoción, como en el tráiler de "
        "una película. Pausas marcadas, dinámica amplia y mucha expresividad."
    ),
    "Personalizado": "",  # user supplies their own text
}


# ===================================================================
# AVAILABILITY
# ===================================================================
def is_kokoro_available() -> bool:
    """True if the Kokoro package is importable."""
    try:
        import kokoro  # noqa: F401
        return True
    except Exception:
        return False


def is_openai_available() -> bool:
    """True if the OpenAI SDK is installed AND a key is configured."""
    if not os.environ.get("OPENAI_API_KEY"):
        return False
    try:
        import openai  # noqa: F401
        return True
    except Exception:
        return False


def available_engines() -> list:
    """
    Return available engine labels.

    A FREE engine is listed first so it is the UI default — on a public
    Space the premium (paid) OpenAI engine must never be the silent default,
    or every visitor would spend the owner's money. Order:
        Kokoro (free, local, best quality) > Edge (free) > OpenAI (paid, last).
    """
    engines = []
    if is_kokoro_available():
        engines.append(ENGINE_KOKORO)
    engines.append(ENGINE_EDGE)  # always available (online)
    if is_openai_available():
        engines.append(ENGINE_OPENAI)  # paid -> selectable but never default
    return engines


def get_voices(engine: str) -> list:
    """Return the list of voice display names for an engine."""
    if engine == ENGINE_KOKORO:
        return list(KOKORO_VOICES.keys())
    if engine == ENGINE_OPENAI:
        return list(OPENAI_VOICES.keys())
    return list(EDGE_VOICES.keys())


def default_voice(engine: str) -> str:
    """Return a sensible default voice for an engine (Spanish-first)."""
    voices = get_voices(engine)
    return voices[0] if voices else None


def engine_char_limit(engine: str) -> int:
    return MAX_CHARS.get(engine, 10000)


# ===================================================================
# PUBLIC ENTRY POINT
# ===================================================================
def synthesize_speech(text: str,
                      engine: str = ENGINE_EDGE,
                      voice_name: str = None,
                      speed: str = "Normal",
                      style_preset: str = None,
                      custom_instructions: str = None,
                      pitch_preset: str = "Normal",
                      master: bool = True) -> str:
    """
    Synthesize speech with the selected engine.

    Pipeline: Spanish text normalization -> cache lookup -> engine synthesis
    -> (optional) audio finalization (silence trim + loudness normalization).

    Args:
        text: Text to speak.
        engine: One of ENGINE_EDGE / ENGINE_KOKORO / ENGINE_OPENAI.
        voice_name: Display name from the engine's voice catalog.
        speed: Speed preset name (SPEED_PRESETS).
        style_preset: Emotion preset key (OpenAI only).
        custom_instructions: Free-text delivery instructions (OpenAI only).
        pitch_preset: Pitch preset (Edge only).
        master: Apply silence-trim + loudness normalization to the result
            (skipped for OpenAI, which is already produced).

    Returns:
        Tuple (audio_path, engine_actually_used). The used engine may differ
        from the requested one when a fallback to Edge occurred.
    """
    if not text or not text.strip():
        raise ValueError("Escribe algún texto para sintetizar.")

    # Enforce the character limit on what the USER typed (so the message matches
    # their input), then normalize for the engines.
    raw = text.strip()
    limit = engine_char_limit(engine)
    if len(raw) > limit:
        raise ValueError(
            f"Texto demasiado largo ({len(raw)} caracteres). "
            f"El máximo para el motor '{engine}' es {limit:,} caracteres."
        )

    # Normalize Spanish text (numbers, currency, dates, abbreviations...) so
    # every engine reads it naturally. Never raises.
    text = _maybe_normalize(raw)

    # Resolve a default voice if the UI sent one from a different engine.
    if voice_name not in get_voices(engine):
        voice_name = default_voice(engine)

    # Cache identical requests (saves CPU on Kokoro and money on OpenAI).
    key = _cache_key(text, engine, voice_name, speed, style_preset,
                     custom_instructions, pitch_preset, master)
    with _cache_lock:
        cached = _tts_cache.get(key)
    if cached and os.path.exists(cached[0]):
        logger.info("TTS cache hit.")
        return cached  # (path, used_engine)

    raw_path, used_engine = _dispatch(
        text, engine, voice_name, speed, style_preset,
        custom_instructions, pitch_preset
    )

    # Finalize (trim + loudness) for Edge/Kokoro. OpenAI is already produced.
    out_path = raw_path
    if master and used_engine != ENGINE_OPENAI:
        try:
            from core.postprocessor import finalize_voice
            out_path = finalize_voice(raw_path)
        except Exception as e:
            logger.warning(f"finalize skipped: {e}")
            out_path = raw_path

    _cache_store(key, (out_path, used_engine))
    return out_path, used_engine


def _dispatch(text, engine, voice_name, speed, style_preset,
              custom_instructions, pitch_preset):
    """Run the chosen engine, returning (audio_path, engine_actually_used)."""
    # --- Premium: OpenAI -------------------------------------------------
    if engine == ENGINE_OPENAI:
        if is_openai_available():
            if _openai_budget_remaining() < len(text):
                raise ValueError(
                    "Se alcanzó el límite diario del motor premium (OpenAI). "
                    "Usa el motor Edge o Kokoro (gratuitos) o inténtalo mañana."
                )
            try:
                out = _synth_openai(
                    text, voice_name, speed, style_preset, custom_instructions
                )
                _record_openai_usage(len(text))
                return out, ENGINE_OPENAI
            except Exception as e:
                logger.error(f"OpenAI TTS failed: {e}. Falling back to Edge.")
        else:
            logger.warning("OpenAI engine requested but not configured. Using Edge.")
        return (_synth_edge(text, default_voice(ENGINE_EDGE), speed, pitch_preset),
                ENGINE_EDGE)

    # --- Local HD: Kokoro ------------------------------------------------
    if engine == ENGINE_KOKORO:
        if is_kokoro_available():
            try:
                return _synth_kokoro(text, voice_name, speed), ENGINE_KOKORO
            except Exception as e:
                logger.error(f"Kokoro TTS failed: {e}. Falling back to Edge.")
        else:
            logger.warning("Kokoro engine requested but not installed. Using Edge.")
        return (_synth_edge(text, default_voice(ENGINE_EDGE), speed, pitch_preset),
                ENGINE_EDGE)

    # --- Default: Edge ---------------------------------------------------
    return _synth_edge(text, voice_name, speed, pitch_preset), ENGINE_EDGE


def _maybe_normalize(text: str) -> str:
    """Apply Spanish text normalization; fall back to raw text on any error."""
    try:
        from core.text_normalizer import normalize_es
        return normalize_es(text)
    except Exception as e:
        logger.debug(f"normalization skipped: {e}")
        return text


# --- Request cache (hash of all params -> (generated file, used_engine)) ----
_tts_cache = {}
_TTS_CACHE_MAX = 64
_cache_lock = threading.Lock()


def _cache_key(*parts) -> str:
    import hashlib
    return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()


def _cache_store(key: str, value):
    with _cache_lock:
        _tts_cache[key] = value
        if len(_tts_cache) > _TTS_CACHE_MAX:
            _tts_cache.pop(next(iter(_tts_cache)), None)  # evict oldest


# ===================================================================
# ENGINE: EDGE (edge-tts)
# ===================================================================
def _normalize_text(text: str) -> str:
    """Light cleanup that improves edge-tts pacing without altering meaning."""
    text = re.sub(r"[ \t]+", " ", text)          # collapse runs of spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)        # cap blank lines
    return text.strip()


# Multi-char Spanish abbreviations whose '.' must NOT end a sentence.
_SENT_ABBR = {"sr", "sra", "srta", "dr", "dra", "ud", "uds", "ee", "uu",
              "etc", "ej", "num", "vs", "av", "avda", "aprox", "pag"}


def segment_sentences(text: str) -> list:
    """
    Split text into sentences (Spanish-aware), without changing any words.
    Guards against breaking on abbreviations (Sr., EE.UU.) and decimals (3.5),
    so engines that phrase per-segment intonate naturally.
    """
    text = re.sub(r"[ \t]+", " ", (text or "").strip())
    out = []
    for para in re.split(r"\n{2,}", text):
        para = para.replace("\n", " ").strip()
        if not para:
            continue
        buf = ""
        for part in re.split(r"(?<=[.!?…])\s+", para):
            cand = (buf + " " + part).strip() if buf else part
            last = cand.rsplit(" ", 1)[-1].rstrip(".!?…").lower()
            # Keep buffering if it ends in an abbreviation or a decimal number.
            if last in _SENT_ABBR or re.search(r"\d[.,]\d*$", cand):
                buf = cand
            else:
                out.append(cand)
                buf = ""
        if buf:
            out.append(buf)
    return [s for s in out if s]


async def _edge_async(text, voice_id, rate, volume, pitch, output_path):
    import edge_tts
    communicate = edge_tts.Communicate(
        text=text, voice=voice_id, rate=rate, volume=volume, pitch=pitch
    )
    await communicate.save(output_path)


def _synth_edge(text: str, voice_name: str, speed: str,
                pitch_preset: str = "Normal") -> str:
    voice_id = EDGE_VOICES.get(voice_name) or EDGE_VOICES[default_voice(ENGINE_EDGE)]
    rate = _SPEED_EDGE.get(speed, "+0%")
    pitch = PITCH_PRESETS.get(pitch_preset, "+0Hz")
    text = _normalize_text(text)

    out = _new_tempfile(".mp3")
    logger.info(f"Edge TTS: voice={voice_id}, rate={rate}, pitch={pitch}, "
                f"len={len(text)}")

    asyncio.run(_edge_async(text, voice_id, rate, "+0%", pitch, out))

    _log_size("Edge", out)
    return out


# ===================================================================
# ENGINE: KOKORO (local, Apache-2.0)
# ===================================================================
_kokoro_pipelines = {}


def _get_kokoro_pipeline(lang_code: str):
    """Cache one KPipeline per language code."""
    if lang_code not in _kokoro_pipelines:
        from kokoro import KPipeline
        logger.info(f"Loading Kokoro pipeline (lang_code='{lang_code}')...")
        _kokoro_pipelines[lang_code] = KPipeline(lang_code=lang_code)
    return _kokoro_pipelines[lang_code]


def _synth_kokoro(text: str, voice_name: str, speed: str) -> str:
    import numpy as np
    import soundfile as sf

    lang_code, voice_id = KOKORO_VOICES[voice_name]
    factor = _SPEED_FACTOR.get(speed, 1.0)
    pipeline = _get_kokoro_pipeline(lang_code)

    logger.info(f"Kokoro TTS: voice={voice_id}, lang={lang_code}, "
                f"speed={factor}, len={len(text)}")

    # Feed one sentence per segment so Kokoro phrases naturally and never
    # silently truncates long text at its internal 510-phoneme cap.
    seg_text = "\n".join(segment_sentences(text)) or text

    segments = []
    for _, _, audio in pipeline(seg_text, voice=voice_id, speed=factor):
        if audio is None:
            continue
        if hasattr(audio, "detach"):  # torch tensor
            audio = audio.detach().cpu().numpy()
        segments.append(np.asarray(audio, dtype=np.float32).reshape(-1))

    if not segments:
        raise RuntimeError("Kokoro produjo audio vacío.")

    full = np.concatenate(segments) if len(segments) > 1 else segments[0]

    out = _new_tempfile(".wav")
    sf.write(out, full, 24000)  # Kokoro outputs 24 kHz
    _log_size("Kokoro", out)
    return out


# ===================================================================
# ENGINE: OPENAI (gpt-4o-mini-tts, premium)
# ===================================================================
def _build_openai_instructions(speed: str, style_preset: str,
                               custom_instructions: str) -> str:
    """Compose the natural-language `instructions` steering string."""
    if custom_instructions and custom_instructions.strip():
        base = custom_instructions.strip()[:MAX_INSTRUCTION_CHARS]
    else:
        base = OPENAI_STYLE_PRESETS.get(
            style_preset, OPENAI_STYLE_PRESETS["Narrador profesional"]
        ) or OPENAI_STYLE_PRESETS["Narrador profesional"]

    # gpt-4o-mini-tts ignores the numeric `speed` param, so fold speed in here.
    speed_hint = _SPEED_HINT_ES.get(speed, "")
    if speed_hint:
        base = f"{base} {speed_hint}"
    return base.strip()


def _synth_openai(text: str, voice_name: str, speed: str,
                  style_preset: str, custom_instructions: str) -> str:
    from openai import OpenAI

    voice_id = OPENAI_VOICES.get(voice_name, "coral")
    instructions = _build_openai_instructions(
        speed, style_preset, custom_instructions
    )

    logger.info(f"OpenAI TTS: voice={voice_id}, len={len(text)}, "
                f"instructions='{instructions[:60]}...'")

    client = OpenAI()  # reads OPENAI_API_KEY from env
    out = _new_tempfile(".mp3")
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice_id,
        input=text,
        instructions=instructions,
        response_format="mp3",
    ) as response:
        response.stream_to_file(out)

    _log_size("OpenAI", out)
    return out


# ===================================================================
# HELPERS
# ===================================================================
def _new_tempfile(suffix: str) -> str:
    f = tempfile.NamedTemporaryFile(
        suffix=f"_tts{suffix}", delete=False, dir=tempfile.gettempdir()
    )
    path = f.name
    f.close()
    return path


def _log_size(engine: str, path: str):
    try:
        kb = os.path.getsize(path) / 1024
        logger.info(f"{engine} TTS complete: {path} ({kb:.0f} KB)")
    except OSError:
        pass
