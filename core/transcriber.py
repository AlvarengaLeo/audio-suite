"""
transcriber.py - Audio -> Texto: transcripción auto-verificada con faster-whisper.

Pipeline de calidad (cada etapa devuelve un nuevo estado, todas desactivables):

    S1  ASR de alta precisión  (preset anti-alucinación: VAD + thresholds +
                                condition_on_previous_text=False + word timestamps)
    S2  Rectificación          (re-transcribe SOLO los segmentos dudosos con
                                large-v3 + reintentos; "el modelo verifica que lo
                                hizo bien"). Acepta el rehecho solo si mejora.
    S3  Pulido con IA (opcional, OFF por defecto, requiere OPENAI_API_KEY):
                                corrige tildes/puntuación/mayúsculas SIN cambiar
                                palabras (gpt-4o-mini, temperature=0, JSON).
    S4  Verificación anti-deriva: revierte cualquier trozo donde la IA cambie
                                contenido (identidad de tokens normalizados +
                                secuencia numérica). Circuit-breaker global.
    clean_text  Punto único de limpieza de caracteres (NFC, control chars,
                zero-width, tartamudeo, líneas duplicadas). Se aplica tras S1 y S3.

faster-whisper se basa en CTranslate2 (sin torch) y decodifica el audio por sí
mismo vía PyAV, así que no hace falta convertir a 16 kHz antes de S1.
"""

import os
import re
import gc
import math
import logging
import tempfile
import subprocess
import unicodedata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración pública (la UI lee estas constantes)
# ---------------------------------------------------------------------------
# Etiquetas mostradas en el dropdown -> id real del modelo whisper.
_MODEL_LABEL_TO_ID = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "Calidad máxima (large-v3)": "large-v3",
}
TRANSCRIBE_MODELS = list(_MODEL_LABEL_TO_ID.keys())
DEFAULT_MODEL = "base"

# Etiqueta UI -> código de idioma whisper (None = autodetectar).
TRANSCRIBE_LANGUAGES = {"Auto": None, "Español": "es", "Inglés": "en"}

# Modelo al que escala S2 para rehacer los fragmentos dudosos (máxima calidad).
RECTIFY_MODEL = "large-v3"

# faster-whisper decodifica internamente; aceptamos audio y vídeo comunes.
SUPPORTED_EXT = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma", ".aac", ".opus",
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".wmv", ".m4v",
}

# Prompt en español (con tildes y signos) que sesga la ortografía/puntuación.
# Solo condiciona la PRIMERA ventana, así que se mantiene corto.
_ES_PROMPT = (
    "Transcripción en español con puntuación y acentos correctos. "
    "Por ejemplo: ¿Cómo estás? ¡Qué día más bonito! "
    "Sí, también, según, café, año, niño, corazón."
)

# VAD afinado para audio largo (recorta silencio -> menos alucinaciones).
_VAD_PARAMS = dict(
    threshold=0.5,
    min_speech_duration_ms=250,
    max_speech_duration_s=30,
    min_silence_duration_ms=2000,
    speech_pad_ms=400,
)

# Caché de modelos por id RESUELTO (concurrencia=1 -> singleton seguro).
_MODELS = {}


# ---------------------------------------------------------------------------
# Disponibilidad / carga de modelos
# ---------------------------------------------------------------------------
def is_transcribe_available() -> bool:
    """True si faster-whisper está instalado en el entorno principal."""
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def _resolve_model_id(model_size: str) -> str:
    return _MODEL_LABEL_TO_ID.get(model_size, "base")


def _load_model(model_id: str):
    """Carga (y cachea) un WhisperModel en CPU int8."""
    if model_id not in _MODELS:
        from faster_whisper import WhisperModel
        logger.info(f"Cargando modelo whisper '{model_id}' (CPU, int8)...")
        _MODELS[model_id] = WhisperModel(
            model_id, device="cpu", compute_type="int8"
        )
    return _MODELS[model_id]


def _evict_models(keep: str = None):
    """Libera de memoria todos los modelos salvo `keep` (E1: evita tener S1 y
    large-v3 residentes a la vez en un Space de CPU)."""
    for mid in list(_MODELS.keys()):
        if mid != keep:
            try:
                del _MODELS[mid]
            except KeyError:
                pass
    gc.collect()


# ---------------------------------------------------------------------------
# Modelo de datos
# ---------------------------------------------------------------------------
class Segment:
    __slots__ = ("start", "end", "text", "raw_text", "avg_logprob",
                 "compression_ratio", "no_speech_prob", "low_word_ratio",
                 "confidence", "rectified")

    def __init__(self, start, end, text, avg_logprob, compression_ratio,
                 no_speech_prob, low_word_ratio):
        self.start = start
        self.end = end
        self.text = text
        self.raw_text = text          # referencia inmutable de S1 (para drift)
        self.avg_logprob = avg_logprob
        self.compression_ratio = compression_ratio
        self.no_speech_prob = no_speech_prob
        self.low_word_ratio = low_word_ratio
        self.confidence = _segment_confidence(
            avg_logprob, compression_ratio, no_speech_prob, low_word_ratio
        )
        self.rectified = False


class Transcript:
    def __init__(self, segments, language, language_probability, duration, meta):
        self.segments = segments
        self.language = language
        self.language_probability = language_probability
        self.duration = duration
        self.meta = meta

    @property
    def text(self) -> str:
        return "\n".join(s.text for s in self.segments if s.text and s.text.strip())

    @property
    def raw_text(self) -> str:
        return "\n".join(
            s.raw_text for s in self.segments if s.raw_text and s.raw_text.strip()
        )


def _segment_confidence(avg_logprob, compression_ratio, no_speech_prob,
                        low_word_ratio) -> float:
    """Confianza 0..1. exp(avg_logprob) = prob. media por token; penaliza
    repetición (CR alto), silencio (no_speech) y palabras dudosas."""
    p = math.exp(avg_logprob) if avg_logprob is not None else 0.5
    cr_excess = max(0.0, (compression_ratio or 0.0) - 2.4)
    cr_factor = max(0.0, 1.0 - cr_excess)           # CR>2.4 -> baja
    speech = 1.0 - (no_speech_prob or 0.0)
    conf = p * cr_factor * speech
    if low_word_ratio and low_word_ratio > 0.30:
        conf *= 0.7
    return max(0.0, min(1.0, conf))


# ---------------------------------------------------------------------------
# Limpieza de caracteres — punto único (se aplica tras S1 y tras S3)
# ---------------------------------------------------------------------------
_ZERO_WIDTH = dict.fromkeys(
    map(ord, "​‌‍⁠﻿"), None
)
_STUTTER_RE = re.compile(r"\b(\w+)(\s+\1\b){2,}", flags=re.IGNORECASE | re.UNICODE)
_SPACES_RE = re.compile(r"[ \t]+")


def clean_text(s: str) -> str:
    """Normaliza a NFC, quita caracteres de control / zero-width / NBSP,
    colapsa espacios y tartamudeos, y deduplica líneas consecutivas iguales."""
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace(" ", " ")                    # NBSP -> espacio
    s = s.translate(_ZERO_WIDTH)
    # quita categorías de control (Cc/Cf/...) salvo salto de línea y tab
    s = "".join(
        c for c in s if unicodedata.category(c)[0] != "C" or c in "\n\t"
    )
    s = _STUTTER_RE.sub(r"\1", s)                    # 3+ repeticiones -> 1
    out_lines = []
    prev = None
    for line in s.split("\n"):
        line = _SPACES_RE.sub(" ", line).strip()
        if line and line == prev:                   # dedup consecutivo
            continue
        out_lines.append(line)
        if line:
            prev = line
    out = "\n".join(out_lines).strip()
    out = re.sub(r"\n{3,}", "\n\n", out)             # máx. 1 línea en blanco
    return out


# ---------------------------------------------------------------------------
# S1 — ASR de alta precisión
# ---------------------------------------------------------------------------
def _safe_transcribe(model, audio_path, **kwargs):
    """Llama a model.transcribe tolerando versiones de faster-whisper que no
    soporten algún kwarg (reintenta quitando los opcionales)."""
    try:
        return model.transcribe(audio_path, **kwargs)
    except TypeError as e:
        logger.warning(f"transcribe kwargs incompatibles ({e}); reintentando reducido.")
        for opt in ("hallucination_silence_threshold", "vad_parameters",
                    "patience", "length_penalty", "best_of"):
            kwargs.pop(opt, None)
        return model.transcribe(audio_path, **kwargs)


def _transcribe_call(model, audio_path, language, beam_size, temperature,
                     with_prompt, word_timestamps=True):
    kwargs = dict(
        language=language,
        task="transcribe",
        beam_size=beam_size,
        best_of=5,
        patience=1.0,
        length_penalty=1.0,
        temperature=temperature,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=False,   # clave anti-bucle en audio largo
        word_timestamps=word_timestamps,
        hallucination_silence_threshold=2.0,
        vad_filter=True,
        vad_parameters=_VAD_PARAMS,
    )
    if with_prompt:
        kwargs["initial_prompt"] = _ES_PROMPT
    return _safe_transcribe(model, audio_path, **kwargs)


def transcribe_base(audio_path, model_size=DEFAULT_MODEL, language=None,
                    progress_cb=None) -> Transcript:
    """S1: transcripción base con el preset anti-alucinación."""
    model_id = _resolve_model_id(model_size)
    model = _load_model(model_id)
    # Prompt español solo si el idioma es 'es' explícito (en Auto/None no, para
    # no sesgar audio en inglés).
    with_prompt = (language == "es")
    full_temp = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    segments_gen, info = _transcribe_call(
        model, audio_path, language=language, beam_size=5,
        temperature=full_temp, with_prompt=with_prompt,
    )
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    segments = []
    for seg in segments_gen:
        # ratio de palabras de baja confianza (se guarda y se sueltan los words)
        low_word_ratio = 0.0
        words = getattr(seg, "words", None)
        if words:
            probs = [w.probability for w in words
                     if getattr(w, "probability", None) is not None]
            if probs:
                low_word_ratio = sum(p < 0.4 for p in probs) / len(probs)
        segments.append(Segment(
            start=float(seg.start or 0.0),
            end=float(seg.end or 0.0),
            text=seg.text or "",
            avg_logprob=getattr(seg, "avg_logprob", None),
            compression_ratio=getattr(seg, "compression_ratio", None),
            no_speech_prob=getattr(seg, "no_speech_prob", None),
            low_word_ratio=low_word_ratio,
        ))
        if progress_cb and duration:
            progress_cb(min(0.99, (seg.end or 0.0) / duration))

    meta = {
        "model": model_id,
        "degraded": False,
        "rectified_pct": 0.0,
        "rectified_spans": 0,
        "proof": "no",
        "reverted_chunks": 0,
        "notes": [],
    }
    return Transcript(
        segments=segments,
        language=getattr(info, "language", language) or (language or "?"),
        language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
        duration=duration,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# S2 — Rectificación de máxima calidad (sin tope de tiempo)
# ---------------------------------------------------------------------------
def _is_silence(seg) -> bool:
    return ((seg.no_speech_prob or 0.0) > 0.6
            and (seg.avg_logprob if seg.avg_logprob is not None else 0.0) < -1.0)


def _is_weak(seg) -> bool:
    if seg.compression_ratio is not None and seg.compression_ratio > 2.4:
        return True
    if seg.avg_logprob is not None and seg.avg_logprob < -0.8:
        return True
    if seg.low_word_ratio and seg.low_word_ratio > 0.30:
        return True
    return False


def _slice_audio(audio_path, start, end, total):
    """Corta [start,end] (con padding ya aplicado) a WAV 16 kHz mono."""
    start = max(0.0, start)
    if total:
        end = min(total, end)
    dur = max(0.05, end - start)
    tf = tempfile.NamedTemporaryFile(
        suffix="_clip.wav", delete=False, dir=tempfile.gettempdir()
    )
    out_path = tf.name
    tf.close()
    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
        "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
        out_path,
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    return out_path


def _retranscribe_clip(model, clip_path, language):
    """Re-transcribe un clip corto con large-v3: beam 8, y si sigue dudoso,
    beam 10 + escalera de temperatura corta. Devuelve (texto, confianza)."""
    with_prompt = (language == "es")

    def run(beam, temps):
        gen, _ = _transcribe_call(
            model, clip_path, language=language, beam_size=beam,
            temperature=temps, with_prompt=with_prompt, word_timestamps=False,
        )
        parts, confs, weights = [], [], []
        for s in gen:
            parts.append(s.text or "")
            c = _segment_confidence(
                getattr(s, "avg_logprob", None),
                getattr(s, "compression_ratio", None),
                getattr(s, "no_speech_prob", None), 0.0,
            )
            w = max(0.1, (s.end or 0.0) - (s.start or 0.0))
            confs.append(c)
            weights.append(w)
        text = " ".join(p.strip() for p in parts if p and p.strip())
        conf = (sum(c * w for c, w in zip(confs, weights)) / sum(weights)
                if weights else 0.0)
        return text, conf

    text, conf = run(8, [0.0])
    if conf < 0.6:                       # aún dudoso -> escalar
        text2, conf2 = run(10, [0.0, 0.2, 0.4])
        if conf2 > conf:
            text, conf = text2, conf2
    return text, conf


def rectify(tr: Transcript, audio_path, progress_cb=None) -> Transcript:
    """S2: rehace los segmentos dudosos con large-v3 y los acepta solo si la
    confianza mejora. Sin tope de tiempo (elección del usuario)."""
    weak_idx = [i for i, s in enumerate(tr.segments)
                if _is_weak(s) and not _is_silence(s)]
    if not weak_idx:
        return tr

    # Fusiona índices dudosos adyacentes (gap <= 1 seg) en spans.
    spans = []
    cur = [weak_idx[0]]
    for i in weak_idx[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            spans.append(cur)
            cur = [i]
    spans.append(cur)

    # E1: libera el modelo de S1 antes de cargar large-v3 (salvo que sea el mismo).
    _evict_models(keep=RECTIFY_MODEL if RECTIFY_MODEL in _MODELS else None)
    big = _load_model(RECTIFY_MODEL)

    total = tr.duration
    rectified_secs = 0.0
    applied = 0
    pad = 0.3
    for n, span in enumerate(spans):
        segs = [tr.segments[i] for i in span]
        start = segs[0].start - pad
        end = segs[-1].end + pad
        old_conf = (sum(s.confidence * max(0.1, s.end - s.start) for s in segs)
                    / sum(max(0.1, s.end - s.start) for s in segs))
        clip = None
        try:
            clip = _slice_audio(audio_path, start, end, total)
            new_text, new_conf = _retranscribe_clip(big, clip, tr.language
                                                    if tr.language in ("es", "en")
                                                    else None)
            if new_text and new_conf > old_conf + 0.05:
                # acepta: el primer segmento del span lleva el texto; los demás
                # se vacían (la salida es texto plano, no subtítulos).
                segs[0].text = new_text
                segs[0].rectified = True
                for s in segs[1:]:
                    s.text = ""
                    s.rectified = True
                rectified_secs += (segs[-1].end - segs[0].start)
                applied += 1
        except Exception as e:
            logger.warning(f"Rectify span {n} falló: {e}")
        finally:
            if clip and os.path.exists(clip):
                try:
                    os.remove(clip)            # borra el clip de inmediato (F4)
                except OSError:
                    pass
        if progress_cb:
            progress_cb((n + 1) / len(spans))

    tr.meta["rectified_spans"] = applied
    tr.meta["rectified_pct"] = round(100.0 * rectified_secs / total, 1) if total else 0.0
    return tr


# ---------------------------------------------------------------------------
# S4 — Verificación anti-deriva (decide si un trozo pulido es válido)
# ---------------------------------------------------------------------------
def _norm_tokens(s: str):
    """Proyección que colapsa exactamente lo que el corrector PUEDE tocar:
    mayúsculas, tildes, puntuación y espacios. Si tras esto la secuencia de
    palabras cambia, hubo cambio de contenido."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))   # quita tildes
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)            # quita puntuación
    return s.split()


def _number_seq(s: str):
    """Secuencia ordenada de números, normalizando separadores de miles."""
    return re.findall(r"\d+", re.sub(r"(?<=\d)[.,\s](?=\d)", "", s))


def verify_chunk(raw: str, corrected: str):
    """True si `corrected` solo arregla ortografía/puntuación de `raw`."""
    if not corrected.strip():
        return False, "vacío"
    if _norm_tokens(raw) != _norm_tokens(corrected):
        return False, "tokens"
    if _number_seq(raw) != _number_seq(corrected):
        return False, "números"
    return True, "ok"


# ---------------------------------------------------------------------------
# S3 — Pulido con IA (opcional, OFF por defecto, requiere OPENAI_API_KEY)
# ---------------------------------------------------------------------------
_PROOF_SYSTEM = (
    "Eres un CORRECTOR ORTOTIPOGRÁFICO, no un editor. Recibes una transcripción "
    "automática y devuelves el MISMO texto corregido SOLO en ortografía, tildes, "
    "mayúsculas, puntuación y espacios. Reglas estrictas: NUNCA añadas, quites, "
    "reordenes, parafrasees, traduzcas ni completes palabras; conserva todas las "
    "palabras, nombres propios, siglas, números y marcas como [inaudible] tal "
    "cual; no cambies el idioma. Devuelve EXCLUSIVAMENTE un objeto JSON con la "
    'forma {"corrected": "..."} y nada más.'
)
_PROOF_MODEL = "gpt-4o-mini"
_PROOF_CHUNK_CHARS = 4000          # ~1.1k-1.3k tokens de entrada por trozo
_PROOF_CONTEXT_CHARS = 300         # contexto acotado (no el documento entero)


def _chunk_lines(text: str, max_chars: int):
    """Agrupa líneas (≈ segmentos) en trozos de hasta max_chars, sin partir
    líneas (para que S4 pueda mapear de vuelta)."""
    chunks, cur, size = [], [], 0
    for line in text.split("\n"):
        ln = len(line) + 1
        if cur and size + ln > max_chars:
            chunks.append("\n".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += ln
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def proofread_text(text: str, progress_cb=None):
    """S3+S4: pule el texto con gpt-4o-mini y verifica cada trozo (revierte si
    deriva). Devuelve (texto_final, meta). Falla en abierto: ante cualquier
    problema conserva el texto crudo del trozo."""
    meta = {"applied": False, "chunks": 0, "reverted": 0,
            "chars_used": 0, "reason": ""}
    try:
        from core.tts import (is_openai_available, _openai_budget_remaining,
                              _record_openai_usage)
    except Exception:
        meta["reason"] = "tts no disponible"
        return text, meta

    if not is_openai_available():
        meta["reason"] = "sin API key"
        return text, meta

    import json
    from openai import OpenAI
    client = OpenAI()

    chunks = _chunk_lines(text, _PROOF_CHUNK_CHARS)
    meta["chunks"] = len(chunks)
    out_parts, prev_corrected, reverted = [], "", 0

    for idx, raw in enumerate(chunks):
        # Reserva presupuesto ANTES de gastar trabajo (comparte cupo con el TTS).
        if _openai_budget_remaining() < len(raw):
            logger.warning("Presupuesto OpenAI agotado; el resto queda sin pulir.")
            out_parts.append(raw)
            meta["reason"] = "presupuesto agotado"
            continue
        context = prev_corrected[-_PROOF_CONTEXT_CHARS:] if prev_corrected else ""
        user_msg = (
            (f"CONTEXTO (solo lectura, NO lo devuelvas):\n{context}\n\n" if context else "")
            + "Corrige el siguiente fragmento:\n<<<\n" + raw + "\n>>>"
        )
        corrected = None
        try:
            resp = client.chat.completions.create(
                model=_PROOF_MODEL,
                messages=[
                    {"role": "system", "content": _PROOF_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                max_tokens=min(8000, int(len(raw) / 2) + 512),
                response_format={"type": "json_object"},
            )
            choice = resp.choices[0]
            if getattr(choice, "finish_reason", None) == "length":
                raise ValueError("respuesta truncada")
            data = json.loads(choice.message.content)
            corrected = clean_text(str(data.get("corrected", "")))
            _record_openai_usage(len(raw))
            meta["chars_used"] += len(raw)
        except Exception as e:
            logger.warning(f"Pulido IA falló en trozo {idx}: {e}")
            corrected = None

        ok, _reason = (verify_chunk(raw, corrected) if corrected else (False, "error"))
        if ok:
            out_parts.append(corrected)
            prev_corrected = corrected
        else:
            out_parts.append(raw)              # revierte al crudo (fail open)
            prev_corrected = raw
            if corrected is not None:          # solo cuenta deriva real, no fallos de red
                reverted += 1
        if progress_cb:
            progress_cb((idx + 1) / len(chunks))

    meta["reverted"] = reverted
    # Circuit-breaker global: demasiada reversión -> descarta todo el pulido.
    if chunks and reverted / len(chunks) > 0.20:
        meta["applied"] = False
        meta["reason"] = "pulido omitido (baja confianza)"
        return text, meta

    meta["applied"] = meta["chars_used"] > 0
    return "\n".join(out_parts), meta


# ---------------------------------------------------------------------------
# Entrada pública
# ---------------------------------------------------------------------------
def transcribe_audio(audio_path, model_size=DEFAULT_MODEL, language_label="Auto",
                     rectify_on=True, proof_on=False, progress_cb=None):
    """Transcribe `audio_path` y devuelve (texto_limpio, nota_calidad: dict).

    Nunca lanza por contenido: en el peor caso devuelve el texto crudo de S1.
    `language_label` es una clave de TRANSCRIBE_LANGUAGES ("Auto"/"Español"/...).
    """
    if not audio_path or not os.path.isfile(audio_path):
        raise ValueError("No se proporcionó un archivo de audio válido.")
    ext = os.path.splitext(audio_path)[1].lower()
    if ext and ext not in SUPPORTED_EXT:
        raise ValueError(f"Formato no soportado: {ext}.")

    language = TRANSCRIBE_LANGUAGES.get(language_label)

    def sub(lo, hi):
        if progress_cb is None:
            return None
        return lambda f: progress_cb(lo + (hi - lo) * max(0.0, min(1.0, f)))

    # S1
    tr = transcribe_base(audio_path, model_size=model_size, language=language,
                         progress_cb=sub(0.05, 0.55 if rectify_on else 0.92))

    # S2
    if rectify_on:
        try:
            tr = rectify(tr, audio_path, progress_cb=sub(0.55, 0.92))
        except Exception as e:
            logger.warning(f"Rectificación falló (se conserva S1): {e}")
            tr.meta["notes"].append("rectificación omitida por error")

    text = clean_text(tr.text)

    # S3 (+S4)
    proof_meta = {"applied": False, "reverted": 0, "chunks": 0,
                  "chars_used": 0, "reason": ""}
    if proof_on:
        text, proof_meta = proofread_text(text, progress_cb=sub(0.92, 0.99))
        text = clean_text(text)

    note = {
        "model": tr.meta.get("model"),
        "language": tr.language,
        "language_probability": tr.language_probability,
        "duration": tr.duration,
        "rectified_spans": tr.meta.get("rectified_spans", 0),
        "rectified_pct": tr.meta.get("rectified_pct", 0.0),
        "proof_applied": proof_meta.get("applied", False),
        "proof_reverted": proof_meta.get("reverted", 0),
        "proof_chunks": proof_meta.get("chunks", 0),
        "proof_chars_used": proof_meta.get("chars_used", 0),
        "proof_reason": proof_meta.get("reason", ""),
        "notes": tr.meta.get("notes", []),
    }
    if progress_cb:
        progress_cb(1.0)
    return text, note
