"""
voice_clone.py - Voice cloning on CPU with OpenVoice V2 (MyShell, MIT)

Clones a speaker's timbre from a reference recording and makes it say new text,
running entirely on the free CPU of the Space (no GPU / no paid API).

How it works (OpenVoice V2):
  1. A base voice speaks the text via MeloTTS (fast on CPU, native Spanish).
  2. `ToneColorConverter` transfers the *tone color* (timbre) of the reference
     speaker onto that base audio.

HONEST SCOPE (senior ML): this gives **good speaker similarity** — clearly
recognizable as the person — but NOT an identical copy nor SOTA-API quality
(that needs a GPU). Longer + cleaner reference (30-60s, mono, no music/noise)
improves similarity.

All heavy imports are guarded so `app.py` imports fine on a machine without the
models installed; `is_clone_available()` gates the UI.
"""

import os
import tempfile
import logging

logger = logging.getLogger(__name__)

# Per-request limits (protect the shared CPU / disk on a public Space)
MIN_REF_SECONDS = 5
MAX_REF_SECONDS = 120
MAX_TEXT_CHARS = 1000

# MeloTTS language code (Spanish-first). Override with CLONE_LANG env.
_LANG = os.environ.get("CLONE_LANG", "ES")

# Lazy, cached singletons (loading is heavy; do it once per process).
_ckpt_dir = None
_converter = None
_melo = None


def is_clone_available() -> bool:
    """True only where the cloning stack is installed (OpenVoice + MeloTTS)."""
    try:
        import openvoice  # noqa: F401
        import melo  # noqa: F401
        return True
    except Exception:
        return False


def _ckpt_path() -> str:
    """OpenVoice V2 checkpoints dir (downloaded + cached from the HF Hub)."""
    global _ckpt_dir
    if _ckpt_dir is None:
        from huggingface_hub import snapshot_download
        _ckpt_dir = snapshot_download("myshell-ai/OpenVoiceV2")
    return _ckpt_dir


def _get_converter():
    global _converter
    if _converter is None:
        from openvoice.api import ToneColorConverter
        d = _ckpt_path()
        _converter = ToneColorConverter(
            os.path.join(d, "converter", "config.json"), device="cpu"
        )
        _converter.load_ckpt(os.path.join(d, "converter", "checkpoint.pth"))
        logger.info("OpenVoice ToneColorConverter loaded (CPU).")
    return _converter


def _get_melo(lang: str):
    global _melo
    if _melo is None:
        from melo.api import TTS
        _melo = TTS(language=lang, device="cpu")
        logger.info(f"MeloTTS base voice loaded (lang={lang}, CPU).")
    return _melo


def _new_tempfile(suffix: str) -> str:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False,
                                    dir=tempfile.gettempdir())
    path = f.name
    f.close()
    return path


def _ref_duration_seconds(path: str) -> float:
    try:
        import soundfile as sf
        info = sf.info(path)
        return info.frames / float(info.samplerate or 1)
    except Exception:
        return 0.0


def clone_voice(reference_path: str, text: str,
                lang: str = None, denoise: bool = True) -> str:
    """
    Clone the voice in `reference_path` and make it say `text`.

    Returns the path to the generated WAV. Raises ValueError on bad input.
    """
    if not reference_path or not os.path.isfile(reference_path):
        raise ValueError("Sube un audio de referencia para clonar la voz.")
    if not text or not text.strip():
        raise ValueError("Escribe el texto que dirá la voz clonada.")
    text = text.strip()
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(
            f"Texto demasiado largo ({len(text)} caracteres). "
            f"El máximo para clonación es {MAX_TEXT_CHARS:,}."
        )

    lang = lang or _LANG
    import torch
    from openvoice import se_extractor

    # 1) Normalize the reference to a clean 48 kHz mono WAV.
    from core.converter import convert_to_working_format
    ref_wav = convert_to_working_format(reference_path)

    dur = _ref_duration_seconds(ref_wav)
    if dur and dur < MIN_REF_SECONDS:
        raise ValueError(
            f"La referencia es muy corta ({dur:.0f}s). Sube al menos "
            f"{MIN_REF_SECONDS}s (idealmente 30-60s) de voz limpia."
        )

    # 2) Optionally denoise the reference for better similarity.
    if denoise:
        try:
            from core.enhancer import enhance_with_ai, is_model_available
            if is_model_available():
                ref_wav = enhance_with_ai(ref_wav, strength="balanced",
                                          engine="DeepFilterNet")
        except Exception as e:
            logger.warning(f"Reference denoise skipped: {e}")

    converter = _get_converter()

    # 3) Target speaker embedding (timbre) from the reference.
    target_se, _ = se_extractor.get_se(ref_wav, converter, vad=True)

    # 4) Base voice speaks the text (MeloTTS, Spanish).
    melo = _get_melo(lang)
    spk2id = melo.hps.data.spk2id
    speaker_key = list(spk2id.keys())[0]
    speaker_id = spk2id[speaker_key]
    src_path = _new_tempfile("_clone_src.wav")
    melo.tts_to_file(text, speaker_id, src_path, speed=1.0)

    # 5) Apply the reference timbre onto the base audio.
    source_se = torch.load(
        os.path.join(_ckpt_path(), "base_speakers", "ses",
                     f"{speaker_key.lower().replace('_', '-')}.pth"),
        map_location="cpu",
    )
    out_path = _new_tempfile("_clone.wav")
    converter.convert(
        audio_src_path=src_path, src_se=source_se, tgt_se=target_se,
        output_path=out_path, message="@MajesDeSivar",
    )
    return out_path
