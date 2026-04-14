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
from core.tts import synthesize_speech, VOICE_CATALOG, SPEED_PRESETS

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")


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
        gr.Error(str(e))
        return None, None


def handle_extract(video_input, out_format, progress=gr.Progress()):
    """Audio Extract handler."""
    if video_input is None:
        gr.Warning("Sube un archivo de video primero.")
        return None, None

    try:
        progress(0.2, desc="Analizando video...")
        info = get_video_info(video_input)
        logger.info(f"Video info: {info}")

        progress(0.4, desc=f"Extrayendo audio ({out_format.upper()})...")
        extracted = extract_audio(video_input, output_format=out_format)

        # Copy to outputs/
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(video_input))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        final = os.path.join(OUTPUTS_DIR, f"{base}_extracted_{ts}.{out_format}")
        shutil.copy2(extracted, final)

        progress(1.0, desc="Extraccion completada!")
        return final, final

    except Exception as e:
        logger.exception("Extraction failed")
        gr.Error(str(e))
        return None, None


def handle_tts(text_input, voice_name, speed, progress=gr.Progress()):
    """Text-to-Audio handler."""
    if not text_input or not text_input.strip():
        gr.Warning("Escribe algun texto para sintetizar.")
        return None, None

    try:
        progress(0.3, desc="Generando voz con IA neural...")
        output = synthesize_speech(text_input, voice_name, speed)

        # Copy to outputs/
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        final = os.path.join(OUTPUTS_DIR, f"tts_output_{ts}.mp3")
        shutil.copy2(output, final)

        progress(1.0, desc="Voz generada!")
        return final, final

    except Exception as e:
        logger.exception("TTS failed")
        gr.Error(str(e))
        return None, None


# ===================================================================
# UI CONSTRUCTION
# ===================================================================

def build_ui():
    """Build the multi-service Gradio UI with Midnight Luxe design."""

    ai_available = is_model_available()
    status_text = (
        "DeepFilterNet AI model is available" if ai_available
        else "DeepFilterNet not found — FFmpeg fallback"
    )

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
        z-index: 9999;
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
        color: #6B6B7A;
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
        color: #6B6B7A !important;
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
        color: #6B6B7A;
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
        color: #6B6B7A;
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
        .suite-header { flex-direction: column; gap: 1rem; }
    }

    /* ── Footer ── */
    .suite-footer {
        text-align: center;
        padding: 1.5rem 2rem;
        border-top: 1px solid rgba(250,248,245,0.04);
        font-family: "JetBrains Mono", monospace;
        font-size: 0.7rem;
        color: #3A3A45;
        letter-spacing: 0.1em;
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
        block_background_fill="#101015",
        block_background_fill_dark="#101015",
        block_border_color="rgba(250,248,245,0.1)",
        block_border_color_dark="rgba(250,248,245,0.1)",
        panel_background_fill="#0D0D12",
        panel_background_fill_dark="#0D0D12",
    )

    # ------------------------------------------------------------------
    # Build Blocks
    # ------------------------------------------------------------------
    with gr.Blocks(title="Majes de Sivar — Audio Suite") as app:

        # Noise overlay
        gr.HTML('<div class="noise-bg"></div>')

        # ── Card frame start ──
        # Header with brand + status
        gr.HTML(f"""
            <div class="suite-header">
                <span class="suite-brand">Majes de Sivar</span>
                <span class="suite-status">
                    <span class="pulse-dot"></span>
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
                            con <strong>Resemble Enhance</strong> y <strong>DeepFilterNet</strong>.
                            Procesamiento <strong>100% local</strong> — cero latencia,
                            máxima privacidad, control milimétrico sin destruir tu voz.
                        </p>
                    </div>
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        enh_input = gr.Audio(
                            label="Subir Audio",
                            sources=["upload"],
                            type="filepath",
                            elem_id="enh-input",
                        )
                        gr.Markdown(
                            "*Formatos:* `WAV` `MP3` `M4A` `FLAC` `OGG` `AAC`"
                        )
                        enh_engine = gr.Radio(
                            label="Motor de IA",
                            choices=["DeepFilterNet", "Resemble Enhance"],
                            value="DeepFilterNet",
                            elem_id="enh-engine",
                        )

                    with gr.Column(scale=1):
                        enh_output = gr.Audio(
                            label="Audio Mejorado",
                            type="filepath",
                            interactive=False,
                            elem_id="enh-output",
                        )
                        enh_download = gr.File(
                            label="Descargar",
                            elem_id="enh-download",
                        )

                with gr.Column(elem_classes="centered-action"):
                    with gr.Accordion("Configuración avanzada", open=False):
                        enh_strength = gr.Dropdown(
                            label="Intensidad (DeepFilterNet)",
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
                                <div class="step-desc"><strong>Resemble Enhance</strong> o <strong>DeepFilterNet</strong>. Denoising + super-resolución.</div>
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
                            en calidad <strong>PCM 24-bit / 48kHz</strong> —
                            sin recompresión, sin pérdida, sin intermediarios.
                            Potenciado por <strong>FFmpeg 8.0</strong>.
                        </p>
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
                            "*Formatos:* `MP4` `MKV` `AVI` `MOV` `WEBM` `FLV`"
                        )
                        ext_format = gr.Radio(
                            label="Formato de salida",
                            choices=["wav", "flac", "mp3"],
                            value="wav",
                            info="WAV = sin pérdida. FLAC = comprimido sin pérdida. MP3 = comprimido.",
                            elem_id="ext-format",
                        )

                    with gr.Column(scale=1):
                        ext_output = gr.Audio(
                            label="Audio Extraído",
                            type="filepath",
                            interactive=False,
                            elem_id="ext-output",
                        )
                        ext_download = gr.File(
                            label="Descargar",
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
                    outputs=[ext_output, ext_download],
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
                            Motor neural <strong>Azure Cognitive Services</strong> —
                            18+ voces premium en español, inglés, portugués, francés,
                            alemán, italiano y japonés.
                        </p>
                    </div>
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        tts_text = gr.Textbox(
                            label="Texto",
                            placeholder="Escribe o pega el texto que deseas narrar...",
                            lines=8,
                            max_lines=20,
                            elem_id="tts-text",
                        )
                        tts_voice = gr.Dropdown(
                            label="Voz",
                            choices=list(VOICE_CATALOG.keys()),
                            value=list(VOICE_CATALOG.keys())[0],
                            elem_id="tts-voice",
                        )
                        tts_speed = gr.Radio(
                            label="Velocidad",
                            choices=list(SPEED_PRESETS.keys()),
                            value="Normal",
                            elem_id="tts-speed",
                        )

                    with gr.Column(scale=1):
                        tts_output = gr.Audio(
                            label="Audio Generado",
                            type="filepath",
                            interactive=False,
                            elem_id="tts-output",
                        )
                        tts_download = gr.File(
                            label="Descargar",
                            elem_id="tts-download",
                        )

                with gr.Column(elem_classes="centered-action"):
                    tts_btn = gr.Button(
                        "Generar Voz",
                        variant="primary",
                        size="lg",
                    )

                tts_btn.click(
                    fn=handle_tts,
                    inputs=[tts_text, tts_voice, tts_speed],
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
                                <div class="step-name">Neural TTS</div>
                                <div class="step-desc">Modelo neural de <strong>Azure Cognitive</strong> con expresividad natural.</div>
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

        # ── Footer ──
        gr.HTML("""
            <div class="suite-footer">
                MAJES DE SIVAR · AUDIO SUITE · ALL PROCESSING LOCAL · ZERO CLOUD LATENCY
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
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=not is_hf_space,  # No share needed on HF Spaces
        show_error=True,
        theme=theme,
        css=css,
        allowed_paths=[os.path.abspath("assets")],
    )
