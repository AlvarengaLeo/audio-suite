"""
tts.py - Text-to-Speech engine

Uses Microsoft Edge TTS for ultra-natural voice synthesis.
Provides access to premium neural voices in multiple languages.
"""

import os
import tempfile
import logging
import asyncio

logger = logging.getLogger(__name__)

# Premium voice catalog — curated for maximum naturalness
VOICE_CATALOG = {
    # Spanish voices
    "🇪🇸 Elvira (ES - Femenina)": "es-ES-ElviraNeural",
    "🇪🇸 Alvaro (ES - Masculina)": "es-ES-AlvaroNeural",
    "🇲🇽 Dalia (MX - Femenina)": "es-MX-DaliaNeural",
    "🇲🇽 Jorge (MX - Masculina)": "es-MX-JorgeNeural",
    "🇦🇷 Elena (AR - Femenina)": "es-AR-ElenaNeural",
    "🇨🇴 Salome (CO - Femenina)": "es-CO-SalomeNeural",
    # English voices
    "🇺🇸 Aria (US - Female)": "en-US-AriaNeural",
    "🇺🇸 Guy (US - Male)": "en-US-GuyNeural",
    "🇺🇸 Jenny (US - Female)": "en-US-JennyNeural",
    "🇬🇧 Sonia (UK - Female)": "en-GB-SoniaNeural",
    "🇬🇧 Ryan (UK - Male)": "en-GB-RyanNeural",
    # Portuguese voices
    "🇧🇷 Francisca (BR - Feminina)": "pt-BR-FranciscaNeural",
    "🇧🇷 Antonio (BR - Masculino)": "pt-BR-AntonioNeural",
    # French voices
    "🇫🇷 Denise (FR - Féminine)": "fr-FR-DeniseNeural",
    "🇫🇷 Henri (FR - Masculin)": "fr-FR-HenriNeural",
    # German voices
    "🇩🇪 Katja (DE - Weiblich)": "de-DE-KatjaNeural",
    # Italian voices
    "🇮🇹 Elsa (IT - Femminile)": "it-IT-ElsaNeural",
    # Japanese voices
    "🇯🇵 Nanami (JP - 女性)": "ja-JP-NanamiNeural",
}

# Speed presets
SPEED_PRESETS = {
    "Muy lento": "-30%",
    "Lento": "-15%",
    "Normal": "+0%",
    "Rapido": "+15%",
    "Muy rapido": "+30%",
}


async def _synthesize_async(text: str, voice_id: str, rate: str,
                             output_path: str):
    """Internal async synthesis using edge-tts."""
    import edge_tts

    communicate = edge_tts.Communicate(text=text, voice=voice_id, rate=rate)
    await communicate.save(output_path)


def synthesize_speech(text: str, voice_name: str,
                      speed: str = "Normal") -> str:
    """
    Synthesize speech from text using Edge TTS.

    Args:
        text: The text to convert to speech.
        voice_name: Display name from VOICE_CATALOG.
        speed: Speed preset name from SPEED_PRESETS.

    Returns:
        Path to the generated audio file.
    """
    if not text or not text.strip():
        raise ValueError("Please enter some text to synthesize.")

    if len(text) > 10000:
        raise ValueError(
            f"Text too long ({len(text)} chars). Maximum is 10,000 characters."
        )

    voice_id = VOICE_CATALOG.get(voice_name)
    if not voice_id:
        raise ValueError(f"Unknown voice: {voice_name}")

    rate = SPEED_PRESETS.get(speed, "+0%")

    # Create output file
    temp_file = tempfile.NamedTemporaryFile(
        suffix="_tts_output.mp3", delete=False,
        dir=tempfile.gettempdir()
    )
    output_path = temp_file.name
    temp_file.close()

    logger.info(
        f"TTS: voice={voice_id}, rate={rate}, "
        f"text_len={len(text)} chars"
    )

    try:
        # Run async synthesis in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                _synthesize_async(text, voice_id, rate, output_path)
            )
        finally:
            loop.close()

        file_size_kb = os.path.getsize(output_path) / 1024
        logger.info(f"TTS complete: {output_path} ({file_size_kb:.0f} KB)")
        return output_path

    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise RuntimeError(f"Voice synthesis failed: {e}") from e
