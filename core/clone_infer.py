"""
clone_infer.py - OpenVoice tone-color bridge (runs in the ISOLATED clone venv)

Invoked as a subprocess by core/voice_clone.py with the venv's Python, so its
ancient dependencies never touch the main Gradio-6 app. It takes a base voice
audio (produced by the main app's Kokoro TTS) and a reference recording, and
transfers the reference speaker's *timbre* onto the base audio.

Usage:  python clone_infer.py <base_wav> <reference_wav> <output_wav>
"""

import os
import sys


def main():
    if len(sys.argv) != 4:
        print("usage: clone_infer.py <base_wav> <reference_wav> <output_wav>",
              file=sys.stderr)
        return 2
    base_wav, ref_wav, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    import torch  # noqa: F401  (ensures torch is importable in the venv)
    from huggingface_hub import snapshot_download
    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter

    ckpt = snapshot_download("myshell-ai/OpenVoiceV2")
    conv_dir = os.path.join(ckpt, "converter")

    converter = ToneColorConverter(
        os.path.join(conv_dir, "config.json"), device="cpu"
    )
    converter.load_ckpt(os.path.join(conv_dir, "checkpoint.pth"))

    # Speaker embeddings: source (base voice) and target (reference).
    src_se, _ = se_extractor.get_se(base_wav, converter, vad=True)
    tgt_se, _ = se_extractor.get_se(ref_wav, converter, vad=True)

    # Transfer the reference timbre onto the base audio.
    converter.convert(
        audio_src_path=base_wav, src_se=src_se, tgt_se=tgt_se,
        output_path=out_path, message="@MajesDeSivar",
    )
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
