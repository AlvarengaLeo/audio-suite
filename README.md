---
title: Majes de Sivar — Audio Suite
emoji: 🎙️
colorFrom: gray
colorTo: yellow
sdk: docker
pinned: true
license: mit
short_description: AI Audio Enhancement, Extraction & Voice Synthesis
---

# Majes de Sivar — Audio Suite

Suite de audio con tres herramientas: **mejora de audio** (DeepFilterNet / Resemble),
**extracción de audio** desde video (FFmpeg) y **texto a voz** multi-motor.

## 🎙️ Text-to-Audio — motores de voz

La pestaña *Text-to-Audio* ofrece tres motores, de menor a mayor naturalidad:

| Motor | Calidad | Coste | Requisitos |
|-------|---------|-------|------------|
| **Edge Neural (Gratis)** | Buena | Gratis, online | Ninguno |
| **Kokoro HD (Local)** | Muy buena, natural | Gratis, sin coste por uso | Corre local en CPU (Apache-2.0) |
| **OpenAI Studio (Premium)** | Casi humana, con emoción dirigible | De pago (~$0.015/min) | Requiere `OPENAI_API_KEY` |

**Calidad extra (todos los motores):**
- **Normalización de texto en español** — números, moneda (`$19.99`), porcentajes,
  fechas, horas y abreviaturas (`Dr.`, `EE.UU.`) se leen en voz natural, no deletreados.
- **Masterización de audio** (casilla *Masterizar audio*, activada por defecto) —
  recorta silencios y normaliza el volumen a ~−16 LUFS para una voz nítida y pareja.
- **Caché de peticiones** — repetir la misma generación es instantáneo (ahorra CPU y coste).

- **Edge** y **Kokoro** funcionan siempre sin claves ni coste — ideales para un Space público.
- **OpenAI** solo aparece en la interfaz si el dueño del Space configura el secreto
  `OPENAI_API_KEY`. Permite dirigir el tono con instrucciones en lenguaje natural
  ("habla cálido y pausado, como un narrador de documentales") y está limitado a
  2.000 caracteres por petición para proteger el presupuesto.

### Activar el motor premium (opcional)

En el Space: **Settings → Variables and secrets → New secret**
- Name: `OPENAI_API_KEY`
- Value: tu clave de OpenAI

Al reiniciar, el motor *OpenAI Studio* aparecerá automáticamente.

## 🚀 Ejecutar localmente

```bash
pip install -r requirements.txt
# Linux: instala también espeak-ng y ffmpeg (ver packages.txt)
python app.py
```

> En despliegue Docker (HF Spaces) las dependencias de sistema (`ffmpeg`,
> `espeak-ng`, `libsndfile1-dev`) se instalan vía el `Dockerfile`.

## 🔒 Privacidad

El procesamiento de audio (mejora, extracción) y los motores Edge/Kokoro son
locales. El motor **OpenAI Studio** (si el dueño lo activa) envía el texto a la
API de OpenAI para sintetizar la voz; no lo uses con información sensible.

## 📜 Licencia y atribuciones

Código bajo licencia [MIT](LICENSE). Este proyecto usa, entre otros:

- **Kokoro-82M** (hexgrad) — Apache-2.0
- **DeepFilterNet** — MIT/Apache-2.0
- **edge-tts** — servicio de Microsoft Edge (sujeto a sus términos)
- **OpenAI gpt-4o-mini-tts** — API de OpenAI (sujeto a sus términos)
- **pedalboard** (Spotify), **num2words**, **pydub**, **soundfile**, **Gradio**, **FFmpeg**
