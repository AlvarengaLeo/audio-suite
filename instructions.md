Quiero que me generes un proyecto completo en Python para mejorar audio de voz a partir de un archivo que yo suba como input, con una calidad tipo “podcast clean voice”, similar en intención a Adobe Podcast, pero usando herramientas open source y, si es posible, AI.

Objetivo:
Crear una app local con interfaz simple donde el usuario pueda:
1. Subir un archivo de audio (wav, mp3, m4a, flac).
2. Procesarlo automáticamente para mejorar claridad de voz.
3. Escuchar una vista previa del audio mejorado.
4. Descargar el archivo final procesado.

Requisitos técnicos:
- Lenguaje: Python 3.11+
- UI simple y rápida: preferiblemente Gradio
- Usar AI para speech enhancement / noise reduction si es posible
- Usar FFmpeg para preprocesamiento y postprocesamiento
- Priorizar voz humana hablada (podcasts, notas de voz, entrevistas, reels hablados)
- Debe funcionar localmente
- Debe generar un resultado natural, evitando voz metálica o demasiado procesada

Pipeline deseado:
1. Cargar audio de entrada
2. Convertir a un formato de trabajo uniforme
   - mono o estéreo según convenga
   - sample rate apropiado para el modelo
3. Aplicar mejora por IA
   - preferiblemente un modelo open source para speech enhancement / denoise
   - si el audio tiene ruido de fondo, reducirlo
   - si tiene leve reverberación, intentar suavizarla sin destruir la voz
4. Postproceso estilo podcast
   - high-pass suave para limpiar graves innecesarios
   - ecualización ligera para mejorar inteligibilidad de voz
   - compresión ligera
   - loudness normalization final
5. Entregar archivo final descargable

Comportamiento esperado:
- Si falla el modelo AI, usar un fallback solo con FFmpeg para al menos limpiar, ecualizar y normalizar.
- Mostrar mensajes de error claros.
- Mantener nombres de archivos limpios.
- Guardar el output en una carpeta /outputs.
- No sobrescribir el archivo original.

Quiero que implementes:
- app.py
- requirements.txt
- README.md
- estructura de carpetas recomendada
- funciones separadas por responsabilidad
- comentarios útiles en el código
- validación de tipos de archivo
- manejo de errores
- botón de descarga del resultado
- reproductor de audio para preview antes y después

Detalles de la UI:
- Título: “AI Audio Enhancer”
- Subtítulo: “Upload your voice audio, enhance it, preview it, and download the cleaned version.”
- Input de archivo/audio
- Botón “Enhance Audio”
- Reproductor para audio original
- Reproductor para audio mejorado
- Botón o componente para descargar el archivo final

Deseo que el proyecto quede listo para ejecutar con:
pip install -r requirements.txt
python app.py

Quiero que tomes decisiones técnicas razonables sin preguntarme más cosas.
Si eliges una librería o modelo AI, explica brevemente por qué la elegiste.
Si necesitas FFmpeg instalado, indícalo en el README.
Entrega el código completo, no pseudocódigo.