# Memoria Viva

Memoria Viva es un MVP para explorar la restauración de fotografías antiguas con inteligencia artificial. Permite reparar zonas que la persona marca, proponer una colorización para imágenes en blanco y negro y comparar los cambios con la entrada. La idea central es hacer visible que una imagen atractiva no necesariamente es una reconstrucción históricamente fiel.

> **Aviso:** la reparación y los colores son resultados generados por modelos. No recuperan información perdida ni sirven como evidencia de cómo era la fotografía original.

## Problema y caso de uso

Muchas familias conservan fotos impresas que se digitalizaron con rayones, manchas, roturas o pérdida de emulsión. Una persona puede querer mejorar una copia para verla o compartirla, pero las herramientas automáticas suelen mostrar el resultado sin explicar qué se alteró. Memoria Viva propone una experiencia sencilla para señalar el daño, elegir reparación y/o colorización, y revisar el antes y el después con algunas señales sobre el aporte de la IA.

El MVP está pensado para exploración y aprendizaje; no es una herramienta de conservación profesional ni certifica la autenticidad de una imagen.

## Persona usuaria

La persona usuaria de diseño es **Marta, de alrededor de 60 años**, una familiar que digitaliza fotos guardadas en álbumes y cajas. Puede usar una computadora o un celular, pero no conoce los modelos generativos. Quiere mejorar la legibilidad de una foto para compartirla y, al mismo tiempo, le preocupa que una aplicación cambie el rostro, la ropa u otros detalles sin avisarle. Por eso la interfaz prioriza controles directos, comparación visual y advertencias en lenguaje común. Marta es una persona hipotética para orientar el diseño, no el resultado de una investigación con usuarios.

## Cómo funciona

1. La persona carga uno de los ejemplos del proyecto o sube una imagen propia.
2. Si quiere reparar, pinta con el editor las zonas dañadas. Esa capa pintada se convierte en una máscara: blanco indica el área seleccionada y negro el resto.
3. Puede activar reparación, colorización o ambas. La reparación usa la máscara; la colorización se aplica a la imagen completa.
4. La interfaz muestra la entrada, la salida de reparación y el resultado final, además de la máscara, el mapa de diferencias y etiquetas detectadas antes y después.
5. La persona revisa el resultado y puede descargarlo.

Si se usa solamente colorización, no hace falta pintar una máscara. Para probar solo reparación, se desactiva “Proponer color”.

## Arquitectura

```text
Foto de ejemplo o foto subida
             │
             ├── Máscara pintada (opcional)
             │       └── Stable Diffusion 1.5 Inpainting
             │           Los píxeles fuera de la máscara se conservan
             │           en la etapa de reparación
             │
             ├── DDColor (opcional)
             │       └── Propone cromaticidad; se mezcla con la entrada
             │           conservando su luminancia
             │
             └── Resultado
                     ├── comparación: original / reparación / final
                     ├── máscara y mapa de diferencias
                     ├── detección de objetos RT-DETR antes y después
                     └── descarga PNG e informe de advertencias
```

La app está en `src/gradio_app.py`. El notebook de `colab/memoria_viva_colab.ipynb` prepara el entorno, localiza la carpeta del proyecto en Drive o permite subir el ZIP, instala las dependencias de la app, carga los modelos y publica una interfaz Gradio temporal. Los pesos se descargan desde Hugging Face en la primera inferencia; el código de DDColor se clona desde su repositorio de GitHub.

## Modelos, tecnologías y hardware

| Componente | Identificador / versión usada | Función | Licencia del modelo |
|---|---|---|---|
| Inpainting | [`stable-diffusion-v1-5/stable-diffusion-inpainting`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting) | Completar la zona marcada | CreativeML OpenRAIL-M |
| Colorización | [`piddnad/ddcolor_modelscope`](https://huggingface.co/piddnad/ddcolor_modelscope) (DDColor) | Estimar colores para la foto | Apache-2.0 |
| Detección | [`PekingU/rtdetr_r50vd`](https://huggingface.co/PekingU/rtdetr_r50vd) (RT-DETR) | Proponer etiquetas de objetos antes y después | Apache-2.0 |

El código solicita los identificadores indicados y usa la revisión predeterminada del repositorio al descargar los modelos; **no fija un hash de commit**. Por eso, una descarga futura podría resolver a pesos distintos. Revisá las fichas y licencias de cada modelo antes de reutilizarlo fuera de este trabajo.

La interfaz usa Python, Gradio, PyTorch, Diffusers, Transformers, OpenCV, NumPy y Pillow. El `pyproject.toml` define Python 3.12 o superior y las dependencias mínimas para las tareas locales. `requirements-colab.txt` define versiones mínimas para la app en Colab; PyTorch y CUDA se conservan desde el runtime administrado por Colab, así que esas versiones no están fijadas por el proyecto.

**Hardware probado:** Google Colab con GPU NVIDIA T4 para ejecutar el MVP. La primera carga descarga los pesos y puede tardar. La disponibilidad de una T4 depende de Colab y no se garantiza en todas las sesiones. El notebook imprime la versión de Python, PyTorch, CUDA, el modelo de GPU y su memoria disponible. La app se ejecutó en el runtime administrado por Colab; no se considera una prueba completa de inferencia en la computadora local.

## Instalación y ejecución

### Preparar el proyecto con `uv`

Con Python 3.12 y `uv` instalados, desde la raíz del repositorio:

```bash
uv sync
```

Este entorno mínimo sirve para las tareas locales de gestión de imágenes y generación de la prueba sintética:

```bash
uv run python src/descargar_dataset.py
uv run python src/generar_dano_sintetico.py
```

El primer comando actualiza las imágenes y metadatos desde Wikimedia Commons; el segundo vuelve a generar el daño y la máscara sintéticos con semilla fija. `uv sync` **no instala los modelos de difusión ni prepara por sí solo la app completa**. Para la demo con modelos se recomienda el flujo de Colab descrito abajo, que usa su GPU y sus dependencias.

### Ejecutar la app en Google Colab

1. Abrí `colab/memoria_viva_colab.ipynb` en Google Colab.
2. En **Entorno de ejecución → Cambiar tipo de entorno de ejecución**, elegí una GPU si Colab la ofrece; el proyecto se ejecutó con T4.
3. Ejecutá las celdas en orden. El notebook te pide subir `colab/memoria_viva_colab.zip` cuando lo solicite.
4. Esperá a que se instalen las dependencias y se carguen los modelos. Abrí el enlace temporal de Gradio que aparece al final y mantené activa la celda de lanzamiento mientras usás la app.

La VM de Colab es temporal: al terminar la sesión se pueden perder los archivos que quedaron allí. Descargá los resultados que quieras conservar antes de cerrar Colab. El enlace de Gradio también deja de funcionar cuando termina la sesión.

## Ejemplos de uso y registro visual

El notebook [`colab/test_memoria_viva.ipynb`](colab/test_memoria_viva.ipynb) contiene las pruebas documentadas por la autora, con capturas de la interfaz y observaciones. Allí se pueden consultar ejemplos de reparación y colorización. Las capturas muestran resultados de corridas concretas; no son una garantía de que una nueva corrida produzca exactamente la misma salida.

Para repetir un flujo desde la app:

- **Solo reparar:** cargá una foto dañada, pintá una o más rayaduras o manchas, activá “Reparar la zona pintada”, desactivá “Proponer color” y procesá.
- **Solo colorizar:** cargá una foto en blanco y negro, dejá apagada la reparación, activá “Proponer color” y procesá. El control de intensidad mezcla el color estimado con la imagen.
- **Reparar y colorizar:** activá las dos opciones; primero se completa la zona pintada y luego se estima el color de la imagen resultante.

En los tres casos, compará las vistas y leé el informe antes de descargar o compartir el resultado. Para mostrar un límite del modelo en la demo, conviene marcar una zona con información ambigua —por ejemplo, parte de un rostro— y señalar que la apariencia generada no permite saber cómo era el detalle perdido.

## Qué es real y qué es inventado

La foto que se carga, las zonas que la persona marca y los píxeles de la entrada son los datos de partida. Durante la reparación, la app pega de nuevo los píxeles originales que están fuera de la máscara; dentro de la máscara utiliza la salida del modelo de difusión. Ese contenido es una generación plausible, no una recuperación de la fotografía perdida.

La colorización es otra estimación: DDColor propone cromaticidad y el control de intensidad regula cuánto se mezcla esa propuesta. La implementación conserva la luminancia de la imagen de entrada, pero **los colores no se deducen como un hecho histórico**. Como la colorización puede modificar el resultado en toda la foto, el mapa de diferencias puede mostrar cambios fuera de la máscara de reparación.

El ejemplo sintético de Charlotte Perkins Gilman se crea a partir de una imagen limpia disponible en el proyecto, aplicándole rayas y manchas artificiales. Esa referencia permite comparar la entrada dañada con la imagen de partida en una prueba controlada. No es una referencia para las fotos reales: en ellas no se conoce cómo era el contenido borrado o deteriorado. Además, la imagen de Commons usada como referencia se describe en sus metadatos como una versión publicada restaurada, no como el negativo original.

RT-DETR informa etiquetas de objetos detectados antes y después. Esa comparación puede ayudar a observar cambios generales, pero no mide la calidad de la reparación ni verifica rostros, identidad, autenticidad o fidelidad histórica. El mapa de diferencias solo señala cambios de píxeles: tampoco indica si esos cambios son correctos.

## Decisiones de diseño desde HAI

- **Control de la persona:** la máscara se dibuja manualmente y permite limitar dónde se aplica la reparación. La reparación no se ejecuta en zonas no seleccionadas.
- **Revisión antes de compartir:** se presentan juntas la foto original, la etapa de reparación y el resultado final, además de la máscara y el mapa de diferencias.
- **Transparencia sobre la generación:** la interfaz avisa que los detalles y colores pueden ser inventados; el informe indica el porcentaje de imagen seleccionado para reparar y la intensidad de color utilizada.
- **Apoyo automatizado acotado:** RT-DETR se presenta como detector de objetos y se advierte que puede omitir o confundir elementos. No se usa como juez de autenticidad o calidad.
- **Color con intensidad regulable:** la persona decide cuánto aplicar de la propuesta de DDColor. El valor inicial de la interfaz es 0,5; es un punto de partida visual, no una medida de confianza.
- **Cuidado con material personal:** la app corre en una sesión de Colab y crea un enlace temporal de Gradio. Para una demo pública no se recomienda subir fotografías privadas o sensibles.

## Uso de agentes de IA

Codex se usó como asistente para proponer y desarrollar partes del flujo, revisar errores de ejecución, explicar el comportamiento de los modelos y ayudar a ordenar la documentación.

**Decisiones de la autora:** elegir la sub-tarea C; definir la restauración de fotos antiguas como caso de uso; seleccionar y probar imágenes; decidir qué zonas marcar; comparar resultados; elegir DDColor cuando la propuesta de color anterior no se distinguía lo suficiente; ajustar la intensidad inicial de color a 0,5; y mantener avisos claros sobre la información generada. La autora también decidió el formato de su propio cuaderno de pruebas y reflexiones de los resultados.

**Aportes del agente:** sugerencias de estructura, código y modelos; ayuda para diagnosticar errores de interfaz y explicar los resultados. La selección final de ejemplos, controles y conclusiones se revisó con la ejecución del proyecto y quedó a cargo de la autora.

## Limitaciones conocidas

- El inpainting puede alterar o completar rasgos de forma plausible pero distinta de la foto perdida. La máscara muy amplia también le da más espacio para cambiar contenido.
- El resultado depende de la foto, la máscara, los parámetros y las versiones de los modelos y bibliotecas. Los repositorios de modelos no están fijados a un commit en el código.
- La app reduce las imágenes a un máximo de 768 píxeles en el lado más largo para procesarlas; no está pensada para producir copias de archivo a resolución completa.
- DDColor propone colores automáticamente y no recibe una descripción de la escena como guía. En imágenes sepia o ya teñidas el cambio puede ser sutil; una intensidad mayor hace más visible la estimación, no más segura.
- El mapa de diferencias no distingue entre cambios deseados, errores y variaciones de color. La comparación RT-DETR se limita a etiquetas de objetos con suficiente puntaje y puede ser incompleta.
- Solo el ejemplo sintético tiene una imagen de referencia asociada. Las fotografías reales no cuentan con una verdad de terreno que permita medir su fidelidad.
- La app requiere CUDA y descarga modelos desde internet. Una T4, el espacio de VRAM y la duración de la sesión dependen de la disponibilidad de Colab.

## Datos e imágenes

Los ejemplos y su procedencia se registran en `metadata/imagenes.csv` y `metadata/imagenes.json`. Consultá la ficha de cada imagen antes de reutilizarla y conservá la atribución que corresponda.

| Archivo | Uso y procedencia |
|---|---|
| `assets/originales_danados/retrato_amoy_danado.jpg` | Retrato atribuido a John Thomson, ca. 1869. Wikimedia Commons indica CC BY 4.0; requiere atribución. [Ficha](https://commons.wikimedia.org/wiki/File:Amoy_Woman_in_Shanghai._Wellcome_V0037201.jpg). |
| `assets/originales_danados/retrato_grupal_danado.jpg` | Fotografía grupal con daños visibles; los metadatos del proyecto registran dominio público según la ficha de Wikimedia Commons. [Ficha](https://commons.wikimedia.org/wiki/File:Gruppebilde_med_blant_annet_Bj%C3%B8rnstjerne_Bj%C3%B8rnson_sterkt_skadet_-_no-nb_digifoto_20160609_00068_bldsa_BB1802.jpg). |
| `assets/referencias_limpias/charlotte_perkins_gilman.jpg` | Retrato de Charlotte Perkins Gilman, ca. 1900, registrado como dominio público en Commons. Se usa como referencia limpia para producir un daño sintético. [Ficha](https://commons.wikimedia.org/wiki/File:Charlotte_Perkins_Gilman_c._1900.jpg). |
| `assets/sinteticas/charlotte_danada_sintetica.png` | Copia de prueba con rayas y manchas artificiales; la máscara correspondiente está en `assets/sinteticas/charlotte_mascara_sintetica.png`. El daño se genera con semilla 42 mediante `src/generar_dano_sintetico.py`. |

## Estructura del proyecto

```text
memoria viva/
├── assets/
│   ├── originales_danados/
│   ├── referencias_limpias/
│   └── sinteticas/
├── colab/
│   ├── memoria_viva_colab.ipynb   # prepara y lanza la app en Colab
│   ├── memoria_viva_colab.zip     # paquete para subir al runtime
│   └── test_memoria_viva.ipynb    # bitácora de pruebas y capturas
├── metadata/                      # autoría, fuentes y licencias de imágenes
├── src/
│   ├── gradio_app.py
│   ├── descargar_dataset.py
│   └── generar_dano_sintetico.py
├── outputs/                       # salidas locales; no se versionan
├── pyproject.toml
├── requirements-colab.txt
├── uv.lock
└── README.md
```

## Referencias

- [Diffusers: documentación de inpainting](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/inpaint)
- [Ficha de Stable Diffusion 1.5 Inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting)
- [Ficha de DDColor](https://huggingface.co/piddnad/ddcolor_modelscope) y [repositorio de implementación](https://github.com/piddnad/DDColor)
- [Ficha de RT-DETR](https://huggingface.co/PekingU/rtdetr_r50vd)
- [Wikimedia Commons](https://commons.wikimedia.org/) y las fichas individuales de las imágenes enlazadas arriba.
- Cuadernos de la cátedra, Procesamiento Digital de Imágenes, IFTS 24, 2026.

## Autora

**Cecilia Bertolo**  
Tecnicatura Superior en Ciencias de Datos e IA — IFTS 24  
Procesamiento Digital de Imágenes — 2026  
Trabajo Integrador N.º 2 