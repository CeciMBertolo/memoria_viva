"""Interfaz Gradio para restaurar daños y proponer color en fotos antiguas.

Los modelos se cargan al iniciar la app. Se espera ejecutarla en Google Colab
con GPU; para desarrollo local, el modo CPU offload reduce VRAM a costa de
velocidad.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image
from transformers import pipeline


ROOT = Path(os.environ.get("MEMORIA_VIVA_ROOT", Path(__file__).resolve().parents[1]))
DEVICE = 0 if torch.cuda.is_available() else -1
INPAINT_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"
COLOR_MODEL = "piddnad/ddcolor_modelscope"
COLOR_CODE = "https://github.com/piddnad/DDColor.git"
COLOR_CODE_DIR = Path("/content/DDColor")
VISION_MODEL = "PekingU/rtdetr_r50vd"
MAX_SIDE = 768
SEED = 17


def ajustar_tamano(imagen: Image.Image, limite: int = MAX_SIDE) -> Image.Image:
    imagen = imagen.convert("RGB")
    escala = min(1.0, limite / max(imagen.size))
    ancho = max(8, int(imagen.width * escala) // 8 * 8)
    alto = max(8, int(imagen.height * escala) // 8 * 8)
    return imagen.resize((ancho, alto), Image.Resampling.LANCZOS)


def cargar_ejemplo(nombre: str):
    archivos = {
        "Retrato Amoy con rayaduras": ROOT / "assets/originales_danados/retrato_amoy_danado.jpg",
        "Foto sintética (permite comparar con referencia)": ROOT / "assets/sinteticas/charlotte_danada_sintetica.png",
        "Foto grupal muy dañada": ROOT / "assets/originales_danados/retrato_grupal_danado.jpg",
        "Retrato limpio para colorizar": ROOT / "assets/referencias_limpias/charlotte_perkins_gilman.jpg",
    }
    ruta = archivos.get(nombre)
    if ruta is None or not ruta.exists():
        raise gr.Error(f"No encontré el ejemplo: {ruta}")
    fondo = ajustar_tamano(Image.open(ruta))
    capas: list[Image.Image] = []
    if nombre.startswith("Foto sintética"):
        ruta_mascara = ROOT / "assets/sinteticas/charlotte_mascara_sintetica.png"
        mascara = Image.open(ruta_mascara).convert("L").resize(fondo.size, Image.Resampling.NEAREST)
        capa = Image.new("RGBA", fondo.size, (255, 55, 40, 0))
        capa.putalpha(mascara)
        capas = [capa]
    return {"background": fondo, "layers": capas, "composite": fondo}


def preparar_foto_subida(foto: Image.Image | None):
    """Pasa una foto propia al editor sin máscara; el usuario decide qué reparar."""
    if foto is None:
        raise gr.Error("Primero subí una foto desde el control de abajo.")
    fondo = ajustar_tamano(foto)
    return {"background": fondo, "layers": [], "composite": fondo}


def mascara_desde_editor(valor: dict | None, tamano: tuple[int, int]) -> Image.Image:
    mascara = Image.new("L", tamano, 0)
    if not valor:
        return mascara
    capas = valor.get("layers") or []
    for capa in capas:
        if capa is None:
            continue
        if not isinstance(capa, Image.Image):
            capa = Image.fromarray(np.asarray(capa))
        if capa.size != tamano:
            capa = capa.resize(tamano, Image.Resampling.NEAREST)
        if capa.mode == "RGBA":
            alpha = capa.getchannel("A")
            # Las pinceladas transparentes no cuentan; las zonas pintadas son blancas.
            pixeles = np.asarray(alpha) > 8
        else:
            pixeles = np.asarray(capa.convert("L")) > 8
        mascara_np = np.maximum(np.asarray(mascara), pixeles.astype(np.uint8) * 255)
        mascara = Image.fromarray(mascara_np.astype(np.uint8))
    return mascara


@lru_cache(maxsize=1)
def cargar_modelos():
    if not torch.cuda.is_available():
        raise RuntimeError("No se detectó GPU CUDA. En Colab elegí Entorno de ejecución > Cambiar tipo de entorno > GPU.")
    dtype = torch.float16
    reparar = AutoPipelineForInpainting.from_pretrained(INPAINT_MODEL, torch_dtype=dtype)
    reparar.enable_model_cpu_offload()
    colorizador = cargar_colorizador()
    detector = pipeline("object-detection", model=VISION_MODEL, device=DEVICE)
    return reparar, colorizador, detector


def cargar_colorizador():
    """Carga DDColor, modelo especializado en colorizar fotos en blanco y negro."""
    if not COLOR_CODE_DIR.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", COLOR_CODE, str(COLOR_CODE_DIR)],
            check=True,
            capture_output=True,
            text=True,
        )
    ruta_codigo = str(COLOR_CODE_DIR)
    if ruta_codigo not in sys.path:
        sys.path.insert(0, ruta_codigo)

    from ddcolor import ColorizationPipeline, DDColor, build_ddcolor_model
    from huggingface_hub import hf_hub_download

    pesos = hf_hub_download(repo_id=COLOR_MODEL, filename="pytorch_model.bin")
    dispositivo = torch.device("cuda")
    modelo = build_ddcolor_model(
        DDColor,
        model_path=pesos,
        input_size=512,
        model_size="large",
        device=dispositivo,
    )
    return ColorizationPipeline(modelo, input_size=512, device=dispositivo)


def fusionar_luminancia(
    original: Image.Image, propuesta: Image.Image, intensidad: float
) -> Image.Image:
    """Usa el color propuesto por DDColor y conserva la luminancia original."""
    original_np = np.asarray(original.convert("RGB"))
    propuesta_np = np.asarray(propuesta.convert("RGB"))
    lab_original = cv2.cvtColor(original_np, cv2.COLOR_RGB2LAB)
    lab_propuesta = cv2.cvtColor(propuesta_np, cv2.COLOR_RGB2LAB)
    ab_original = lab_original[:, :, 1:3].astype(np.float32)
    ab_propuesto = lab_propuesta[:, :, 1:3].astype(np.float32)
    ab_final = ab_original + float(intensidad) * (ab_propuesto - ab_original)
    lab_final = np.dstack(
        (lab_original[:, :, 0], np.clip(np.rint(ab_final), 0, 255).astype(np.uint8))
    )
    return Image.fromarray(cv2.cvtColor(lab_final, cv2.COLOR_LAB2RGB))


def mapa_diferencias(antes: Image.Image, despues: Image.Image) -> Image.Image:
    a = np.asarray(antes.convert("RGB"), dtype=np.int16)
    b = np.asarray(despues.convert("RGB"), dtype=np.int16)
    gris = np.abs(a - b).mean(axis=2).clip(0, 255).astype(np.uint8)
    return Image.fromarray(cv2.applyColorMap(gris, cv2.COLORMAP_INFERNO)[:, :, ::-1])


def describir_detecciones(detector, imagen: Image.Image) -> list[dict]:
    resultados = detector(imagen)
    return [d for d in resultados if d.get("score", 0) >= 0.5]


def procesar(
    editor: dict | None,
    reparar_dano: bool,
    colorizar: bool,
    intensidad_reparacion: float,
    intensidad_color: float,
):
    if not editor or editor.get("background") is None:
        raise gr.Error("Subí una foto o cargá uno de los ejemplos antes de procesar.")
    reparar_pipe, colorizador, detector = cargar_modelos()
    original = ajustar_tamano(editor["background"])
    mascara = mascara_desde_editor(editor, original.size)
    reconstruida = original

    if reparar_dano and np.asarray(mascara).max() > 0:
        salida = reparar_pipe(
            prompt="restore the marked damaged areas of this old black and white photograph; preserve the people, clothing, pose and original composition; monochrome black and white photo",
            negative_prompt="new person, extra objects, color, text, watermark, changed face, distorted face, cartoon",
            image=original,
            mask_image=mascara,
            # Diffusers usa 512x512 por defecto si no se especifican dimensiones;
            # conservar el tamaño evita incompatibilidad al recomponer la máscara.
            height=original.height,
            width=original.width,
            strength=float(intensidad_reparacion),
            num_inference_steps=24,
            guidance_scale=6.5,
            generator=torch.Generator(device="cpu").manual_seed(SEED),
        ).images[0].convert("RGB")
        # Los píxeles sin máscara quedan exactamente como estaban.
        if salida.size != original.size:
            salida = salida.resize(original.size, Image.Resampling.LANCZOS)
        mask_np = np.asarray(mascara)[:, :, None] > 0
        reconstruida = Image.fromarray(np.where(mask_np, np.asarray(salida), np.asarray(original)).astype(np.uint8))

    final = reconstruida
    if colorizar:
        entrada_bgr = cv2.cvtColor(np.asarray(reconstruida.convert("RGB")), cv2.COLOR_RGB2BGR)
        propuesta_bgr = colorizador.process(entrada_bgr)
        propuesta = Image.fromarray(cv2.cvtColor(propuesta_bgr, cv2.COLOR_BGR2RGB))
        if propuesta.size != reconstruida.size:
            propuesta = propuesta.resize(reconstruida.size, Image.Resampling.LANCZOS)
        final = fusionar_luminancia(reconstruida, propuesta, intensidad_color)

    antes_det = describir_detecciones(detector, original)
    despues_det = describir_detecciones(detector, final)
    etiquetas_antes = sorted({d["label"] for d in antes_det})
    etiquetas_despues = sorted({d["label"] for d in despues_det})
    lista_antes = ", ".join(etiquetas_antes) if etiquetas_antes else "ningún objeto superó el umbral"
    lista_despues = ", ".join(etiquetas_despues) if etiquetas_despues else "ningún objeto superó el umbral"
    porcentaje_mascara = float(np.asarray(mascara).mean() / 255)
    informe = (
        "### Análisis automático (RT-DETR)\n"
        f"- Antes: {lista_antes}.\n"
        f"- Después: {lista_despues}.\n\n"
        "El detector puede pasar por alto objetos pequeños o confundir detalles. "
        "Este análisis no verifica identidad ni autenticidad.\n\n"
        f"### Qué aportó la IA\n- Zona marcada para reparación: {porcentaje_mascara:.1%} de la imagen.\n"
        f"- Colorización: {'propuesta por DDColor, intensidad ' + format(intensidad_color, '.0%') if colorizar else 'no solicitada'}. "
        "Los colores son estimaciones generadas, no información recuperada del original.\n\n"
        "> Revisá especialmente rostros, ropa y objetos antes de compartir. El resultado no es evidencia histórica."
    )
    comparacion = [(original, "Original"), (reconstruida, "Reparación"), (final, "Resultado")]
    archivo = Path(tempfile.gettempdir()) / "memoria_viva_resultado.png"
    final.save(archivo)
    return comparacion, mascara, mapa_diferencias(original, final), informe, str(archivo)


def crear_app():
    ejemplos = [
        "Retrato Amoy con rayaduras",
        "Foto sintética (permite comparar con referencia)",
        "Foto grupal muy dañada",
        "Retrato limpio para colorizar",
    ]
    with gr.Blocks(title="Memoria Viva | Restauración de fotos") as demo:
        gr.Markdown(
            "# Memoria Viva\n"
            "Elegí un ejemplo del proyecto o subí una foto propia; repará zonas marcadas "
            "y probá una colorización generada por IA. "
            "**La IA puede inventar detalles y colores.**"
        )
        with gr.Row():
            with gr.Column(scale=1):
                ejemplo = gr.Dropdown(ejemplos, label="Ejemplo del proyecto", value=None)
                cargar = gr.Button("Cargar ejemplo")
                foto_subida = gr.Image(
                    label="O subí una foto propia desde tu dispositivo",
                    type="pil",
                    sources=["upload"],
                    height=180,
                )
                usar_foto = gr.Button("Usar mi foto")
                editor = gr.ImageEditor(
                    label="Foto: pintá encima de rayones o manchas para crear la máscara",
                    type="pil",
                    image_mode="RGBA",
                    height=520,
                    sources=["upload"],
                )
                reparar = gr.Checkbox(value=True, label="Reparar la zona pintada")
                color = gr.Checkbox(value=True, label="Proponer color")
                fuerza_reparacion = gr.Slider(0.2, 1.0, value=0.75, step=0.05, label="Intensidad de reparación")
                fuerza_color = gr.Slider(0.15, 0.45, value=0.28, step=0.01, label="Cuánto puede cambiar la foto al colorizar")
                gr.Markdown(
                    "La colorización la propone DDColor de forma automática; no usa texto como guía. "
                    "La intensidad mezcla los colores estimados con la foto original."
                )
                fuerza_color = gr.Slider(0.2, 1.0, value=0.5, step=0.05, label="Intensidad de los colores propuestos")
                procesar_btn = gr.Button("Procesar foto", variant="primary")
            with gr.Column(scale=2):
                comparacion = gr.Gallery(label="Original, reparación y resultado", columns=3, height=300)
                with gr.Row():
                    mascara_out = gr.Image(label="Máscara de reparación", height=260)
                    diferencias = gr.Image(label="Mapa de diferencias", height=260)
                informe = gr.Markdown()
                descarga = gr.File(label="Descargar resultado")
        cargar.click(cargar_ejemplo, inputs=ejemplo, outputs=editor)
        usar_foto.click(preparar_foto_subida, inputs=foto_subida, outputs=editor)
        procesar_btn.click(
            procesar,
            inputs=[editor, reparar, color, fuerza_reparacion, fuerza_color],
            outputs=[comparacion, mascara_out, diferencias, informe, descarga],
        )
        gr.Markdown(
            "**Uso:** subí o cargá una foto; pintá las zonas dañadas; elegí reparación y/o colorización; "
            "procesá y revisá la comparación, la máscara, el mapa y las detecciones."
        )
    return demo


if __name__ == "__main__":
    crear_app().queue().launch(share=True)
