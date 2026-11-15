"""Crea un ejemplo dañado y una máscara reproducibles desde la referencia limpia."""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
FUENTE = ROOT / "assets/referencias_limpias/charlotte_perkins_gilman.jpg"
SALIDA = ROOT / "assets/sinteticas/charlotte_danada_sintetica.png"
MASCARA = ROOT / "assets/sinteticas/charlotte_mascara_sintetica.png"
SEMILLA = 42


def main() -> None:
    rng = random.Random(SEMILLA)
    original = Image.open(FUENTE).convert("RGB")
    ancho, alto = original.size
    imagen = original.copy()
    mascara = Image.new("L", original.size, 0)
    dibujo = ImageDraw.Draw(imagen)
    marcar = ImageDraw.Draw(mascara)

    # Rayones largos, claros y oscuros, con leves desviaciones.
    for _ in range(14):
        x1 = rng.randrange(ancho)
        y1 = rng.randrange(alto)
        largo = rng.randint(int(alto * 0.08), int(alto * 0.42))
        x2 = max(0, min(ancho - 1, x1 + rng.randint(-45, 45)))
        y2 = max(0, min(alto - 1, y1 + largo))
        grosor = rng.choice([2, 3, 4, 6])
        color = rng.choice([(242, 240, 232), (25, 25, 25), (220, 220, 216)])
        dibujo.line((x1, y1, x2, y2), fill=color, width=grosor)
        marcar.line((x1, y1, x2, y2), fill=255, width=grosor)

    # Polvo y pequeñas pérdidas de emulsión.
    for _ in range(180):
        x, y = rng.randrange(ancho), rng.randrange(alto)
        radio = rng.choice([2, 3, 4, 6, 9])
        caja = (x - radio, y - radio, x + radio, y + radio)
        color = rng.choice([(248, 247, 240), (30, 30, 30), (210, 210, 205)])
        dibujo.ellipse(caja, fill=color)
        marcar.ellipse(caja, fill=255)

    # Un par de manchas mayores con bordes suaves y forma irregular.
    for _ in range(3):
        x, y = rng.randrange(ancho), rng.randrange(alto)
        rx, ry = rng.randint(18, 42), rng.randint(12, 34)
        caja = (x - rx, y - ry, x + rx, y + ry)
        color = rng.choice([(226, 224, 215), (38, 38, 37)])
        dibujo.ellipse(caja, fill=color)
        marcar.ellipse(caja, fill=255)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    imagen.save(SALIDA)
    mascara.save(MASCARA)
    print(f"Imagen sintética: {SALIDA.relative_to(ROOT)}")
    print(f"Máscara binaria: {MASCARA.relative_to(ROOT)}")
    print(f"Semilla reproducible: {SEMILLA}")


if __name__ == "__main__":
    main()
