"""Descarga el conjunto inicial de imágenes desde Wikimedia Commons.

Uso:
    python src/descargar_dataset.py
    python src/descargar_dataset.py --forzar

La ficha de Commons y la licencia de cada archivo se conservan en
metadata/imagenes.csv. Revisá la licencia antes de redistribuir cualquier
archivo; la ficha del archivo es la referencia para la atribución.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "MemoriaViva/0.1 (student image-restoration project)"

# Selección curada inicial. Se pueden agregar elementos manteniendo su ficha.
IMAGENES = [
    {
        "titulo": "File:Amoy Woman in Shanghai. Wellcome V0037201.jpg",
        "archivo": "assets/originales_danados/retrato_amoy_danado.jpg",
        "tipo": "foto_real_danada",
        "nota": "Retrato histórico de 1869 con rayaduras; licencia CC BY 4.0.",
    },
    {
        "titulo": "File:Gruppebilde med blant annet Bjørnstjerne Bjørnson sterkt skadet - no-nb digifoto 20160609 00068 bldsa BB1802.jpg",
        "archivo": "assets/originales_danados/retrato_grupal_danado.jpg",
        "tipo": "foto_real_danada",
        "nota": "Fotografía histórica con rayaduras y pérdida de emulsión visibles.",
    },
    {
        "titulo": "File:Charlotte Perkins Gilman c. 1900.jpg",
        "archivo": "assets/referencias_limpias/charlotte_perkins_gilman.jpg",
        "tipo": "referencia_limpia",
        "nota": "Retrato en blanco y negro. Se usará como referencia para simular daños y comparar con la entrada limpia; el archivo de Commons es una restauración publicada, no el negativo original.",
    },
]


class TextoHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.partes: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.partes.append(data.strip())


def texto_plano(valor: str) -> str:
    parser = TextoHTML()
    parser.feed(html.unescape(valor or ""))
    return re.sub(r"\s+", " ", " ".join(parser.partes)).strip()


def consultar_archivo(titulo: str) -> dict:
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "titles": titulo,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|dimensions|sha1",
        }
    )
    request = urllib.request.Request(
        f"{API_URL}?{query}", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        payload = json.load(response)
    pagina = next(iter(payload["query"]["pages"].values()))
    if "missing" in pagina or not pagina.get("imageinfo"):
        raise RuntimeError(f"No se encontró el archivo en Commons: {titulo}")
    info = pagina["imageinfo"][0]
    metadatos = info.get("extmetadata", {})

    def meta(clave: str) -> str:
        return texto_plano(metadatos.get(clave, {}).get("value", ""))

    autor = meta("Artist") or meta("Credit")
    licencia = meta("LicenseShortName")
    atribucion = meta("Attribution")
    if not atribucion:
        atribucion = "; ".join(
            parte
            for parte in (
                f"Autoría/crédito: {autor}" if autor else "Autoría desconocida",
                f"Wikimedia Commons — {licencia}" if licencia else "Wikimedia Commons",
            )
        )
    return {
        "titulo": pagina["title"],
        "url_descarga": info["url"],
        "pagina_fuente": info["descriptionurl"],
        "autor": autor,
        "fecha": meta("DateTimeOriginal") or meta("DateTime"),
        "licencia": licencia,
        "enlace_licencia": meta("LicenseUrl") or info["descriptionurl"],
        "nota_derechos": meta("UsageTerms") or meta("Copyrighted"),
        "atribucion": atribucion,
        "categorias_commons": meta("Categories"),
        "descripcion": meta("ImageDescription"),
        "ancho": info.get("width"),
        "alto": info.get("height"),
        "sha1": info.get("sha1"),
    }


def descargar(url: str, destino: Path, forzar: bool) -> None:
    if destino.exists() and not forzar:
        print(f"Ya existe, se conserva: {destino.relative_to(ROOT)}")
        return
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    destino.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=90) as response:
        destino.write_bytes(response.read())
    print(f"Descargado: {destino.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forzar", action="store_true", help="volver a descargar los archivos")
    args = parser.parse_args()

    registros = []
    for item in IMAGENES:
        datos = consultar_archivo(item["titulo"])
        destino = ROOT / item["archivo"]
        descargar(datos["url_descarga"], destino, args.forzar)
        registros.append(
            {
                "archivo": item["archivo"],
                "tipo": item["tipo"],
                "nota": item["nota"],
                **{k: v for k, v in datos.items() if k != "url_descarga"},
            }
        )

    metadata_dir = ROOT / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "imagenes.json").write_text(
        json.dumps(registros, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    columnas = list(registros[0].keys())
    with (metadata_dir / "imagenes.csv").open("w", newline="", encoding="utf-8-sig") as salida:
        writer = csv.DictWriter(salida, fieldnames=columnas)
        writer.writeheader()
        writer.writerows(registros)
    print(f"Metadatos guardados en {metadata_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
