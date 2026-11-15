"""Empaqueta solo lo necesario para subir Memoria Viva a Colab."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
DESTINO = ROOT / "colab" / "memoria_viva_colab.zip"
INCLUIR = ("assets", "src", "metadata", "requirements-colab.txt")


def main() -> None:
    with ZipFile(DESTINO, "w", compression=ZIP_DEFLATED) as archivo_zip:
        for nombre in INCLUIR:
            ruta = ROOT / nombre
            rutas = [ruta] if ruta.is_file() else sorted(ruta.rglob("*"))
            for archivo in rutas:
                if not archivo.is_file() or "__pycache__" in archivo.parts:
                    continue
                archivo_zip.write(archivo, archivo.relative_to(ROOT))
    print(f"Paquete creado: {DESTINO} ({DESTINO.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
