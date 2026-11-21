"""Descarga y preparación de datos y pesos.

Todo se descarga al disco de la máquina virtual de Colab, que se borra al
cerrar la sesión. Por eso cada notebook vuelve a llamar a estas funciones.
Responsable de este archivo: rol Datos (notebook 01).
"""
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

import yaml


def _secret(nombre):
    """Lee un secret de Colab o, fuera de Colab, una variable de entorno."""
    try:
        from google.colab import userdata
        return userdata.get(nombre)
    except ImportError:
        return os.environ[nombre]


def descargar_wildfire(cfg):
    """Descarga de Roboflow la versión fija de Wildfire Smoke, en formato YOLO."""
    rb = cfg["datasets"]["wildfire_smoke"]
    if Path(rb["data_yaml"]).exists():
        return rb["local_dir"]
    from roboflow import Roboflow
    rf = Roboflow(api_key=_secret("ROBOFLOW_API_KEY"))
    project = rf.workspace(rb["workspace"]).project(rb["project"])
    project.version(rb["version"]).download("yolov8", location=rb["local_dir"])
    return rb["local_dir"]


def descargar_dfire(cfg):
    """Descarga D-Fire desde Kaggle (necesita KAGGLE_USERNAME y KAGGLE_KEY)."""
    d = cfg["datasets"]["dfire"]
    if Path(d["raw_dir"]).exists():
        return d["raw_dir"]
    os.environ["KAGGLE_USERNAME"] = _secret("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = _secret("KAGGLE_KEY")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", d["kaggle_slug"],
         "-p", d["raw_dir"], "--unzip"],
        check=True,
    )
    return d["raw_dir"]


def preparar_dfire(cfg):
    """Arma la versión de D-Fire para la prueba externa.

    Del split de test se queda solo con las cajas de humo, renumeradas al
    índice 0, y conserva las imágenes sin humo (con su etiqueta vacía) para
    medir falsos positivos. Devuelve la ruta a su data.yaml.
    """
    d = cfg["datasets"]["dfire"]
    smoke = d["smoke_class_index"]
    if smoke is None:
        raise ValueError("Completar datasets.dfire.smoke_class_index en config.yaml")

    origen_img = Path(d["raw_dir"]) / d["test_images"]
    origen_lbl = Path(d["raw_dir"]) / d["test_labels"]
    destino = Path(d["local_dir"])
    (destino / "images").mkdir(parents=True, exist_ok=True)
    (destino / "labels").mkdir(parents=True, exist_ok=True)

    for img in sorted(origen_img.iterdir()):
        shutil.copy(img, destino / "images" / img.name)
        lbl = origen_lbl / f"{img.stem}.txt"
        lineas = lbl.read_text().splitlines() if lbl.exists() else []
        humo = ["0 " + l.split(maxsplit=1)[1] for l in lineas
                if l.strip() and int(l.split()[0]) == smoke]
        (destino / "labels" / f"{img.stem}.txt").write_text("\n".join(humo))

    data_yaml = Path(d["data_yaml"])
    data_yaml.write_text(yaml.safe_dump({
        "path": str(destino.resolve()),
        "train": "images",  # Ultralytics lo exige, pero con D-Fire no se entrena
        "val": "images",
        "test": "images",
        "nc": 1,
        "names": ["smoke"],
    }))
    return str(data_yaml)


def descargar_pesos(cfg, modelo):
    """Descarga <modelo>_best.pt desde la Release del repo. modelo: 'yolov8' o 'rtdetr'."""
    r = cfg["release"]
    nombre = f"{modelo}_best.pt"
    destino = Path(cfg["paths"]["weights"]) / nombre
    if not destino.exists():
        destino.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/{r['repo']}/releases/download/{r['tag']}/{nombre}"
        urllib.request.urlretrieve(url, destino)
    return str(destino)
