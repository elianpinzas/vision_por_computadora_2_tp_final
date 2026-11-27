"""División de Wildfire Smoke por cámara.

La división aleatoria de Roboflow deja frames casi idénticos del mismo incendio
(tomados con minutos de diferencia por la misma cámara fija) repartidos entre
train y valid/test, lo que infla las métricas. Acá se agrupan las imágenes por
vista de cámara y se reparten cámaras completas entre los splits.

Todos los parámetros salen de la sección `division` de config.yaml.
Uso: notebooks/00_division_eventos.ipynb.
Responsable: Gonzalo Rodríguez.
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

SPLITS = ["train", "valid", "test"]


# --------------------------------------------------------------------------
# Inventario
# --------------------------------------------------------------------------
def clave(nombre):
    """Nombre original de la imagen: lo que está antes de '.rf.' (Roboflow
    agrega un hash después, que puede cambiar entre descargas)."""
    return nombre.split(".rf.")[0]


def _es_escala_de_grises(im):
    rgb = np.asarray(im.convert("RGB").resize((64, 48)), dtype=np.int16)
    return bool(np.abs(rgb[..., 0] - rgb[..., 1]).mean() < 2
                and np.abs(rgb[..., 1] - rgb[..., 2]).mean() < 2)


def inventario(data_dir):
    """Una fila por imagen de la descarga de Roboflow (formato YOLO)."""
    data_dir = Path(data_dir)
    filas = []
    for s in SPLITS:
        for img in sorted((data_dir / s / "images").glob("*")):
            lbl = data_dir / s / "labels" / f"{img.stem}.txt"
            cajas = [l for l in lbl.read_text().splitlines() if l.strip()] if lbl.exists() else []
            with Image.open(img) as im:
                ancho, alto = im.size
                gris = _es_escala_de_grises(im)
            filas.append({"imagen": img.name, "clave": clave(img.name), "split_original": s,
                          "ruta": str(img), "n_cajas": len(cajas), "ancho": ancho,
                          "alto": alto, "escala_grises": gris})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# Imágenes casi iguales ("gemelas")
# --------------------------------------------------------------------------
def dhash(ruta, lado=16):
    """dHash: imagen en grises reducida a lado x (lado+1); cada bit dice si un
    píxel es más claro que su vecino de la derecha. lado=16 -> 256 bits."""
    a = np.asarray(Image.open(ruta).convert("L").resize((lado + 1, lado)), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def matriz_distancias(df):
    """Distancia de Hamming entre los dHash de todas las imágenes."""
    h = np.array([dhash(r) for r in df.ruta])
    return (h[:, None, :] != h[None, :, :]).sum(-1)


def contar_gemelas(df, dist, col_split, umbral):
    """Cuántas imágenes de valid+test tienen en train una imagen a distancia
    <= umbral. Devuelve (gemelas, total valid+test)."""
    es_train = (df[col_split] == "train").values
    otras, train = np.where(~es_train)[0], np.where(es_train)[0]
    dmin = dist[np.ix_(otras, train)].min(1)
    return int((dmin <= umbral).sum()), len(otras)


def sensibilidad_umbral(df, dist, col_split, umbrales):
    """Tabla de gemelas para varios umbrales (para justificar el valor elegido)."""
    filas = []
    for u in umbrales:
        g, n = contar_gemelas(df, dist, col_split, u)
        filas.append({"umbral": u, "gemelas": g, "valid+test": n, "%": round(100 * g / n, 1)})
    return pd.DataFrame(filas)


def distancia_minima_a_train(df, dist, col_split):
    """Distancia al vecino más cercano de train, para cada imagen de valid+test."""
    es_train = (df[col_split] == "train").values
    return dist[np.ix_(np.where(~es_train)[0], np.where(es_train)[0])].min(1)


# --------------------------------------------------------------------------
# Agrupar por cámara
# --------------------------------------------------------------------------
def descriptor_paisaje(ruta, desde=0.55):
    """Mitad inferior de la imagen (el paisaje, sin el cielo que cambia con
    nubes y humo), en grises, 32x16, normalizada."""
    im = Image.open(ruta).convert("L")
    w, h = im.size
    a = np.asarray(im.crop((0, int(h * desde), w, h)).resize((32, 16), Image.BILINEAR),
                   dtype=np.float32)
    a = (a - a.mean()) / (a.std() + 1e-6)
    return a.flatten()


def agrupar_por_camara(df, umbral):
    """Une las imágenes con similitud coseno del paisaje > umbral
    (componentes conexas). Devuelve el grupo de cada imagen (0 = el más grande)."""
    f = np.array([descriptor_paisaje(r) for r in df.ruta])
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    sim = f @ f.T
    padre = list(range(len(df)))

    def raiz(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    for i, j in np.argwhere(np.triu(sim, 1) > umbral):
        padre[raiz(i)] = raiz(j)
    raices = pd.Series([raiz(i) for i in range(len(df))])
    orden = {r: k for k, r in enumerate(raices.value_counts().index)}
    return raices.map(orden).values


# --------------------------------------------------------------------------
# Reparto de cámaras en splits
# --------------------------------------------------------------------------
def repartir_grupos(df, objetivo, min_grupos, intentos, seed):
    """Prueba asignaciones al azar de grupos completos a train/valid/test y se
    queda con la que deja imágenes y cajas más cerca del objetivo, con al menos
    `min_grupos` grupos en valid y en test. Devuelve {grupo: split}."""
    res = df.groupby("grupo").agg(img=("imagen", "size"), cajas=("n_cajas", "sum"))
    obj = np.array([objetivo[s] for s in SPLITS])
    rng = np.random.default_rng(seed)
    mejor, mejor_costo = None, np.inf
    for _ in range(intentos):
        asig = rng.choice(3, size=len(res), p=obj)
        n = np.bincount(asig, minlength=3)
        if n[1] < min_grupos or n[2] < min_grupos:
            continue
        fi = np.bincount(asig, weights=res.img.values, minlength=3) / res.img.sum()
        fc = np.bincount(asig, weights=res.cajas.values, minlength=3) / res.cajas.sum()
        costo = np.abs(fi - obj).sum() + 0.5 * np.abs(fc - obj).sum()
        if costo < mejor_costo:
            mejor, mejor_costo = asig.copy(), costo
    if mejor is None:
        raise ValueError("No se encontró ninguna asignación válida: bajar min_grupos o subir intentos")
    return {int(g): SPLITS[a] for g, a in zip(res.index, mejor)}


def resumen_division(df, col_split):
    """Cámaras, imágenes, cajas y porcentajes por split."""
    t = df.groupby(col_split).agg(camaras=("grupo", "nunique"), imagenes=("imagen", "size"),
                                  cajas=("n_cajas", "sum"),
                                  escala_grises=("escala_grises", "sum")).reindex(SPLITS)
    t["% imagenes"] = (100 * t.imagenes / t.imagenes.sum()).round(1)
    t["% cajas"] = (100 * t.cajas / t.cajas.sum()).round(1)
    return t


def guardar_csv(df, ruta):
    """Guarda la división (una fila por imagen)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    cols = ["imagen", "clave", "grupo", "split_original", "split_camara", "n_cajas", "escala_grises"]
    df[cols].sort_values(["split_camara", "grupo", "imagen"]).to_csv(ruta, index=False)
    return ruta


# --------------------------------------------------------------------------
# Armar la carpeta del dataset con la nueva división
# --------------------------------------------------------------------------
def aplicar_division(csv, origen, destino):
    """Copia imágenes y etiquetas de la descarga de Roboflow a
    destino/{train,valid,test} según el CSV y escribe su data.yaml."""
    origen, destino = Path(origen), Path(destino)
    tabla = pd.read_csv(csv)
    ubicacion = {clave(p.name): p for s in SPLITS for p in (origen / s / "images").glob("*")}
    for s in SPLITS:
        (destino / s / "images").mkdir(parents=True, exist_ok=True)
        (destino / s / "labels").mkdir(parents=True, exist_ok=True)
    faltan = []
    for _, fila in tabla.iterrows():
        img = ubicacion.get(fila.clave)
        if img is None:
            faltan.append(fila.clave)
            continue
        lbl = img.parent.parent / "labels" / f"{img.stem}.txt"
        shutil.copy(img, destino / fila.split_camara / "images" / img.name)
        if lbl.exists():
            shutil.copy(lbl, destino / fila.split_camara / "labels" / lbl.name)
    data_yaml = destino / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(destino.resolve()), "train": "train/images", "val": "valid/images",
        "test": "test/images", "nc": 1, "names": ["smoke"]}))
    if faltan:
        print(f"Atención: {len(faltan)} imágenes del CSV no están en la descarga")
    return data_yaml


def preparar_wildfire_por_camara(cfg, data_dir):
    """Atajo para los notebooks de entrenamiento: si ya existe el CSV de la
    división, arma la carpeta con esa división y devuelve su data.yaml."""
    d = cfg["division"]
    data_yaml = Path(d["local_dir"]) / "data.yaml"
    if data_yaml.exists():
        return data_yaml
    return aplicar_division(d["csv"], data_dir, d["local_dir"])
