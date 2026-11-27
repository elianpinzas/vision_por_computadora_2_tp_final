# Detección de humo de incendios forestales: YOLOv8 vs RT-DETR

Trabajo práctico de **Visión por Computadora II**, Carrera de Especialización en Inteligencia Artificial (CEIA), FIUBA, 2026. Grupo 2.

## Objetivo

Comparar dos detectores pre-entrenados en COCO, **YOLOv8** (CNN de una etapa) y **RT-DETR** (basado en transformers), para detectar humo de incendios forestales en imágenes de cámaras fijas, y medir cuánto cae su rendimiento cuando se los prueba, sin reentrenar, con imágenes de otra fuente.

## Datasets

| Dataset | Uso | Link |
|---|---|---|
| Wildfire Smoke (Roboflow), cámaras HPWREN, clase *smoke* | Entrenamiento, validación y test | https://universe.roboflow.com/brad-dwyer/wildfire-smoke |
| D-Fire (Brasil), *smoke* y *fire*, incluye imágenes sin ninguno | Prueba externa, sin reentrenar | https://github.com/gaiasd/DFireDataset |

Los datasets no se suben al repo: los notebooks los descargan en cada sesión de Colab con las funciones de `src/datos.py`.

**División por cámara.** No se usa el split original de Roboflow porque tiene imágenes casi idénticas entre train, valid y test (fotos de la misma cámara con segundos de diferencia), lo que infla las métricas. Las imágenes se agrupan en 19 cámaras y se reparten cámaras enteras: train 516 imágenes (13 cámaras), valid 146 (3) y test 75 (3). La asignación está en `results/split_por_camara.csv` y la genera `notebooks/00_division_eventos.ipynb`. Para entrenar o evaluar se usa `division.preparar_wildfire_por_camara(cfg, DATA_DIR)`, que arma `datasets/wildfire_smoke_camara/data.yaml`.

## Estructura

```
├── config.yaml          # configuración común (hiperparámetros, versiones, rutas)
├── requirements.txt
├── notebooks/
│   ├── 00_division_eventos.ipynb
│   ├── 01_datos_eda.ipynb
│   ├── 02_train_yolov8.ipynb
│   ├── 03_train_rtdetr.ipynb
│   └── 04_evaluacion.ipynb
├── src/
│   ├── datos.py         # descarga y preparación de datasets y pesos
│   ├── division.py      # división del dataset por cámara (sin fuga entre splits)
│   ├── utils.py         # carga de config y semilla
│   └── demo_video.py    # inferencia sobre video para la demo
├── results/             # tablas y figuras finales
│   ├── split_por_camara.csv   # split asignado a cada imagen
│   └── division/              # figuras de la división
└── paper/               # informe en formato IEEE
```

## Cómo correrlo

Cada notebook se abre directamente en Google Colab (con GPU T4):

| Notebook | |
|---|---|
| 00 · División por cámara | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elianpinzas/vision_por_computadora_2_tp_final/blob/main/notebooks/00_division_eventos.ipynb) |
| 01 · Datos y EDA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elianpinzas/vision_por_computadora_2_tp_final/blob/main/notebooks/01_datos_eda.ipynb) |
| 02 · Entrenamiento YOLOv8 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elianpinzas/vision_por_computadora_2_tp_final/blob/main/notebooks/02_train_yolov8.ipynb) |
| 03 · Entrenamiento RT-DETR | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elianpinzas/vision_por_computadora_2_tp_final/blob/main/notebooks/03_train_rtdetr.ipynb) |
| 04 · Evaluación | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elianpinzas/vision_por_computadora_2_tp_final/blob/main/notebooks/04_evaluacion.ipynb) |

Requisitos: cargar en los *Secrets* de Colab (ícono de llave) `ROBOFLOW_API_KEY` y, para D-Fire, `KAGGLE_USERNAME` y `KAGGLE_KEY` (se obtienen en kaggle.com, en Settings → API).

## Pesos entrenados

Los pesos (`yolov8_best.pt`, `rtdetr_best.pt`) y sus `results.csv` y `args.yaml` están en las [Releases](https://github.com/elianpinzas/vision_por_computadora_2_tp_final/releases) del repo. El notebook de evaluación y la demo los descargan solos.

## Integrantes

- Carmen María Rodríguez Pastrano
- Francisco Julio Facchini Strohmayer
- Elian Ricardo Pinzas
- Gonzalo Martín Rodríguez
