"""Inferencia sobre un video de incendio para la demo.

Uso (ejemplo):
    python -m src.demo_video --weights pesos/yolov8/best.pt --video incendio.mp4 --out demo_yolov8.mp4
"""
import argparse

from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Detección de humo sobre video")
    parser.add_argument("--weights", required=True, help="Ruta al best.pt")
    parser.add_argument("--video", required=True, help="Video de entrada (.mp4)")
    parser.add_argument("--out", default="demo.mp4", help="Video de salida con las detecciones")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    conf = cfg["eval"]["conf"]

    # TODO: cargar el modelo (YOLO o RTDETR de ultralytics) con args.weights,
    # recorrer el video cuadro por cuadro, dibujar las cajas con conf >= conf,
    # mostrar los FPS y guardar el resultado en args.out.
    # Opcional: tracking del humo con model.track(...).
    raise NotImplementedError


if __name__ == "__main__":
    main()
