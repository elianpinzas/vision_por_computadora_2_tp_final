"""Funciones comunes a todos los notebooks."""
import random

import numpy as np
import yaml


def load_config(path="config.yaml"):
    """Lee la configuración común del grupo."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """Fija la semilla para que los resultados sean reproducibles."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
