import numpy as np


def cosine_similarity(a, b):
    """
    Calcula la similitud coseno entre dos embeddings.

    Retorna un valor entre -1 y 1, aunque para este caso normalmente
    se trabaja con valores positivos entre 0 y 1.
    """

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def normalizar_umbral(valor, nombre="umbral"):
    """
    Acepta umbrales en formato decimal o porcentaje.

    Ejemplos validos:
    - 0.70 -> 0.70
    - 70   -> 0.70
    - 90   -> 0.90
    - 0    -> 0.00
    - 100  -> 1.00
    """

    try:
        valor = float(valor)
    except (TypeError, ValueError):
        raise ValueError(f"{nombre} debe ser un numero. Valor recibido: {valor}")

    if valor < 0:
        raise ValueError(f"{nombre} no puede ser menor que 0. Valor recibido: {valor}")

    if valor > 1:
        if valor > 100:
            raise ValueError(f"{nombre} no puede ser mayor que 100. Valor recibido: {valor}")
        valor = valor / 100

    if valor > 1:
        raise ValueError(f"{nombre} no puede ser mayor que 1. Valor recibido: {valor}")

    return valor


def formatear_porcentaje(valor):
    """Convierte un valor decimal 0-1 a porcentaje redondeado."""

    return round(float(valor) * 100, 2)
