import os
import json
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from recognition_model import generar_embedding, normalizar_url_google_drive
from utils import cosine_similarity, formatear_porcentaje, normalizar_umbral


def escanear_prenda(
    imagen_a_escanear,
    imagenes_referencia=None,
    umbral_minimo=0.70,
    mostrar_logs=True,
    margen_top_matches=0.05,
    max_top_matches=None,
    conjunto_de_prendas=None,
    umbral=None,
    modelo=None
):
    """
    Escanea una imagen JPG directa y la compara contra referencias.

    Contrato principal para Eduardo:
    escanear_prenda(
        imagen_a_escanear=<bytes JPG recibidos por backend>,
        imagenes_referencia=[
            {"id": 1, "url": "https://drive.google.com/uc?id=..."},
            {"id": 2, "url": "https://drive.google.com/file/d/ID/view?usp=drivesdk"}
        ],
        umbral_minimo=90
    )

    Tambien se puede llamar con:
    - umbral_minimo=0.90 o umbral_minimo=90.
    - imagenes_referencia como lista de strings con URLs.
    - imagenes_referencia como lista ya registrada con embeddings.

    Compatibilidad con tu version anterior:
    escanear_prenda(
        imagen_a_escanear=...,
        conjunto_de_prendas=registrar_prendas(...),
        umbral=0.70
    )

    top_matches:
    - Se ordena de mayor a menor score.
    - Por defecto incluye resultados con score >= umbral_minimo - 5 puntos.
      Ejemplo: si el umbral es 70, top_matches muestra scores >= 65.
    - Si quieres que top_matches use exactamente el umbral, manda margen_top_matches=0.
    - Si quieres limitar cantidad, manda max_top_matches=5, 10, etc.
    """

    # Compatibilidad con nombre anterior del parametro.
    if umbral is not None:
        umbral_minimo = umbral

    if imagenes_referencia is None and conjunto_de_prendas is not None:
        imagenes_referencia = conjunto_de_prendas

    if imagenes_referencia is None:
        return {
            "detectado": False,
            "prenda_id": None,
            "similitud": 0,
            "similitud_porcentaje": 0,
            "best_match": None,
            "top_matches": [],
            "referencias_fallidas": [],
            "mensaje": "No se recibieron imagenes de referencia."
        }

    try:
        umbral_decimal = normalizar_umbral(umbral_minimo, nombre="umbral_minimo")
        margen_decimal = normalizar_umbral(margen_top_matches, nombre="margen_top_matches")
    except ValueError as e:
        return {
            "detectado": False,
            "prenda_id": None,
            "similitud": 0,
            "similitud_porcentaje": 0,
            "best_match": None,
            "top_matches": [],
            "referencias_fallidas": [],
            "mensaje": str(e)
        }

    umbral_top_matches = max(0, umbral_decimal - margen_decimal)

    # La imagen a escanear debe llegar como JPG/bytes desde el backend.
    # Se mantiene soporte para ruta local o URL solo para pruebas.
    embedding_nuevo = generar_embedding(
        imagen_a_escanear,
        mostrar_logs=mostrar_logs,
        modelo=modelo
    )

    if embedding_nuevo is None:
        return {
            "detectado": False,
            "prenda_id": None,
            "similitud": 0,
            "similitud_porcentaje": 0,
            "best_match": None,
            "top_matches": [],
            "referencias_fallidas": [],
            "umbral_minimo": round(umbral_decimal, 4),
            "umbral_minimo_porcentaje": formatear_porcentaje(umbral_decimal),
            "umbral_top_matches": round(umbral_top_matches, 4),
            "umbral_top_matches_porcentaje": formatear_porcentaje(umbral_top_matches),
            "mensaje": "No se pudo generar el embedding de la imagen enviada."
        }

    resultados = []
    referencias_fallidas = []

    for indice, referencia_original in enumerate(imagenes_referencia):
        prenda_id = referencia_original.get("id")

        embedding_referencia = referencia_original.get("embedding")

        if embedding_referencia is None:
            referencias_fallidas.append({
                "id": prenda_id,
                "embedding": embedding_referencia,
                "motivo": "No se pudo leer o procesar el embedding de referencia."
            })
            continue
        embedding_referencia = json.loads(embedding_referencia)
        embedding_referencia = np.array(embedding_referencia, dtype=np.float32)
        score = cosine_similarity(embedding_nuevo, embedding_referencia)
        score_redondeado = round(float(score), 4)

        resultado = {
            "id": prenda_id,
            "score": score_redondeado,
            "score_porcentaje": formatear_porcentaje(score)
        }

        resultados.append(resultado)

        if mostrar_logs:
            print(f"Comparando con prenda ID {prenda_id}: {score_redondeado}")

    resultados = sorted(
        resultados,
        key=lambda x: x["score"],
        reverse=True
    )

    if not resultados:
        return {
            "detectado": False,
            "prenda_id": None,
            "similitud": 0,
            "similitud_porcentaje": 0,
            "best_match": None,
            "top_matches": [],
            "referencias_fallidas": referencias_fallidas,
            "umbral_minimo": round(umbral_decimal, 4),
            "umbral_minimo_porcentaje": formatear_porcentaje(umbral_decimal),
            "umbral_top_matches": round(umbral_top_matches, 4),
            "umbral_top_matches_porcentaje": formatear_porcentaje(umbral_top_matches),
            "total_referencias": len(imagenes_referencia),
            "referencias_procesadas": 0,
            "mensaje": "No se pudo procesar ninguna imagen de referencia."
        }

    best_match = resultados[0]
    detectado = best_match["score"] >= umbral_decimal

    top_matches = [
        resultado
        for resultado in resultados
        if resultado["score"] >= umbral_top_matches
    ]

    if max_top_matches is not None:
        top_matches = top_matches[:int(max_top_matches)]

    if mostrar_logs:
        print("Mejor similitud:", best_match["score"])
        print("Mejor prenda ID:", best_match["id"])
        print("Umbral minimo:", umbral_decimal)
        print("Umbral top_matches:", umbral_top_matches)

    return {
        "detectado": detectado,
        "prenda_id": best_match["id"] if detectado else None,
        "similitud": best_match["score"],
        "similitud_porcentaje": best_match["score_porcentaje"],
        "best_match": best_match,
        "top_matches": top_matches,
        "referencias_fallidas": referencias_fallidas,
        "umbral_minimo": round(umbral_decimal, 4),
        "umbral_minimo_porcentaje": formatear_porcentaje(umbral_decimal),
        "umbral_top_matches": round(umbral_top_matches, 4),
        "umbral_top_matches_porcentaje": formatear_porcentaje(umbral_top_matches),
        "total_referencias": len(imagenes_referencia),
        "referencias_procesadas": len(resultados),
        "mensaje": (
            "Prenda detectada correctamente."
            if detectado
            else "No se encontro una prenda suficientemente parecida."
        )
    }
