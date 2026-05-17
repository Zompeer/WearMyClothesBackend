import os
import re
from io import BytesIO
from urllib.parse import parse_qs, urlparse

import numpy as np
import requests
import tensorflow as tf
from PIL import Image


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def extraer_id_google_drive(url):
    """
    Extrae el ID de un archivo de Google Drive desde formatos comunes.

    Soporta, por ejemplo:
    - https://drive.google.com/uc?id=ID
    - https://drive.google.com/uc?export=download&id=ID
    - https://drive.google.com/file/d/ID/view?usp=drivesdk
    - https://drive.google.com/open?id=ID

    Si la URL no parece ser de Google Drive o no contiene ID, retorna None.
    """

    if not isinstance(url, str):
        return None

    if "drive.google.com" not in url:
        return None

    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)

    if "id" in query_params and query_params["id"]:
        return query_params["id"][0]

    patrones = [
        r"/file/d/([^/]+)",
        r"/d/([^/]+)",
    ]

    for patron in patrones:
        match = re.search(patron, parsed.path)
        if match:
            return match.group(1)

    return None


def normalizar_url_google_drive(url):
    """
    Convierte URLs compartidas de Google Drive al formato directo esperado.

    Si la URL no es de Google Drive o no se puede extraer un ID, se retorna
    la URL original para no afectar URLs directas de imagen u otras fuentes.
    """

    archivo_id = extraer_id_google_drive(url)

    if archivo_id is None:
        return url

    return f"https://drive.google.com/uc?id={archivo_id}"


def _contenido_es_imagen(contenido):
    """Verifica si un contenido binario puede abrirse como imagen."""

    try:
        Image.open(BytesIO(contenido)).verify()
        return True
    except Exception:
        return False


def descargar_imagen_desde_url(url, mostrar_logs=False):
    """
    Descarga una imagen desde una URL.

    Funciona con:
    - URLs directas de imagen.
    - URLs de Google Drive tipo uc?id=.
    - URLs de Google Drive tipo file/d/.../view, transformandolas antes.
    """

    try:
        url_descarga = normalizar_url_google_drive(url)
        session = requests.Session()

        response = session.get(url_descarga, timeout=15)

        if response.status_code != 200:
            if mostrar_logs:
                print(f"Error descargando imagen. Codigo HTTP: {response.status_code}")
            return None

        content_type = response.headers.get("Content-Type", "")

        if mostrar_logs:
            print(f"URL original: {url}")
            print(f"URL usada: {url_descarga}")
            print(f"Content-Type recibido: {content_type}")

        # Si recibimos una imagen real o bytes que PIL puede abrir como imagen.
        if "image" in content_type or _contenido_es_imagen(response.content):
            return response.content

        # Google Drive a veces manda una pagina intermedia con cookie de confirmacion.
        confirm_token = None

        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                confirm_token = value
                break

        if confirm_token:
            separador = "&" if "?" in url_descarga else "?"
            url_confirmada = f"{url_descarga}{separador}confirm={confirm_token}"

            response = session.get(url_confirmada, timeout=15)

            if response.status_code != 200:
                if mostrar_logs:
                    print(f"Error descargando imagen confirmada. Codigo HTTP: {response.status_code}")
                return None

            content_type = response.headers.get("Content-Type", "")

            if mostrar_logs:
                print(f"Content-Type despues de confirmacion: {content_type}")

            if "image" in content_type or _contenido_es_imagen(response.content):
                return response.content

        if mostrar_logs:
            print("El link no devolvio una imagen real.")
            texto = response.text[:300] if hasattr(response, "text") else ""
            print("Primeros caracteres recibidos:")
            print(texto)

        return None

    except Exception as e:
        if mostrar_logs:
            print(f"Error descargando imagen desde URL: {e}")
        return None


def cargar_imagen(origen_imagen, mostrar_logs=False):
    """
    Carga una imagen desde diferentes fuentes.

    Para el flujo con Eduardo:
    - La imagen a escanear debe llegar directa como JPG, normalmente bytes.
    - Las imagenes de referencia pueden llegar como URLs de Google Drive.

    Tambien soporta rutas locales y objetos tipo archivo para pruebas/backend.
    """

    try:
        img = None

        # CASO 1: URL o ruta local.
        if isinstance(origen_imagen, str):
            if origen_imagen.startswith("http://") or origen_imagen.startswith("https://"):
                contenido = descargar_imagen_desde_url(
                    origen_imagen,
                    mostrar_logs=mostrar_logs
                )

                if contenido is None:
                    return None

                img = Image.open(BytesIO(contenido))
            else:
                img = Image.open(origen_imagen)

        # CASO 2: bytes de imagen recibidos desde camara/backend.
        elif isinstance(origen_imagen, (bytes, bytearray)):
            img = Image.open(BytesIO(origen_imagen))

        # CASO 3: objeto BytesIO u objeto tipo archivo con metodo read().
        elif hasattr(origen_imagen, "read"):
            posicion_actual = None

            if hasattr(origen_imagen, "tell"):
                try:
                    posicion_actual = origen_imagen.tell()
                except Exception:
                    posicion_actual = None

            contenido = origen_imagen.read()

            if hasattr(origen_imagen, "seek"):
                try:
                    origen_imagen.seek(posicion_actual or 0)
                except Exception:
                    pass

            img = Image.open(BytesIO(contenido))

        # CASO 4: imagen PIL ya cargada.
        elif isinstance(origen_imagen, Image.Image):
            img = origen_imagen

        else:
            if mostrar_logs:
                print(f"Tipo de imagen no soportado: {type(origen_imagen)}")
            return None

        # Convertir a RGB para evitar problemas con PNG, WEBP, transparencia, etc.
        if img.mode != "RGB":
            img = img.convert("RGB")

        # MobileNetV2 requiere imagenes de 224x224.
        img = img.resize((224, 224))

        # Convertir imagen a arreglo numerico.
        img = np.array(img)

        # Preprocesamiento requerido por MobileNetV2.
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

        # Agregar dimension de batch: (224, 224, 3) -> (1, 224, 224, 3).
        return np.expand_dims(img, axis=0)

    except Exception as e:
        if mostrar_logs:
            print(f"Error cargando imagen: {e}")
        return None


def generar_embedding(origen_imagen, mostrar_logs=False, modelo=None):
    """
    Genera un embedding a partir de una imagen.

    La imagen puede venir como:
    - bytes JPG desde el backend.
    - URL de referencia de Google Drive.
    - ruta local para pruebas.
    - objeto tipo archivo.
    - imagen PIL.
    """

    img = cargar_imagen(origen_imagen, mostrar_logs=mostrar_logs)

    if img is None:
        return None

    embedding = modelo.predict(img, verbose=0)
    print(embedding)
    return embedding[0]
