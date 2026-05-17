import numpy as np
from tensorflow import keras
from datetime import datetime
import os
from dateutil import parser

BASE_MODEL_PATH = "models/base_model.h5"
USER_MODELS_DIR = "models/"

def encode_hour(hora: int) -> int:
    if hora < 12:
        return 0
    elif hora < 18:
        return 1
    else:
        return 2

def encode_dirtiness(dirtiness: int) -> int:
    if dirtiness < 33:
        return 0
    elif dirtiness < 66:
        return 1
    else:
        return 2

def encode_categories(cat_str: str) -> list[int]:
    return [int(c) for c in cat_str]  # "1001" -> [1,0,0,1]

def preprocess(cloth: dict, hora_del_dia: int) -> np.ndarray:
    return np.array([
        encode_hour(hora_del_dia),
        cloth["dirtiness"],
        cloth["times_used"],
        cloth["last_time_used"],
        *encode_categories(cloth["categories"])
    ])

def preprocess_outfit(clothes: list[dict], hora_del_dia: int) -> np.ndarray:
    vectors = [preprocess(c, hora_del_dia) for c in clothes]
    # Ejemplo: promedio de todas las prendas
    return np.mean(vectors, axis=0)



def build_model():
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(8,)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_base_model():
    model = build_model()

    # Ejemplo de outfits simulados
    outfit1 = [
        {"categories": "0100", "dirtiness": 0, "times_used": 1, "last_time_used": 1},
        {"categories": "0110", "dirtiness": 0, "times_used": 1, "last_time_used": 1},
        {"categories": "0001", "dirtiness": 0, "times_used": 1, "last_time_used": 1},
    ]
    outfit2 = [
        {"categories": "1000", "dirtiness": 0, "times_used": 1, "last_time_used": 1},
    ]

    hora_actual = datetime.now().hour
    X_base = np.array([
        preprocess_outfit(outfit1, hora_actual),
        preprocess_outfit(outfit2, hora_actual),
    ])
    y_base = np.array([1, 0])  # outfit1 recomendable, outfit2 no

    model.fit(X_base, y_base, epochs=50, verbose=0)

    os.makedirs(USER_MODELS_DIR, exist_ok=True)
    model.save(BASE_MODEL_PATH)
    return model



def load_user_model(user_id: str):
    """Carga el modelo de un usuario. Si no existe, copia el modelo base."""
    user_model_path = os.path.join(USER_MODELS_DIR, f"{user_id}_model.h5")
    if not os.path.exists(user_model_path):
        if not os.path.exists(BASE_MODEL_PATH):
            train_base_model()
        base_model = keras.models.load_model(BASE_MODEL_PATH)
        base_model.save(user_model_path)
    model = keras.models.load_model(user_model_path)
    # recompilar siempre al cargar
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# -------------------------------
# API de recomendación
# -------------------------------

def recommend_outfit(user_id: str, clothes: list[dict], hora_del_dia: int,
                     threshold: float = 0.6, max_per_category: dict = None) -> dict:
    if not user_id:
        return []
    model = load_user_model(user_id)
    outfit = {}
    category_names = ["head", "torso", "arms", "feet"]

    # valores por defecto si no se pasa max_per_category
    if max_per_category is None:
        max_per_category = {"head": 1, "torso": 1, "arms": 1, "feet": 1}

    for i, cat_name in enumerate(category_names):
        candidates = [c for c in clothes if c["categories"][i] == "1"]
        if not candidates:
            continue

        scored_candidates = []
        for c in candidates:
            score = model.predict(preprocess(c, hora_del_dia).reshape(1, -1))[0][0]
            if score >= threshold:
                scored_candidates.append((c, score))

        # Ordenar por score descendente
        scored_candidates.sort(key=lambda x: x[1], reverse=False)
        print("Candidates:", [s for c, s in scored_candidates])

        # Limitar según max_per_category
        outfit[cat_name] = [c for c, s in scored_candidates[:max_per_category[cat_name]]]

    return outfit



# -------------------------------
# API de actualización
# -------------------------------

def update_user_model(user_id: str, outfits: list[list[dict]], labels: list[int]):
    if not user_id:
        return {"satus": "model could not load"}
    model = load_user_model(user_id)
    hora_actual = datetime.now().hour

    X = [preprocess_outfit(outfit, hora_actual) for outfit in outfits]
    y = labels

    model.fit(np.array(X), np.array(y), epochs=10, verbose=0)
    user_model_path = os.path.join(USER_MODELS_DIR, f"{user_id}_model.h5")
    model.save(user_model_path)
    return {"status": f"model updated for user {user_id}"}




if __name__ == "__main__":
    train_base_model()
