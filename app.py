from flask import Flask, jsonify, request
from flask_cors import CORS
import mysql.connector
import requests
from recomendation import recommend_outfit, update_user_model
from datetime import datetime, date, timedelta
import tensorflow as tf
from tensorflow.keras import Input
import recognition
import json
import os


modelo = None

app = Flask(__name__)
CORS(app)
def load_model():
    global modelo
    if modelo is None:
        inputs = Input(shape=(224,224,3))
        modelo = tf.keras.applications.MobileNetV2(
        weights=None,
        include_top=False,
        pooling="avg",
        input_tensor=inputs
    )
    modelo.load_weights("models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5")
    
def get_db():
    
    return mysql.connector.connect(
        host=os.environ["MYSQLHOST"],
        user=os.environ["MYSQLUSER"],
        password=os.environ["MYSQLPASSWORD"],
        database=os.environ["MYSQLDATABASE"],
        port=int(os.environ["MYSQLPORT"])
    )

@app.route("/health")
def ping():
    return jsonify({"message": "MySQL + Flask OK"})

@app.route("/getClothes", methods=["POST"])
def get_clothes():
    try:
        data = request.json or {}
        
        folder_id = data.get("folder_id")
        folder_id = "1" if folder_id is None else folder_id
        available = data.get("available")
        owner = data.get("owner", 1)
        
        conn = get_db()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM cloth WHERE folder_id=1 AND owner=%s", (owner,)) if available else cur.execute("SELECT * FROM cloth WHERE folder_id=%s AND owner=%s", (folder_id, owner))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return jsonify(rows), 200
    except Exception as e:
        print(e)
    return jsonify({}), 200

@app.route("/searchCloth", methods=["GET"])
def search_clothes():
    try:
        data = request.args or {}
        
        category: str = data.get("category")
        cloth_name = data.get("cloth_name")
        owner = data.get("owner")

        if category is None:
            return jsonify([]), 200
        
        conn = get_db()
        cur = conn.cursor(dictionary=True)
        if cloth_name:
            cur.execute("SELECT * FROM cloth WHERE categories=%s AND name COLLATE utf8_unicode_ci LIKE %s AND owner=%s ORDER BY name", (category, f"%{cloth_name}%", owner))
        else:
            cur.execute("SELECT * FROM cloth WHERE categories=%s AND owner=%s ORDER BY name", (category, owner))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return jsonify(rows), 200
    except Exception as e:
        print(e)
    return jsonify({}), 200

@app.route("/clothOffset", methods=["GET"])
def get_offset():
    try:
        data = request.args or {}

        owner = data.get("owner")
        cloth_id = data.get("cloth_id")
        cloth_category = data.get("cloth_category")
        cloth_category = cloth_category.replace("0", ".")

        conn = get_db()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT COUNT(*) AS offset FROM cloth WHERE owner=%s AND id < %s AND categories REGEXP %s ORDER BY id", (owner, cloth_id, f"^{cloth_category}$"))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        print(rows[0])
        return jsonify(rows[0])

    except Exception as e:
        print(e, owner, cloth_category)
        return jsonify({"offset": 0})

@app.route("/getCloth", methods=["POST"])
def get_cloth():
    try:
        data = request.json or {}
        
        cloth_id = data.get("cloth_id")
        cloth_category: str = data.get("cloth_category")
        cloth_category = cloth_category.replace("0", ".")
        offset = data.get("offset")
        owner = data.get("owner", 1)
        
        conn = get_db()
        cur = conn.cursor(dictionary=True)
        if cloth_id:
            cur.execute("SELECT * FROM cloth WHERE id=%s" , (cloth_id,))
            print(f"SELECT * FROM cloth WHERE id={cloth_id}")
        else:
            cur.execute("SELECT * FROM cloth WHERE categories REGEXP %s AND owner=%s ORDER BY id LIMIT 1 OFFSET %s" , (f'^{cloth_category}$', owner, offset))
            print("REQUEST IS",cloth_id)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        #print(rows)
        if rows[0]:
            rows[0]["offset"] = offset
        return jsonify(rows), 200
    except Exception as e:
        print(e, cloth_category, offset, owner)
    return jsonify({}), 200

@app.route("/createFolder", methods=["POST"])
def create_folder():
    data = request.json
    base_name = data.get("name")
    parent_id = data.get("parent_id")
    folder_type = data.get("type")
    owner = data.get("owner", 1)

    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    # 1️⃣ Buscar nombres similares
    cursor.execute("""
        SELECT name FROM folder
        WHERE name LIKE %s
        AND parent <=> %s AND owner=%s
    """, (f"{base_name}%", parent_id, owner))

    existing = cursor.fetchall()

    # 2️⃣ Si no existe, usar el nombre original
    if not existing and not (parent_id == None and base_name=="Desorganizado"):
        final_name = base_name
    else:
        # 3️⃣ Extraer números
        suffixes = []
        for row in existing:
            name = row["name"]
            if name == base_name:
                suffixes.append(0)
            elif name.startswith(base_name):
                try:
                    suffix = int(name.replace(base_name, ""))
                    suffixes.append(suffix)
                except:
                    pass

        # 4️⃣ Generar siguiente número
        final_name = f"{base_name}{max(suffixes, default=0) + 1}"

    # 5️⃣ Insertar
    try:
        cursor.execute(
            "INSERT INTO folder (name, parent, type, owner) VALUES (%s, %s, %s, %s)",
            (final_name, parent_id, 0 if folder_type is None else folder_type, owner)
        )
    except:
        print("final values", final_name, parent_id, 0 if folder_type is None else folder_type)
    conn.commit()
    newid = cursor.lastrowid
    cursor.close()
    conn.close()

    return jsonify({"newid": newid, "name": final_name, "parent": parent_id, "type": 0 if folder_type is None else folder_type}), 201

@app.route("/dropFolder", methods=["POST"])
def dropFolder():
    try:
        data = request.json or {}
        folder_id = data["id"]
        conn = get_db()
        cur = conn.cursor(dictionary=True)

        cur.execute("DELETE FROM folder WHERE id=%s", (folder_id,))
        cur.execute("UPDATE cloth SET folder_id=1 WHERE folder_id=NULL")
        conn.commit()
        cur.close()
        conn.close()
        
    except Exception as e:
        print(e)
    return jsonify({}), 200

@app.route("/folders", methods=["POST"])
def get_folders():
    try:
        data = request.json or {}
        parent = data.get("parent")
        owner = data.get("owner", 1)

        conn = get_db()
        cur = conn.cursor(dictionary=True)

        if parent:
            cur.execute(
                "SELECT name, id, type, image FROM folder WHERE parent = %s AND owner=%s",
                (parent, owner)
            )
        else:
            cur.execute(
                "SELECT name, id, type, image FROM folder WHERE (parent IS NULL and owner=%s) OR id=1"
            , (owner,))

        rows = cur.fetchall()
        if not parent and not rows:
            cur.execute("INSERT INTO folder(name, type, owner) VALUES (%s, %s, %s)", ("Desorganizado", 1, owner))
            conn.commit()
            cur.execute("SELECT name, id, type, image FROM folder WHERE parent IS NULL AND owner=%s", (owner,))
            rows = cur.fetchall()
        cur.close()
        conn.close()

        return jsonify(rows), 200

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/getFolder", methods=["GET"])
def get_folder():
    try:
        data = request.args or {}
        
        folder_id = data.get("folder_id")
        folder_id = folder_id if folder_id!="null" else "1"
        conn = get_db()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT name, image FROM folder WHERE id=%s", (folder_id,))
        
        rows = cur.fetchall()
        print(rows)
        cur.close()
        conn.close()
        return jsonify(rows), 200
    except Exception as e: 
        print(e)
        return jsonify({"error": "Error desconocido"}), 500

@app.route("/updateFolder", methods=["POST"])
def update_folder():
    try:
        data = request.json or {}
        name = data.get("name")
        id = data.get("id")
        

        conn = get_db()
        cur = conn.cursor(dictionary=True)

        # 1️⃣ Buscar nombres similares
        cur.execute("""
            SELECT name FROM folder
            WHERE name LIKE %s
            AND parent <=> (SELECT parent FROM folder WHERE id=%s)
        """, (f"{name}%", id))

        existing = cur.fetchall()

        # 2️⃣ Si existe, cancelar registro
        if existing:
            return jsonify({"modified": 0}) , 200

        cur.execute("UPDATE folder SET NAME=%s WHERE ID=%s", (name, id))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(name, id, e)
    return jsonify({"modified": 1}) , 200

@app.route("/setImage", methods=["POST"])
def setImage():
    try:
        data = request.json or {}
        id = data["id"]
        img_uri = data["uri"]

        conn = get_db()
        cur = conn.cursor(dictionary=True)

        cur.execute("UPDATE folder SET image=%s WHERE id=%s", (img_uri, id))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print()
    return jsonify({"statusChanged": True}), 200

@app.route("/permissions", methods=["POST"])
def request_token():
    try:
        data = request.json or {}
        refresh_token = data["refresh_token"]
        print(refresh_token)
    except:
        print()
    return refresh_access_token(refresh_token), 200

@app.route("/newCloth", methods=["POST"])
def add_cloth():
    load_model()
    try:
        data = request.json or {}

        name = data["name"]
        categories = data["categories"]
        owner = data["owner"]
        image = data["image"]

        conn = get_db()
        cur = conn.cursor(dictionary=True)
        embedding=None
        if(image):
            embedding = recognition.generar_embedding(image, modelo=modelo)
            embedding = json.dumps(embedding.tolist())
        cur.execute("INSERT INTO cloth(name, categories, owner, image, embedding) VALUES (%s, %s, %s, %s, %s)",
                    (name, categories, owner, image, embedding))
        
        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        print(e)
        return jsonify({"error": (name, categories, owner, image)}), 500
    return jsonify({"newid": cur.lastrowid}), 200

@app.route("/placeCloth", methods=["POST"])
def place_cloth():
    try:
        data = request.json or {}

        cloth_id = data["cloth_id"]
        folder_id = data["folder_id"]

        conn = get_db()
        cur = conn.cursor(dictionary=True)
        cur.execute("UPDATE cloth SET folder_id=NULL WHERE id=%s",
                    (cloth_id,)) if folder_id == 0 else cur.execute("UPDATE cloth SET folder_id=%s WHERE id=%s",
                                                                              (folder_id, cloth_id))
        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        print(e, "folderidis",folder_id)
        return jsonify({"error": "error while trying to place cloth into folder"}), 500
    return jsonify({"saved_in": folder_id}), 200

def refresh_access_token(refresh_token):
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "client_id": "381050945108-nh9001lmdfgiqkhptvbj63vbud495s7q.apps.googleusercontent.com",
        "client_secret": "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",
        "refresh_token": refresh_token,
        "grant_type": "refresh_token"
    }
    response = requests.post(token_url, data=data)
    return response.json()

@app.route("/deleteCloth", methods=["POST"])
def delete_cloth():
    try:
        data = request.json or {}

        cloth_id = data.get("cloth_id")
        conn= get_db()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT image FROM cloth WHERE id=%s", (cloth_id,))
        rows = cur.fetchall()
        cur.execute("DELETE FROM cloth WHERE id=%s", (cloth_id,))
        conn.commit()
        cur.close()
        conn.close()
        print(rows)
        return jsonify({"deleted_id": cloth_id, "file_url": rows[0]["image"]}), 200
    except Exception as e:
        print(e)
        return jsonify({"error": "Something went wrong"}), 500
    
@app.route("/alterCloth", methods=["POST"])
def alterCloth():
    load_model()
    try:
        data = request.json or {}

        cloth_id = data.get("cloth_id")
        dirtiness = data.get("dirtiness")
        image = data.get("image")
        name = data.get("name")
        categories = data.get("categories")

        conn = get_db()
        cur = conn.cursor(dictionary=True)
        if(dirtiness is not None):
            cur.execute("UPDATE cloth SET dirtiness=%s WHERE id=%s", (dirtiness, cloth_id))
        if(image is not None):
            embedding = recognition.generar_embedding(image, modelo=modelo)
            embedding = json.dumps(embedding)
            cur.execute("UPDATE cloth SET image=%s, embedding=%s WHERE id=%s", (image, embedding, cloth_id))
        if(name is not None):
            cur.execute("UPDATE cloth SET name=%s WHERE id=%s", (name, cloth_id))
        if(categories is not None):
            cur.execute("UPDATE cloth SET categories=%s WHERE id=%s", (categories, cloth_id))
        
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({"cloth_id": cloth_id, "changed_fields": [dirtiness is not None]})
    except Exception as e:
        print(e)
        return jsonify({"error": 500}), 500

@app.route("/useCloth", methods=["POST"])
def use_cloth():
    try:
        data = request.json or {}

        cloths = data["cloths"]

        ids = [item["id"] for item in cloths]

        print(ids)
        conn = get_db()
        cur = conn.cursor(dictionary=True)
        if cloths:
            cur.execute(f"UPDATE cloth SET folder_id=1, dirtiness=dirtiness+40, times_used=times_used+1, last_time_used=CURDATE() WHERE id IN ({','.join(['%s']*len(cloths))})", ids)
        else:
            cur.execute("UPDATE cloth SET folder_id=1")
        cur.execute("UPDATE cloth SET dirtiness=100 WHERE dirtiness>100")
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"cloths": cloths}), 200
    except Exception as e:
        print(e)
        return jsonify({"error": "Server error"}), 500

@app.route("/getRecommendation", methods=["GET"])
def get_recommendation():
    try:
        data = request.args or {}

        owner = data.get("user_id", default=1)
        threshold = float(data.get("threshold", 0.9))

        conn = get_db()
        cur = conn.cursor(dictionary=True)
        cur.execute("""
SELECT folder_id, name, image, owner, id, dirtiness, categories, CASE WHEN times_used < (SELECT AVG(times_used) FROM cloth WHERE owner=%s) THEN 0 ELSE 1 END AS times_used,
CASE WHEN DATEDIFF(last_time_used, CURDATE()) < (SELECT AVG(DATEDIFF(last_time_used, CURDATE())) FROM cloth WHERE owner=%s) THEN 0 ELSE 1 END 
AS last_time_used FROM cloth WHERE owner=%s
""", (owner, owner, owner))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        outfit = recommend_outfit(f"user_{owner}", rows, datetime.now().hour, threshold=threshold)
        print(outfit, "threshold", threshold)
        return jsonify({"recommendation": outfit}), 200

    except Exception as e:
        return jsonify({"error": e}), 500

@app.route("/trainModel", methods=["POST"])
def train_model():
    try:
        data = request.json or {}
        print(data)
        owner = data.get("user_id")
        outfit = data.get("outfit")
        like = data.get("like", 0)

        if not outfit:
            return jsonify({"warning": "empty request"}), 200
        
        conn = get_db()
        cur = conn.cursor(dictionary=True)
        cur.execute("""
SELECT id, dirtiness, categories, CASE WHEN times_used < (SELECT AVG(times_used) FROM cloth WHERE owner=%s) THEN 0 ELSE 1 END AS times_used,
CASE WHEN DATEDIFF(last_time_used, CURDATE()) < (SELECT AVG(DATEDIFF(last_time_used, CURDATE())) FROM cloth WHERE owner=%s) THEN 0 ELSE 1 END 
AS last_time_used FROM cloth WHERE id IN (%s)
""", (owner, owner, ",".join(map(str, outfit))))
        
        processed_outfit = cur.fetchall()
        cur.close()
        conn.close()

        update_user_model(f"user_{owner}", [processed_outfit], [like])
        print("Model trainend")
        return jsonify({"status": "model trained"})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.json or {}
        mail = data.get("mail")
        if not mail:
            return jsonify({"error": "missiing email"})

        conn = get_db()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM client WHERE mail=%s", (mail,))
        rows = cur.fetchall()
        if rows:# YA ESTA REGISTRADO
            cur.execute("UPDATE client SET last_session=%s WHERE mail=%s", (date.today(), mail))
            conn.commit()
            cur.close()
            conn.close()
            return jsonify({"user_id": rows[0].get("uuid"), "daily_login": rows[0].get("last_session") != date.today()})
        #REGISTRAR NUEVO USUARIO
        cur.execute("INSERT INTO client(mail, last_session) VALUES (%s, %s)", (mail, date.today()-timedelta(days=1)))
        newid = cur.lastrowid
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"user_id": newid, "daily_login": False})

    except Exception as e:
        print(e)
        return jsonify({"error": "Ocurrio un error"})

@app.route("/getStats", methods=["GET"])
def get_stats():
    try:
        data = request.args or {}

        owner = data.get("owner")

        conn = get_db()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT lvl, points FROM client WHERE uuid=%s", (owner,))
        row = cur.fetchall()[0]
        cur.close()
        conn.close()

        return jsonify(row)
    except Exception as e:
        print(e)
        return jsonify({"error": "No se pudo obtener las estadisticas"})

@app.route("/createTask", methods=["POST"])
def create_task():
    try:
        data = request.json or {}

        owner = data.get("owner")
        task = data.get("task")
        task_type = data.get("type")
        points = data.get("points")
        if(task_type == 0):
            exp_date = date.today() + timedelta(days=1)
        else:
            exp_date = date.today() + timedelta(days=7)

        conn = get_db()
        cur = conn.cursor(dictionary=True)
        cur.execute("INSERT INTO task(owner, task, points, task_type, exp_date) VALUES (%s, %s, %s, %s, %s)", 
                    (owner, task, points, task_type, exp_date))
        conn.commit()
        newid = cur.lastrowid
        cur.close()
        conn.close()

        return jsonify({"newid": newid}), 200
    except Exception as e:
        print(e)
        return jsonify({"error": "Error al crear la tarea"}), 500

@app.route("/getTasks", methods=["GET"])
def get_tasks():
    try:
        data = request.args or {}

        owner = data.get("owner")
        available = data.get("available", False)

        conn = get_db()
        cur = conn.cursor(dictionary=True)
        cur.execute("UPDATE task SET state=2 WHERE exp_date <= CURDATE() AND state=0")
        if available:
            cur.execute("SELECT * FROM task WHERE owner=%s AND exp_date > CURDATE()", (owner,))
        else:
            cur.execute("SELECT * FROM task WHERE owner=%s", (owner,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        return jsonify(rows), 200
    except Exception as e:
        print(e)
        return jsonify({"error": "Error al intentar obtener las tareas"}), 500

@app.route("/progressTask", methods=["POST"])
def progress_task():
    try:
        data = request.json or {}
        
        task_id = data.get("task")
        progress = data.get("progression")
        state = data.get("state", 0)

        conn = get_db()
        cur = conn.cursor(dictionary=True)
        cur.execute("UPDATE task SET progress=%s, state=%s WHERE id=%s", (progress, state, task_id))
        if state:
            cur.execute("SELECT owner, points FROM task WHERE id=%s", (task_id,))
            rows = cur.fetchall()[0]
            points = rows.get("points")
            owner = rows.get("owner")
            cur.execute("UPDATE client SET points=points+%s WHERE uuid=%s", (points, owner))
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({"changed": True}), 200
    except Exception as e:
        print(e)
        return jsonify({"error": "No fue posible progresar"})

@app.route("/levelUp", methods=["POST"])
def level_up():
    try:
        data = request.json or {}

        owner = data.get("owner")
        remaining = data.get("remaining", 0)

        conn = get_db()
        cur = conn.cursor()
        cur.execute("UPDATE client SET lvl=lvl+1, points=%s WHERE uuid=%s", (remaining, owner))
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"status", "ok"}), 200
    except Exception as e:
        print(e)
        return jsonify({"error": "Ocurrio un error al actualiizar"})

@app.route("/scanCloth", methods=["POST"])
def scan_test():
    load_model()
    try:
        owner = request.form.get("owner", None)
        if owner is None:
            return "No user id", 400
        if "cloth" not in request.files:
            return "No file", 400
        cloth = request.files.get("cloth")
        conn = get_db()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT id, embedding FROM cloth WHERE owner=%s", (owner,))
        cloths = cur.fetchall()
        match_cloth = recognition.escanear_prenda(cloth, cloths, mostrar_logs=True, modelo=modelo)
        print(match_cloth)
        match_id = match_cloth.get("top_matches", [{"id": None}])
        print(match_id)
        match_id = match_id[0].get("id", None) if len(match_id) > 0 else None
        print(match_id)
        if match_id is None:
            return jsonify({"match": {}}), 200
        cur.execute("SELECT id, name, categories, dirtiness, last_time_used, times_used, folder_id, owner, image FROM cloth WHERE id=%s", (match_id,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return jsonify({"match": rows[0]}), 200
    except Exception as e:
        print(e)
        return jsonify({"error": "No se encontraron resultados"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
