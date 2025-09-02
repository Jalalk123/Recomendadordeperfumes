import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify
from flask_cors import CORS
import os # Para verificar la existencia de archivos

app = Flask(__name__)
CORS(app) # Habilitar CORS para todas las rutas

# --- Carga de datos y Preprocesamiento (se ejecuta una sola vez al iniciar el servidor) ---
print("Cargando datos y preprocesando... Esto puede tomar un momento.")

df_fragrantica = None
df_stock = None
tfidf_matrix = None
stock_perfumes_set = None
fragrantica_perfumes_data = [] # Para almacenar datos de Fragrantica de forma optimizada

try:
    df_fragrantica = pd.read_excel("fra_cleaned.xlsx", usecols=["brand", "perfume", "notes"])
    df_fragrantica = df_fragrantica[df_fragrantica["notes"].notna()]
    df_fragrantica.reset_index(inplace=True, drop=True)
    print("DataFrame de Fragrantica cargado.")

    df_stock = pd.read_excel("Inventario_solonombres.xlsx")
    print("DataFrame de Stock cargado.")

    # Normalización de nombres
    def normalize_name(name):
        return str(name).lower().replace(" ", "-").replace("_", "-")

    df_fragrantica["perfume_normalized"] = df_fragrantica["perfume"].apply(normalize_name)
    df_fragrantica["brand_normalized"] = df_fragrantica["brand"].apply(normalize_name)

    df_stock["perfume_normalized"] = df_stock["perfume"].apply(normalize_name)
    df_stock["brand_normalized"] = df_stock["brand"].apply(normalize_name)

    # Crear un conjunto de perfumes en stock para una búsqueda eficiente
    stock_perfumes_set = set(zip(df_stock["brand_normalized"], df_stock["perfume_normalized"]))
    print("Conjunto de stock creado.")

    # Preprocesamiento de notas
    corpus_df = pd.DataFrame(df_fragrantica["notes"])
    itens_to_remove = [
        "[", "]", "\"", "{", "}",
        "middle: ", "top: ", "base: ", "null"
    ]
    def remove_items(text):
        for item in itens_to_remove:
            text = text.replace(item, "")
        return text

    corpus_df["notes"] = corpus_df["notes"].astype(str)
    corpus_df["notes"] = corpus_df["notes"].str.lower()
    corpus_df["notes"] = corpus_df["notes"].apply(remove_items)

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(corpus_df["notes"])
    print("Matriz TF-IDF calculada.")

    # Preparar una lista de diccionarios para un acceso más rápido en los endpoints
    # Esto también incluye un placeholder para imageUrl
    fragrantica_perfumes_data = df_fragrantica.apply(
        lambda row: {
            "index": row.name, # Mantener el índice original
            "brand": row["brand"],
            "perfume": row["perfume"],
            "brand_normalized": row["brand_normalized"],
            "perfume_normalized": row["perfume_normalized"],
            "notes": corpus_df.iloc[row.name]["notes"], # Usar las notas limpias
            # Placeholder para imagen, en un sistema real, esto vendría de tu DB
            "imageUrl": f"https://placehold.co/80x80/f7f7f7/6B46C1?text={row['perfume'].split(' ')[0]}"
        },
        axis=1
    ).tolist()
    print("Datos de perfumes de Fragrantica preparados para acceso rápido.")

except FileNotFoundError as e:
    print(f"Error: Uno de los archivos XLSX no se encontró. Asegúrate de que 'fra_cleaned.xlsx' e 'Inventario_solonombres.xlsx' estén en el mismo directorio. {e}")
    exit(1) # Salir si los archivos no se encuentran
except Exception as e:
    print(f"Ocurrió un error durante la carga o preprocesamiento de datos: {e}")
    exit(1)


# --- Tu función de recomendación (ligeramente adaptada para el backend) ---
def filter_by_perfume(selected_brand, selected_perfume):
    selected_brand_norm = normalize_name(selected_brand)
    selected_perfume_norm = normalize_name(selected_perfume)

    perfume_row_match = next((p for p in fragrantica_perfumes_data if
                             p["brand_normalized"] == selected_brand_norm and
                             p["perfume_normalized"] == selected_perfume_norm), None)

    if not perfume_row_match:
        return [] # Retorna lista vacía si no se encuentra el perfume

    perfume_index = perfume_row_match["index"]

    selected_perfume_vector = tfidf_matrix[perfume_index]
    similarities = cosine_similarity(selected_perfume_vector, tfidf_matrix).flatten()

    top_n_similar = 100
    
    # Crear una máscara para excluir el propio perfume
    mask = np.ones(len(similarities), dtype=bool)
    mask[perfume_index] = False
    
    # Obtener los índices de los perfumes más similares (excluyendo el propio perfume)
    # y mapearlos de vuelta a los índices originales del df_fragrantica
    similar_indices_all = np.argsort(similarities[mask])[-top_n_similar:][::-1]
    original_indices = np.arange(len(similarities))[mask][similar_indices_all]
    similar_values = similarities[original_indices]

    results = []
    for i, original_idx in enumerate(original_indices):
        perfume_data = fragrantica_perfumes_data[original_idx]
        normalized_perfume_key = f"{perfume_data['brand_normalized']}_{perfume_data['perfume_normalized']}"

        if normalized_perfume_key in stock_perfumes_set:
            results.append({
                "brand": perfume_data["brand"],
                "perfume": perfume_data["perfume"],
                "similarity": float(similar_values[i]), # Convertir a float para JSON
                "notes": perfume_data["notes"],
                "status": "En Stock",
                "imageUrl": perfume_data["imageUrl"]
            })

    # Ordenar por similitud de mayor a menor y tomar los 10 primeros
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:10]


# --- Endpoints de la API ---

@app.route('/brands', methods=['GET'])
def get_brands():
    """Endpoint para obtener todas las marcas únicas."""
    if df_fragrantica is None:
        return jsonify({"error": "Datos no cargados"}), 500
    unique_brands = sorted(list(df_fragrantica["brand"].unique()))
    return jsonify(unique_brands)

@app.route('/perfumes_by_brand/<string:brand_name>', methods=['GET'])
def get_perfumes_by_brand(brand_name):
    """Endpoint para obtener perfumes de una marca específica."""
    if df_fragrantica is None:
        return jsonify({"error": "Datos no cargados"}), 500
    # Usar el DataFrame de Fragrantica directamente para filtrar por la marca original
    perfumes = df_fragrantica[df_fragrantica["brand"] == brand_name]["perfume"].unique()
    return jsonify(sorted(list(perfumes)))

@app.route('/recommend', methods=['POST'])
def recommend_perfumes():
    """Endpoint para obtener recomendaciones de perfumes."""
    data = request.get_json()
    selected_brand = data.get('brand')
    selected_perfume = data.get('perfume')

    if not selected_brand or not selected_perfume:
        return jsonify({"error": "Marca y perfume son requeridos"}), 400

    print(f"Recibida solicitud para: {selected_brand} - {selected_perfume}")
    recommendations = filter_by_perfume(selected_brand, selected_perfume)
    print(f"Encontradas {len(recommendations)} recomendaciones.")
    return jsonify(recommendations)

if __name__ == '__main__':
    # Verificar si los archivos existen antes de intentar ejecutar la app
    if not os.path.exists("fra_cleaned.xlsx") or not os.path.exists("Inventario_solonombres.xlsx"):
        print("¡Advertencia! Asegúrate de que 'fra_cleaned.xlsx' e 'Inventario_solonombres.xlsx' estén en el mismo directorio que este script.")
        print("La aplicación no se iniciará sin estos archivos.")
    else:
        # Ejecutar el servidor Flask en modo de depuración para desarrollo.
        # En producción, usar un servidor WSGI como Gunicorn.
        app.run(debug=True, port=5000)
