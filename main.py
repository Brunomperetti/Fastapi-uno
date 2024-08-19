from fastapi import FastAPI, HTTPException
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Cargar los datasets
credits_df = pd.read_parquet('dataset/dataset_credits.parquet')
movies_df = pd.read_parquet('dataset/dataset_movies.parquet')

try:
    movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], errors='coerce')
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error al cargar el archivo CSV: {str(e)}")


@app.get("/Cantidad_estrenos_mes/", description= "Devuelve el numero de peliculas estrenadas en un mes especifico")
async def cantidad_estrenos_mes(mes: str= "Enero"):
    try:
        # Diccionario para convertir el nombre del mes en español a número
        months_dict = {
            'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
            'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
            'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
        }
        # Convierte el mes a su número correspondiente
        month_number = months_dict.get(mes.lower())
        if month_number is None:
            raise HTTPException(status_code=400, detail="Mes no válido")

        # Filtrar el DataFrame por el mes
        cantidad = movies_df[movies_df['release_date'].dt.month == month_number].shape[0]
        return {f"{cantidad} películas fueron estrenadas en el mes de {mes}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"El mes debe estar escrito en español. No se aceptan números: {str(e)}")



@app.get("/cantidad_estrenos_dia/", description= "Proporciona la cantidad de peliculas estrenadas en un día especifico")
def cantidad_estrenos_dia(dia: str= "Lunes"):
    # Diccionario para traducir los días de español a números
    dias_espanol = {
        'lunes': 0, 'martes': 1, 'miércoles': 2, 'jueves': 3,
        'viernes': 4, 'sábado': 5, 'domingo': 6
    }

    # Convertir el día ingresado a número
    dia_num = dias_espanol.get(dia.lower())

    if dia_num is None:
        return {"El Día esta mal escrito o no es válido. Por favor ingrese un día en español, recuerde usar los acentos."}

    # Filtrar las películas que fueron estrenadas en ese día de la semana
    cantidad = movies_df[movies_df['release_date'].dt.dayofweek == dia_num].shape[0]

    return f"{cantidad} cantidad de películas fueron estrenadas en los días {dia.capitalize()}."


@app.get("/score_titulo/", description= "Devuelve el año de estreno y la popularidad de una película según su título")
def popularidad_titulo(titulo: str= "Toy Story"):
    # Filtrar la película por título
    pelicula = movies_df[movies_df['title'].str.lower() == titulo.lower()]
    
    if pelicula.empty:
        return {"Película no encontrada"}
    
    titulo = pelicula['title'].values[0]
    anio = pelicula['release_year'].values[0]
    score = pelicula['popularity'].values[0]
    
    return f"La película {titulo} fue estrenada en el año {anio} con un score/popularidad de {score}."


@app.get("/votos_titulo/", description= "Devuelve el año de estreno y la cantidad total de votos de una película basada en su título")
def votos_titulo(titulo: str= "Titanic"):
    # Filtrar la película por título
    pelicula = movies_df[movies_df['title'].str.lower() == titulo.lower()]
    
    if pelicula.empty:
        return {"Película no encontrada."}
    
    titulo = pelicula['title'].values[0]
    votos = pelicula['vote_count'].values[0]
    promedio = pelicula['vote_average'].values[0]
    
    if votos < 2000:
        return {"La película no cumple con el mínimo de 2000 valoraciones."}
    
    return f"La película {titulo} fue estrenada en el año {pelicula['release_year'].values[0]}. La misma cuenta con un total de {votos} valoraciones, con un promedio de {promedio}."


# Realizar el merge de los DataFrames usando la columna común
merged_df = pd.merge(credits_df, movies_df, on='id')

@app.get("/get_actor/", description= "Devuelve la cantidad de películas, la recaudación total y el promedio de recaudación de un actor específico")
def get_actor(nombre_actor: str= "Jim Carrey"):
    # Filtrar el DataFrame por el nombre del actor en la columna 'cast_name'
    actor_df = merged_df[merged_df['cast_name'] == nombre_actor]
    
    # Verificar si el actor tiene películas en el DataFrame
    if actor_df.empty:
        return {f"Actor: {nombre_actor} no encontrado. Verificá que el nombre y apellido estén con la primera letra en mayúscula y con acentos correctos."}

    # Calcular la cantidad de películas
    num_peliculas = actor_df.shape[0]
    
    # Calcular el retorno total y promedio
    retorno_total = actor_df['return'].sum()
    retorno_promedio = actor_df['return'].mean()

    # Generar la respuesta
    return {
        "Actor": nombre_actor,
        "Cantidad_peliculas": num_peliculas,
        "Recaudación_total": round(retorno_total, 2),
        "Recaudación_promedio": round(retorno_promedio, 2)
    }


@app.get("/get_director/", description= "Devuelve la lista de películas dirigidas por un director específico, incluyendo la fecha de lanzamiento, el retorno, el costo y la ganancia. También muestra el éxito total en términos de retorno de todas las películas del director.")
def get_director(nombre_director: str= "Steven Spielberg"):
    # Filtrar el DataFrame para obtener solo registros donde el director es el solicitado
    director_df = merged_df[(merged_df['crew_name'] == nombre_director) & (merged_df['crew_job'] == 'Director')]
    
    # Verificar si el director tiene películas en el DataFrame
    if director_df.empty:
        return {f"Director: {nombre_director} no encontrado. Verificá que el nombre y apellido estén bien escritos y con la primera letra en mayúscula"}

    # Seleccionar las columnas relevantes y renombrar para claridad
    peliculas_info = director_df[['original_title', 'release_date', 'return', 'budget', 'revenue']].copy()
    peliculas_info.columns = ['Título', 'Fecha de lanzamiento', 'Retorno', 'Costo', 'Ganancia']
    
    # Calcular el éxito total
    exito_total = peliculas_info['Retorno'].sum()

    # Convertir el DataFrame a un diccionario para la respuesta
    peliculas_lista = peliculas_info.to_dict(orient='records')

    # Generar la respuesta
    return {
        "director": nombre_director,
        "exito_total": round(exito_total, 2),
        "peliculas": peliculas_lista
    }

# Preparar el modelo de recomendación
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['overview'].fillna(''))

def recomendar_peliculas(titulo_pelicula, num_recomendaciones=5):
    # Verificar si la película está en el dataset
    if titulo_pelicula not in movies_df['title'].values:
        raise HTTPException(status_code=404, detail="Película no encontrada, verifique que la palabra esté escrita correctamente con la primera letra en mayúscula.")
    
    # Obtener el índice de la película
    idx = movies_df[movies_df['title'] == titulo_pelicula].index[0]
    
    # Calcular la similitud con todas las demás películas
    cosine_similarities = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
    
    # Obtener los índices de las películas más similares
    similar_indices = cosine_similarities.argsort()[-num_recomendaciones-1:-1][::-1]
    
    # Obtener los nombres de las películas más similares
    similar_movies = movies_df['title'].iloc[similar_indices].tolist()
    return similar_movies

@app.get("/Recomendaciones_de_peliculas/",description= "Devuelve una listas de peliculas recomendadas basadas en la similitud del titulo")
def recomendacion(titulo_pelicula: str= "Dracula"):
    # Obtener las recomendaciones
    similar_movies = recomendar_peliculas(titulo_pelicula)
    
    return {"Te recomendamos las siguientes películas": similar_movies}
